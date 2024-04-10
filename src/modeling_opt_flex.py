from typing import List, Optional, Tuple, Union
import copy
import torch
import heapq, time
from torch import Tensor
from typing import Callable
from collections import OrderedDict
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.opt.configuration_opt import OPTConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "facebook/opt-350m"
_CONFIG_FOR_DOC = "OPTConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/opt-125m",
    "facebook/opt-350m",
    "facebook/opt-1.3b",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "facebook/opt-13b",
    "facebook/opt-30b",
    # See all OPT models at https://huggingface.co/models?filter=opt
]


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class OPTLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # OPT is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, attention_mask: torch.LongTensor, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        attention_mask = attention_mask.long()

        # create positions depending on attention_mask
        positions = (torch.cumsum(attention_mask, dim=1).type_as(attention_mask) * attention_mask).long() - 1

        # cut positions if `past_key_values_length` is > 0
        positions = positions[:, past_key_values_length:]

        return super().forward(positions + self.offset)


class OPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class OptFlashAttention2(OPTAttention):
    """
    OPT flash attention module. This module inherits from `OPTAttention` as the weights of the module stays untouched.
    The only required change would be on the forward pass where it needs to correctly call the public API of flash
    attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, _, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states)
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        query_length = query_states.shape[1]
        tgt_len = key_states.shape[-2]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        query_states = query_states.view(bsz, query_length, self.num_heads, self.head_dim)
        key_states = key_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)
        value_states = value_states.transpose(1, 2).view(bsz, tgt_len, self.num_heads, self.head_dim)

        attn_dropout = self.dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, query_length, dropout=attn_dropout
        )

        attn_weights_reshaped = attn_output.reshape(bsz, query_length, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_weights_reshaped)

        if not output_attentions:
            attn_weights_reshaped = None

        return attn_output, attn_weights_reshaped, past_key_value

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


OPT_ATTENTION_CLASSES = {
    "eager": OPTAttention,
    "flash_attention_2": OptFlashAttention2,
}


class OPTDecoderLayer(nn.Module):
    def __init__(self, config: OPTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size

        self.self_attn = OPT_ATTENTION_CLASSES[config._attn_implementation](config=config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`OPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTPreTrainedModel(PreTrainedModel):
    config_class = OPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
        head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
            Mask to nullify selected heads of the attention modules in the encoder. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


class OPTDecoder(OPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)
class OPTModel(OPTPreTrainedModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = OPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return BaseModelOutputWithPast(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
        )


class OPTForCausalLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


class FlexAttention(nn.Module):
    def __init__(
        self,
        config: OPTConfig,
        is_decoder: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config

        def _handle_deprecated_argument(config_arg_name, config, fn_arg_name, kwargs):
            """
            If a the deprecated argument `fn_arg_name` is passed, raise a deprecation
            warning and return that value, otherwise take the equivalent config.config_arg_name
            """
            val = None
            if fn_arg_name in kwargs:
                logging.warning(
                    "Passing in {} to {self.__class__.__name__} is deprecated and won't be supported from v4.38."
                    " Please set it in the config instead"
                )
                val = kwargs.pop(fn_arg_name)
            else:
                val = getattr(config, config_arg_name)
            return val

        self.embed_dim = _handle_deprecated_argument("hidden_size", config, "embed_dim", kwargs)
        self.num_heads = _handle_deprecated_argument("num_attention_heads", config, "num_heads", kwargs)
        self.dropout = _handle_deprecated_argument("attention_dropout", config, "dropout", kwargs)
        self.enable_bias = _handle_deprecated_argument("enable_bias", config, "bias", kwargs)

        self.head_dim = self.embed_dim // self.num_heads
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.k_hr_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_hr_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous().requires_grad_(True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        hr_key_value_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            # key_states = self.k_proj(hidden_states)
            # value_states = self.v_proj(hidden_states)
            if hr_key_value_states is not None:
                ## TODO: inference
                
                # hr_key_value_states.requires_grad_(True)
                ## hr_key_value_states [bs, text_seq_len, dense_token_num, dim]
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
                dense_token_num = hr_key_value_states.shape[2]
                
                hr_attention_mask = torch.ones((bsz, hr_key_value_states.shape[1], hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device)*torch.finfo(hr_key_value_states.dtype).min
                ## 每个text token只能跟自己选出来的dense scene token进行交互
                ## pad的text token也需要mask
                batch_valid_len_w_eos = (attention_mask.squeeze(1)[:, -1, :] == 0).sum(dim=1) - self.config.scene_token_num
                for bs in range(bsz):
                    valid_length = batch_valid_len_w_eos[bs].item()
                    for j in range(valid_length):
                        start_index = j * dense_token_num
                        end_index = (j+1) * dense_token_num
                        hr_attention_mask[bs, j, start_index:end_index] = 0.
                ## 最后为scene token加上attn mask，即每个scene token都可以于dense region token交互
                scene_token_attention_mask = torch.zeros((bsz, self.config.scene_token_num, hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device)
                hr_attention_mask = torch.cat([scene_token_attention_mask, hr_attention_mask], dim=1)
                ## concat 到原来的attention mask上
                attention_mask = torch.cat([attention_mask, hr_attention_mask.unsqueeze(1)], dim=-1)
                hr_key_value_states = hr_key_value_states.view(bsz, -1, hidden_states.shape[-1]).contiguous()
                hr_key_states = self._shape(self.k_hr_proj(hr_key_value_states), -1, bsz)
                hr_value_states = self._shape(self.v_hr_proj(hr_key_value_states), -1, bsz)
                key_states = torch.cat([key_states, hr_key_states], dim=2)
                value_states = torch.cat([value_states, hr_value_states], dim=2)
            
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            
        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )
        
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_bf_sm = torch.clone(attn_weights)
        # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        if attn_weights.dtype == torch.float16:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(torch.float16)
        else:
            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)
        
        # ## TODO：check inference
        # if hr_key_value_states is not None:        ## [bs, text_seq_len, 4*32, 2048]
        #     dense_scene_tokens_num = hr_key_value_states.shape[2]
        #     src_len = key_states.size(1) + dense_scene_tokens_num
        #     ## 计算dense_scene_tokens的attention
        #     for text_query_token_id in range(query_states.shape[1]-self.config.scene_token_num):
                
        #         if not self.training:
        #             text_query_token_id=query_states.shape[1]-self.config.scene_token_num-1
                
        #         query_state_per_token = query_states[:, text_query_token_id+ self.config.scene_token_num, :].unsqueeze(1)
        #         attn_weight_per_token = attn_weights[:, text_query_token_id+ self.config.scene_token_num, :].unsqueeze(1)
        #         attn_mask_per_token = attention_mask[:, :, text_query_token_id+ self.config.scene_token_num, :].unsqueeze(2)
                
        #         flex_key_state_per_token = self._shape(hr_key_states[:, text_query_token_id, :].unsqueeze(1), -1, bsz).view(*proj_shape)
                
        #         flex_attn_weight_per_token = torch.bmm(query_state_per_token, flex_key_state_per_token.transpose(1,2))
        #         flex_attn_mask_per_token = torch.zeros(bsz, 1, 1, dense_scene_tokens_num, device=attn_mask_per_token.device, dtype=attn_mask_per_token.dtype)
                
        #         attn_weight_per_token = torch.cat([attn_weight_per_token, flex_attn_weight_per_token], dim=-1)
        #         attn_mask_per_token = torch.cat([attn_mask_per_token, flex_attn_mask_per_token], dim=-1)

        #         flex_value_states_per_token = self._shape(hr_value_states[:, text_query_token_id, :].unsqueeze(1), -1, bsz).view(*proj_shape)
        #         value_states_per_token = torch.cat([value_states, flex_value_states_per_token], dim=1)
                
        #         if attn_weight_per_token.size() != (bsz * self.num_heads, 1, src_len):
        #             raise ValueError(
        #                 f"Attention weights should be of size {(bsz * self.num_heads, 1, src_len)}, but is"
        #                 f" {attn_weight_per_token.size()}"
        #             )
            
        #         if attn_mask_per_token.size() != (bsz, 1, 1, src_len):
        #             raise ValueError(
        #                 f"Attention mask should be of size {(bsz, 1, 1, src_len)}, but is {attn_mask_per_token.size()}"
        #             )
        #         attn_weight_per_token = attn_weight_per_token.view(bsz, self.num_heads, 1, src_len) + attn_mask_per_token
        #         attn_weight_per_token = torch.max(
        #             attn_weight_per_token, torch.tensor(torch.finfo(attn_weight_per_token.dtype).min, device=attn_weight_per_token.device)
        #         )
        #         attn_weight_per_token = attn_weight_per_token.view(bsz * self.num_heads, 1, src_len)

        #         # attn_weights_bf_sm[:, text_query_token_id, :] = attn_weight_per_token[:, :, :key_states.size(1)]
        #         # upcast to fp32 if the weights are in fp16. Please see https://github.com/huggingface/transformers/pull/17437
        #         if attn_weight_per_token.dtype == torch.float16:
        #             attn_weight_per_token = nn.functional.softmax(attn_weight_per_token, dim=-1, dtype=torch.float32).to(torch.float16)
        #         else:
        #             attn_weight_per_token = nn.functional.softmax(attn_weight_per_token, dim=-1)

        #         if layer_head_mask is not None:
        #             if layer_head_mask.size() != (self.num_heads,):
        #                 raise ValueError(
        #                     f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
        #                     f" {layer_head_mask.size()}"
        #                 )
        #             attn_weight_per_token = layer_head_mask.view(1, -1, 1, 1) * attn_weight_per_token.view(bsz, self.num_heads, 1, src_len)
        #             attn_weight_per_token = attn_weight_per_token.view(bsz * self.num_heads, 1, src_len)

        #         attn_probs_per_token = nn.functional.dropout(attn_weight_per_token, p=self.dropout, training=self.training)

        #         attn_output_per_token = torch.bmm(attn_probs_per_token, value_states_per_token)
                
        #         attn_output[:, text_query_token_id, :] = attn_output_per_token.squeeze(1)
        #         attn_weights[:, text_query_token_id, :] = attn_weight_per_token[:, :, :key_states.size(1)].squeeze(1)
                
        #         if not self.training:
        #             break
            
        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned aross GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)
        
        src_len = key_states.size(1)
        multi_head_attn_weights = attn_weights_bf_sm.view(bsz, self.num_heads, tgt_len, src_len)
        global_attention_map = torch.sum(multi_head_attn_weights, dim=1).detach()
        scene_token_hidden_state = global_attention_map[:, self.config.scene_token_num:, :][:, :, :self.config.scene_token_num] #* self.config.attn_softmax_scale
        scene_token_hidden_state = nn.functional.softmax(scene_token_hidden_state, dim=-1)
        
        ## text token中会有pad token，根据attn mask处理
        # batch_valid_len_w_eos = (attention_mask.squeeze(1)[:, -1, :] == 0).sum(dim=1) - self.config.scene_token_num
        # scene_token_hidden_state = [hs[:batch_valid_len_w_eos[hi].item()] for hi,hs in enumerate(scene_token_hidden_state)]
        
        return attn_output, attn_weights_reshaped, past_key_value, scene_token_hidden_state


from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes

from third_party.pointnet2.pointnet2_utils import ball_query, grouping_operation
class Dense_Region_Selector(nn.Module):
    def __init__(self, config, select_threshold=0.5):
        super().__init__()
        self.select_threshold = select_threshold
        self.region_radius = config.region_radius
        self.region_nsample = config.region_sample_num
        self.hidden_dim = config.word_embed_proj_dim
        ## +3 means concat xyz
        # mlp_dims = [768+3, 2048]
        # self.pcd_tokenizer = PointnetSAModuleVotes(
        #     radius=config.region_token_sample_radius,
        #     nsample=config.region_token_sample_num_per_ball,
        #     npoint=config.region_token_sample_num,
        #     mlp=mlp_dims,
        #     normalize_xyz=True,
        # )
        
    def forward(self, scene_token_hidden_state, dense_region_tokens):
        ## [[bs],[idx]]
        # interest_tokens = torch.where(last_sparse_scene_token_hidden_state > self.select_threshold)
        # if len(interest_tokens[0]) == 0:
        ## 选4个版本
        k = 4
        top_values, top_indices = torch.topk(scene_token_hidden_state, k, dim=-1)
        batch_select_dense_region_tokens = []
        for bs in range(len(top_indices)):
            ## [dense_token_num, hidden_dim]
            select_dense_region_tokens = torch.stack([dense_region_tokens[bs, text_token_i, :, :].view(-1, self.hidden_dim) for text_token_i in top_indices[bs]],dim=0)
            batch_select_dense_region_tokens.append(select_dense_region_tokens)
        batch_select_dense_region_tokens = torch.stack(batch_select_dense_region_tokens, dim=0)
        # batch_select_dense_region_tokens = self.scene_token_head(batch_select_dense_region_tokens)
        return batch_select_dense_region_tokens

        ## argmax版本 
        # max_indices = torch.argmax(last_sparse_scene_token_hidden_state, dim=-1)
        # interest_tokens = [torch.tensor(range(len(max_indices))).to(max_indices.device), max_indices]
        # interest_xyz_idx = {i:[] for i in range(len(last_sparse_scene_token_hidden_state))}
        # for i in range(len(interest_tokens[0])):
        #     interest_xyz_idx[interest_tokens[0][i].item()].append(interest_tokens[1][i].item())
        # interest_xyz = [sparse_scene_xyz[k,v,:] for k,v in interest_xyz_idx.items()]
        # interest_xyz = torch.stack(interest_xyz, dim=0)
        # ## 感兴趣xyz的半径圆周范围self.region_radius内选self.region_nsample个点作为这个区域的稠密点云表示
        # idx = ball_query(self.region_radius, self.region_nsample, dense_scene_xyz, interest_xyz).long().squeeze(1)
        # ## 这里要保证idx的最大值小于dense_scene_xyz的长度由于现在的点云都是pad过的
        # for ii, id in enumerate(idx):
        #     assert id.max().item() < valid_pcd_len[ii]
        # interest_dense_scene_xyz = torch.stack([dense_scene_xyz[i, idx[i], :] for i in range(len(idx))], dim=0)
        # interest_dense_scene_fts = torch.stack([dense_scene_fts[i, idx[i], :] for i in range(len(idx))], dim=0)
        # interest_dense_scene_fts = torch.cat([interest_dense_scene_fts, interest_dense_scene_xyz], dim=-1).transpose(1, 2).contiguous()
        # ## 选择一定点聚合周围特征作为区域的表示
        # pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pcd_tokenizer(interest_dense_scene_xyz, interest_dense_scene_fts)
        # return pre_enc_features.transpose(1, 2).contiguous()

        
class FlexOPTDecoderLayer(OPTDecoderLayer):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.embed_dim = config.hidden_size

        self.self_attn = FlexAttention(config=config, is_decoder=True)

        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)

        self.dense_region_selector = Dense_Region_Selector(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        hr_key_value_states: Optional[torch.Tensor] = None,
        dense_pcd_info = None
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`, *optional*): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        ## Donot need attn mask as need to compute visual token how importance to text token
        hidden_states, self_attn_weights, present_key_value, scene_token_hidden_state = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            hr_key_value_states=hr_key_value_states,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)

        # 350m applies layer norm AFTER attention
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
            
        hr_key_value_states = self.dense_region_selector(scene_token_hidden_state, dense_pcd_info['dense_region_tokens'])

        return outputs, hr_key_value_states
    

class FlexOPTDecoder(OPTDecoder):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = nn.Linear(config.hidden_size, config.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        if config.use_flex_layer:
            self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
            self.flex_layers = nn.ModuleList([FlexOPTDecoderLayer(config) for _ in range(config.num_flex_hidden_layers)])
        else:
            print("warning!!! do not use flex layer")
            self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers+config.num_flex_hidden_layers)])
            self.flex_layers = []
        
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        # self.gradient_checkpointing = True
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        dense_pcd_info = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        ## TODO
        # if not self.training:
        #     past_key_values = None
        #     attention_mask = None
        
        batch_size, seq_length = input_shape
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
            elif attention_mask.shape[1] != mask_seq_length:
                raise ValueError(
                    f"The provided attention mask has length {attention_mask.shape[1]}, but its length should be "
                    f"{mask_seq_length} (sum of the lengths of current and past inputs)"
                )
            causal_attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, input_shape, inputs_embeds, past_key_values_length
            )

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        if self.project_in is not None:
            inputs_embeds = self.project_in(inputs_embeds)

        hidden_states = inputs_embeds + pos_embeds
        

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        use_cache = False
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        hr_key_value_states = None
        for idx, decoder_layer in enumerate(self.flex_layers):
            torch.cuda.empty_cache()
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                    hr_key_value_states,
                    dense_pcd_info
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    hr_key_value_states=hr_key_value_states,
                    dense_pcd_info=dense_pcd_info
                )

            
            hr_key_value_states = layer_outputs[-1]
            layer_outputs = layer_outputs[0]
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        
        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class FlexOPTModel(OPTModel):
    def __init__(self, config: OPTConfig):
        super().__init__(config)
        self.decoder = FlexOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


from models.ll3da.position_embedding import PositionEmbeddingCoordsSine
class PromptEncoder(nn.Module):
    def __init__(self):
        super(PromptEncoder, self).__init__()
        
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
        )
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=256, 
            pos_type='fourier', 
            normalize=True
        )
        
        
    def forward(self, 
        point_cloud_dims,
        click_query=None,
    ):
        
        # click prompt encoding: batch x nquery x nproposal
        click_xyz = click_query     # batch x nquery x 3
        ## TODO: check point_cloud_dims
        click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)
        click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))

        return click_prompt
    

class FlexOPTForCausalLM(OPTForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = FlexOPTModel(config)
        self.prompt_encoder = PromptEncoder()
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.scene_token_in_head = nn.Linear(768+3, config.hidden_size, bias=False)
        # from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
        ## +3 means concat xyz
        # mlp_dims = [768+3, 2048]
        # self.pcd_tokenizer = PointnetSAModuleVotes(
        #     radius=config.scene_token_sample_radius,
        #     nsample=config.scene_token_sample_num_per_ball,
        #     npoint=config.scene_token_num,
        #     mlp=mlp_dims,
        #     normalize_xyz=True,
        # )
        # Initialize weights and apply final processing
        self.post_init()
        
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids=None,
        # is_eval=False,
        # task_name=None,
        attention_mask=None,
        gradient_mask=None,
        batch_data_label=None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
            
        if inputs_embeds is None:
            inputs_embeds = self.model.decoder.embed_tokens(input_ids)
        
        ## Only for inference(bs 1)    
        if attention_mask is None or not self.training:
            if not past_key_values is None:
                attention_mask = torch.ones((inputs_embeds.shape[0], self.prefix_len+attention_mask.shape[-1]), device=inputs_embeds.device)
            else:
                attention_mask = torch.ones_like(inputs_embeds[..., 0], device=inputs_embeds.device)
                
                
        ## get sparse scene object tokens from openscene_sparse_fts by set abstract layer here
        if batch_data_label is None:
            batch_data_label = copy.deepcopy(self.batch_data_label)
        pre_enc_features = self.scene_token_in_head(batch_data_label['scene_tokens'])
        batch_data_label['dense_region_tokens'] = self.scene_token_in_head(batch_data_label['dense_region_tokens'])
        # features = batch_data_label['openscene_sparse_fts']
        # features = features.transpose(1, 2).contiguous()
        # xyz = batch_data_label['openscene_sparse_pcd']
        # pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pcd_tokenizer(xyz, features)
        # pre_enc_features = pre_enc_features.transpose(1, 2).contiguous()
        pre_enc_features_mask = torch.ones_like(pre_enc_features[..., 0])
        
        if past_key_values is None :
            if batch_data_label['click_mask'][0, 0].item() == 1:
                click_query = batch_data_label['click_query'][:, 0, :]
                click_query = click_query.unsqueeze(1)
                point_cloud_dims = [
                    batch_data_label["point_cloud_dims_min"],
                    batch_data_label["point_cloud_dims_max"],
                ]
                prompt_embeds = self.prompt_encoder(point_cloud_dims, click_query)
                prompt_mask = torch.ones_like(prompt_embeds[..., 0])
                
                inputs_embeds = torch.cat([pre_enc_features, prompt_embeds, inputs_embeds], dim=1)
                attention_mask = torch.cat([pre_enc_features_mask, prompt_mask, attention_mask], dim=1)
                self.prefix_len = pre_enc_features.shape[1] + prompt_embeds.shape[1]
            else:
                inputs_embeds = torch.cat([pre_enc_features, inputs_embeds], dim=1)
                attention_mask = torch.cat([pre_enc_features_mask, attention_mask], dim=1)
                self.prefix_len = pre_enc_features.shape[1]
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            dense_pcd_info={
                'dense_region_tokens':batch_data_label['dense_region_tokens'],
            }
            # dense_pcd_info={
            #     'sparse_scene_xyz':pre_enc_xyz,
            #     'dense_scene_fts':batch_data_label['openscene_dense_fts'],
            #     'pre_enc_inds':pre_enc_inds,
            #     'dense_scene_xyz':batch_data_label['openscene_dense_pcd'],
            #     'valid_pcd_len': batch_data_label['valid_pcd_len']
            # }
        )

        logits = self.lm_head(outputs[0])
        # labels = batch_data_label['input_ids']

        loss = None
        if labels is not None:
            assert gradient_mask.shape[1] == logits[:, self.prefix_len-1:-1, :].shape[1]
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[:, self.prefix_len-1:-1, :].contiguous()
            shift_labels = labels.contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            gradient_mask = gradient_mask.contiguous().view(-1)
            loss = torch.sum(loss * gradient_mask) / torch.sum(gradient_mask + 1e-6)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def set_batch_data_label_cache(self, batch_data_label):
        self.batch_data_label = batch_data_label

    def forward_preprocess_scene_token(
        self,
        batch_data_label,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        import os
        self.scan_name_list = os.listdir('/mnt/nfs/share/Adaptive/openscene_scene_tokens')
        scan_name = batch_data_label['scan_name'][0]
        ot_dir = f'/mnt/nfs/share/Adaptive/openscene_scene_tokens/{scan_name}'
        if not os.path.exists(ot_dir):
            os.makedirs(ot_dir, exist_ok=True)
        from third_party.pointnet2.pointnet2_modules import  PointnetSAModuleVotes_WoMlp
        from third_party.pointnet2.pointnet2_utils import ball_query
        from tqdm import tqdm
        
        
        if scan_name not in self.scan_name_list:
            self.scan_name_list.append(scan_name)
        else:
            return 
        
        scene_tokenizer = PointnetSAModuleVotes_WoMlp(
            radius=self.config.scene_token_sample_radius,
            nsample=self.config.scene_token_sample_num_per_ball,
            npoint=self.config.scene_token_num,
            normalize_xyz=True,
        )
        region_tokenizer = PointnetSAModuleVotes_WoMlp(
            radius=self.config.region_token_sample_radius,
            nsample=self.config.region_token_sample_num_per_ball,
            npoint=self.config.region_token_sample_num,
            normalize_xyz=True,
        )
        region_radius = self.config.region_radius
        region_nsample = self.config.region_sample_num
        

        ## concat xyz with fts
        features = batch_data_label['openscene_sparse_fts']
        dense_scene_xyz = batch_data_label['openscene_dense_pcd']
        dense_scene_fts = batch_data_label['openscene_dense_fts']
        
        ## get sparse scene object tokens from openscene_sparse_fts by set abstract layer here
        features = features.transpose(1, 2).contiguous()
        xyz = batch_data_label['openscene_sparse_pcd']
        pre_enc_xyz, pre_enc_features, pre_enc_inds = scene_tokenizer(xyz, features)
        pre_enc_features = pre_enc_features.transpose(1, 2).contiguous()
        
        torch.save(pre_enc_inds, f'{ot_dir}/enc_inds.pt')
        torch.save(pre_enc_xyz, f'{ot_dir}/enc_xyz.pt')
        torch.save(pre_enc_features, f'{ot_dir}/enc_features.pt')
        
        
        ## visualization
        import open3d
        import numpy as np
        # sparse_vis_pcd = open3d.geometry.PointCloud()
        # sparse_vis_pcd.points = open3d.utility.Vector3dVector(xyz[0].cpu().numpy())
        # colors = np.ones_like(xyz[0].cpu().numpy()) * 0.8
        # colors[pre_enc_inds[0].cpu().numpy()] = [1., 0., 0.]
        # sparse_vis_pcd.colors = open3d.utility.Vector3dVector(colors)
        # open3d.visualization.draw_geometries([sparse_vis_pcd])
        
        for ri, r_xyz in tqdm(enumerate(pre_enc_xyz.squeeze(0))):
            idx = ball_query(region_radius, region_nsample, dense_scene_xyz, r_xyz.unsqueeze(0).unsqueeze(0)).long()
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (region_nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind
            idx.squeeze_(1)
            
            interest_dense_scene_xyz = torch.stack([dense_scene_xyz[i, idx[i], :] for i in range(len(idx))], dim=0)
            interest_dense_scene_fts = torch.stack([dense_scene_fts[i, idx[i], :] for i in range(len(idx))], dim=0)
            interest_dense_scene_fts = interest_dense_scene_fts.transpose(1, 2).contiguous()
            pre_reg_xyz, pre_reg_features, pre_reg_inds = region_tokenizer(interest_dense_scene_xyz, interest_dense_scene_fts)
            
            torch.save(pre_reg_inds, f'{ot_dir}/region_inds_{ri}.pt')
            torch.save(pre_reg_xyz, f'{ot_dir}/region_xyz_{ri}.pt')
            torch.save(pre_reg_features.transpose(1,2), f'{ot_dir}/region_features_{ri}.pt')
            
            # dense_vis_pcd = open3d.geometry.PointCloud()
            # dense_vis_pcd.points = open3d.utility.Vector3dVector(dense_scene_xyz[0].cpu().numpy())
            # colors = np.ones_like(dense_scene_xyz[0].cpu().numpy()) * 0.8
            # # dense_vis_pcd.colors = open3d.utility.Vector3dVector(colors)
            # select_vis_pcd = open3d.geometry.PointCloud()
            # select_vis_pcd.points = open3d.utility.Vector3dVector(interest_dense_scene_xyz[0].cpu().numpy())
            # colors = np.ones_like(interest_dense_scene_xyz[0].cpu().numpy()) * 0.3
            # colors[pre_reg_inds[0].cpu().numpy()] = [1., 0., 0.]
            # select_vis_pcd.colors = open3d.utility.Vector3dVector(colors)
            # open3d.visualization.draw_geometries([dense_vis_pcd, select_vis_pcd])
            # open3d.visualization.draw_geometries([select_vis_pcd])
            pass

@torch.no_grad()
def greedy_decode(transformer: Callable, **kwargs) -> Tensor:
    
    ## prepare inputs
    max_length = kwargs['max_length']
    inputs_embeds = kwargs['inputs_embeds']
    batch_data_label = kwargs['batch_data_label']
    # lm_head = kwargs['lm_head']
    
    batch, _, channel = inputs_embeds.shape
    
    ## prepare storage
    output_ids = torch.ones(batch, max_length).long().to(inputs_embeds.device)
    output_ids = output_ids * kwargs['eos_token_id']
    
    ## prepare temporal storage of inputs
    temporal_inputs = inputs_embeds
    finished_batchs = torch.zeros(batch).bool().to(inputs_embeds.device)
    embedding_layer = transformer.get_input_embeddings()
    for word_id in range(max_length):
        
        step_output = transformer(
            inputs_embeds=temporal_inputs,
            batch_data_label=copy.deepcopy(batch_data_label)
        )
        
        # logits = lm_head(step_output[0]).contiguous()
        logits = step_output.logits
        ## greedy decoding, find out whats the most possible word
        next_word_id = logits[:, -1, :].argmax(-1)
        
        # check those finished sentences and overwrite
        finished_batchs |= (next_word_id == kwargs['eos_token_id'])
        next_word_id[finished_batchs] = kwargs['eos_token_id']
        
        output_ids[:, word_id] = next_word_id.long()    # (batch, )
        
        temporal_inputs = torch.cat((inputs_embeds, embedding_layer(output_ids[:, :word_id+1])), dim=1)
        
    return OrderedDict({'output_ids': output_ids.long()})
    
    
@torch.no_grad()
def beam_search_decode(transformer: Callable, **kwargs) -> Tensor:
    ## prepare inputs
    max_length = kwargs['max_length']
    attention_mask = kwargs['attention_mask']
    inputs_embeds = kwargs['inputs_embeds'][attention_mask == 1].unsqueeze(0) # batch x nwords x channel
    dense_pcd_info = kwargs['dense_pcd_info']
    lm_head = kwargs['lm_head']
    # for safety issues
    assert kwargs['num_beams'] is not None, (
        'num_beams should not be provided if calling beam search!'
    )
    nbeams = kwargs['num_beams']
    
    batch, prefix_length, channel = inputs_embeds.shape
    # batch x nbeams x length x channel
    expanded_inputs_embeds = inputs_embeds.unsqueeze(1).repeat(1, nbeams, 1, 1)
    expanded_dense_pcd_info = {k:v.repeat(nbeams, 1, 1, 1) for k,v in dense_pcd_info.items()}
    ## prepare storage
    output_scores = torch.zeros(batch, nbeams).to(inputs_embeds.device)
    output_ids = torch.ones(batch, nbeams, max_length).to(inputs_embeds.device)
    output_ids = output_ids * kwargs['eos_token_id']
    batch_beam_results = OrderedDict({
        batch_id: [
            [float('-inf'), (float('-inf'), float('-inf')), None, None] \
                for b in range(nbeams)] \
                    for batch_id in range(batch)
    })
    embedding_layer = kwargs['embedding_layer']
    
    for word_id in range(max_length):
        
        if word_id == 0:    # cold start for the first generation step
        
            step_output = transformer(
                inputs_embeds=inputs_embeds,
                dense_pcd_info=dense_pcd_info
            )
            logits = lm_head(step_output[0]).contiguous()
            # topk inds
            topk_scores, topk_inds = logits[:, -1, :].topk(
                k=nbeams, largest=True, dim=-1
            )   # batch x nbeams
            
            # store temporal scores for each beam
            output_ids[..., word_id] = topk_inds
            output_scores += torch.log_softmax(topk_scores, dim=-1)
            
        else:   # warm start from the previous step
            
            # batch x nbeams x word_id
            generated_words = output_ids[..., :word_id]
            
            # batch x nbeams x (length + word_id) x channel
            temporal_inputs = torch.cat((expanded_inputs_embeds, embedding_layer(generated_words.long())), dim=2)
            
            step_output = transformer(
                inputs_embeds=temporal_inputs.reshape(
                    batch * nbeams, prefix_length + word_id, channel
                ),
                dense_pcd_info=expanded_dense_pcd_info
            )
            logits = lm_head(step_output[0]).contiguous()
            last_word_logits = logits[:, -1, :].reshape(
                batch, nbeams, -1
            )   # batch x nbeams x nvocabs
            
            # beam_scores: batch x nbeams x nvocabs
            if word_id != max_length - 1:
                beam_scores = output_scores.unsqueeze(-1) + torch.log_softmax(
                    last_word_logits, dim=-1
                )
                
                output_scores, select_inds = beam_scores.reshape(batch, -1).topk(
                    k=nbeams, largest=True, dim=-1
                )
                # batch x k
                select_beam_id = select_inds // last_word_logits.shape[-1]
                select_word_id = select_inds % last_word_logits.shape[-1]
                
            else:
                
                # force ends of certain captions
                last_word_probs = torch.log_softmax(last_word_logits, dim=-1)
                output_scores += last_word_probs[..., kwargs['eos_token_id']]
                select_beam_id = \
                    torch.arange(nbeams).to(output_ids.device).unsqueeze(0).repeat(batch, 1)
                select_word_id = \
                    torch.ones_like(output_ids[..., -1]) * kwargs['eos_token_id']
            
            # gather generated beams
            output_ids = torch.gather(
                output_ids, 1, 
                select_beam_id.unsqueeze(-1).repeat(1, 1, max_length)
            )
            output_ids[..., word_id] = select_word_id
            
            ## ---- process the finished beams: batch x nbeams
            sentence_log_prob = output_scores / (word_id + 1)
            
            finished_batch, finished_beams = torch.where(
                select_word_id == kwargs['eos_token_id']
            )
            for batch_id, beam_id in zip(
                    finished_batch.cpu().tolist(), 
                    finished_beams.cpu().tolist()
                ):
                sentence = [
                    sentence_log_prob[batch_id, beam_id].cpu().tolist(),
                    (word_id, beam_id),
                    output_ids[batch_id, beam_id],          # max_length
                    sentence_log_prob[batch_id, [beam_id]]  # 1
                ]
                heapq.heappushpop(batch_beam_results[batch_id], sentence)
                
            
            # neglect the finished beam
            output_scores[select_word_id == kwargs['eos_token_id']] = -float('inf')
            
    ## final call, gather beam results from heaps
    output_ids = torch.cat([
        torch.cat(
            [
                beam_sentence.unsqueeze(0) \
                    for _, _, beam_sentence, _ in batch_beam_results[batch_id]
            ], dim=0
        ).unsqueeze(0) \
            for batch_id in range(batch)
    ], dim=0)   # batch x beam x max_length
    
    output_scores = torch.cat([
        torch.cat(
            [
                beam_log_prob.unsqueeze(0) \
                    for _, _, _, beam_log_prob in batch_beam_results[batch_id]
            ], dim=0
        ).unsqueeze(0) \
            for batch_id in range(batch)
    ], dim=0).squeeze(-1)   # batch x beam x 1
    
    return OrderedDict({
        'output_ids': torch.gather(
            output_ids.long(), 1, 
            output_scores.argmax(-1, keepdim=True).unsqueeze(1).repeat(1, 1, max_length)
        ).squeeze(1),
        'output_scores': output_scores,
        'beam_output_ids': output_ids.long()
    })
    


class Shell_Model(nn.Module):
    def __init__(self, config) -> None:
        super(Shell_Model, self).__init__()
        self.model = FlexOPTForCausalLM.from_pretrained('ckpts/opt-model', config=config)
        
    def forward(self, batch_data_label, is_eval, task_name=None):
        if is_eval:
            input_ids = batch_data_label['instruction']
            attention_mask = batch_data_label['instruction_mask']
            caption_config = {
                'early_stopping': True,
                'eos_token_id': self.model.tokenizer.eos_token_id,
                'num_beams': 4 
            }
            response_config = {
                'ov-det': 64,
                'vg': 64,
                'dense-cap': 48,
                'object_caption': 48,
                'qa': 64,
                'chat': 512,
            }
            ## TODO:处理 batch
            output_ids_list = []
            for batch_id in range(input_ids.shape[0]):
                data_label = {
                    'dense_region_tokens': batch_data_label['dense_region_tokens'][batch_id].unsqueeze(0),
                    'click_mask': batch_data_label['click_mask'][batch_id].unsqueeze(0),
                    'click_query': batch_data_label['click_query'][batch_id].unsqueeze(0),
                    'scene_tokens': batch_data_label['scene_tokens'][batch_id].unsqueeze(0),
                    'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'][batch_id].unsqueeze(0),
                    'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'][batch_id].unsqueeze(0),
                    }
                self.model.set_batch_data_label_cache(data_label)
                output_ids = self.model.generate(input_ids = (input_ids[batch_id].unsqueeze(0)[attention_mask[batch_id].unsqueeze(0)==1].unsqueeze(0)), max_length=128)    
                output_ids_list.append(output_ids[:, len(input_ids[batch_id].unsqueeze(0)[attention_mask[batch_id].unsqueeze(0)==1]):])
            output_ids_list = torch.cat(output_ids_list, dim=0)
            #     caption_config['max_length'] = response_config[task_name]
            #     caption_config['batch_data_label'] = data_label
            #     caption_config['inputs_embeds'] = self.model.model.decoder.embed_tokens(input_ids[batch_id].unsqueeze(0)[attention_mask[batch_id].unsqueeze(0)==1].unsqueeze(0))
            #     output_ids = greedy_decode(self.model, **caption_config)['output_ids']
            #     output_ids_list.append(output_ids)
            # output_ids_list = torch.cat(output_ids_list, dim=0)
            return {'output_ids':  output_ids_list}
        else:
            input_ids = batch_data_label['input_ids']
            attention_mask = batch_data_label['attention_mask']
            gradient_mask = batch_data_label['gradient_mask']
        
            ## 现在的input_ids pad到了128个token太长了 这里重新pad
            batch_valid_len_w_eos = (attention_mask != 0).sum(dim=1)
            pad_len = batch_valid_len_w_eos.max().item()
            input_ids = input_ids[:, :pad_len]
            attention_mask = attention_mask[:, :pad_len]
            gradient_mask = gradient_mask[:, :pad_len]
            batch_data_label['input_ids'] = input_ids
            return self.model(input_ids=input_ids, attention_mask=attention_mask, gradient_mask=gradient_mask, batch_data_label=batch_data_label, labels = batch_data_label['input_ids'])