from typing import List, Optional, Tuple, Union
import copy
import os
import torch
import math
import sys
from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetSAModuleVotesFP16
from third_party.pointnet2.pointnet2_utils import ball_query, grouping_operation
sys.path.append('/home/admin/Projects/LL3DA/')
import heapq, time
from torch import Tensor
import numpy as np
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
from models.detector_Vote2Cap_DETR.transformer import (
    MaskedTransformerEncoder, TransformerDecoder, MaskedTransformerEncoderFP16, TransformerEncoder, TransformerEncoderLayer
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

        ## FLEX
        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(256, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
        )
        self.k_hr_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        self.v_hr_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=self.enable_bias)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        reg_features = None
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

            ## FLEX
            if reg_features is not None and hasattr(self, 'encoder_to_llm_projection'):
                reg_features = self.encoder_to_llm_projection(reg_features)
                bsz, seq_len, topk, hdim = reg_features.shape
                hr_key_value_states = reg_features
                
                dense_token_num = hr_key_value_states.shape[2]
                scene_token_num = 32
                
                hr_attention_mask = torch.ones((bsz, hr_key_value_states.shape[1], hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device, dtype=hr_key_value_states.dtype)*torch.finfo(hr_key_value_states.dtype).min
                ## 每个TEXT TOKEN只能跟自己选出来的DENSE SCENE TOKEN进行交互
                ## PAD的TEXT TOKEN也需要MASK
                batch_valid_len_w_eos = (attention_mask.squeeze(1)[:, -1, :] == 0).sum(dim=1) - scene_token_num
                for bs in range(bsz):
                    valid_length = batch_valid_len_w_eos[bs].item()
                    for j in range(valid_length):
                        start_index = j * dense_token_num
                        end_index = (j+1) * dense_token_num
                        hr_attention_mask[bs, j, start_index:end_index] = 0.
                        
                ## 最后为SCENE TOKEN加上ATTN MASK
                ## OPTION1: 每个SCENE TOKEN都可以于DENSE TOKEN交互
                scene_token_attention_mask = torch.zeros((bsz, scene_token_num, hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device, dtype=hr_key_value_states.dtype)
                        
                ## OPTION2: 每个SCENE TOKEN只能和自己选出来的DENSE TOKEN交互
                # scene_token_attention_mask = torch.ones((bsz, scene_token_num, hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device, dtype=hr_key_value_states.dtype)*torch.finfo(hr_key_value_states.dtype).min
                # for bs in range(bsz):
                #     top_indices_bs = top_indices[bs].view(-1)
                #     for dj in range(len(top_indices_bs)):
                #         id = top_indices_bs[dj]
                #         scene_token_attention_mask[bs, id, dj*10:(dj+1)*10] = 0.
                
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
            ## DROP HR ATTN HERE
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
            attn_weights_reshaped = attn_weights_reshaped[..., :tgt_len]
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


from models.detector_Vote2Cap_DETR.config import model_config_flex
from scipy.spatial import cKDTree

class DenseTokenSelection(nn.Module):
    def __init__(self):
        super().__init__()
        self._query_topk = 4
        self._scene_token_topk = 2
        
        '''
        self._sample_nearest_pcd_radius = 0.3
        self._sample_nearest_pcd_npoint = 512
        '''
        
        self._dense_pcd_dir = 'data/scannet/scannet_data_dense'
        
        '''
        self._preenc_npoints = 5
        cfg = model_config_flex(self._preenc_npoints)
        self.tokenizer = self._build_preencoder(cfg)
        self.tokenizer = self.tokenizer.float()
        self.encoder = self._build_encoder(cfg)
        self.encoder.interim_downsampling = self.encoder.interim_downsampling.float()
        '''
        
        self.pcd_dict = {}
        self._preload_dense_pcd()
    
    def _build_preencoder(self, cfg):
        mlp_dims = [cfg.in_channel, 64, 128, cfg.enc_dim]
        preencoder = PointnetSAModuleVotesFP16(
            radius=0.2,
            nsample=256,
            npoint=cfg.preenc_npoints,
            mlp=mlp_dims,
            normalize_xyz=True,
        )
        return preencoder
        
    def _build_encoder(self, cfg):
        if cfg.enc_type == "vanilla":
            encoder_layer = TransformerEncoderLayer(
                d_model=cfg.enc_dim,
                nhead=cfg.enc_nhead,
                dim_feedforward=cfg.enc_ffn_dim,
                dropout=cfg.enc_dropout,
                activation=cfg.enc_activation,
            )
            encoder = TransformerEncoder(
                encoder_layer=encoder_layer, num_layers=cfg.enc_nlayers
            )
        elif cfg.enc_type in ["masked"]:
            encoder_layer = TransformerEncoderLayer(
                d_model=cfg.enc_dim,
                nhead=cfg.enc_nhead,
                dim_feedforward=cfg.enc_ffn_dim,
                dropout=cfg.enc_dropout,
                activation=cfg.enc_activation,
            )
            interim_downsampling = PointnetSAModuleVotesFP16(
                radius=0.2,
                nsample=256,
                npoint=cfg.preenc_npoints,
                mlp=[cfg.enc_dim, 256, 256, cfg.enc_dim],
                normalize_xyz=True,
            )
            
            masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
            encoder = MaskedTransformerEncoderFP16(
                encoder_layer=encoder_layer,
                num_layers=3,
                interim_downsampling=interim_downsampling,
                masking_radius=masking_radius,
            )
        else:
            raise ValueError(f"Unknown encoder type {cfg.enc_type}")
        return encoder
    
    def _retrive_topk_query_from_text(self, opt_attn_map):
        '''
        opt_attn_map : [BS, HEAD, SEQ_LEN, SEQ_LEN]
        '''
        ## MEAN IN HEAD
        opt_attn_map = torch.mean(opt_attn_map, dim=1)
        ## FILTER 32 LEARNABLE QUERY IN ROW
        opt_attn_map = opt_attn_map[:, 32:, :]
        ## FILTER TEXT IN COLUMN
        opt_attn_map = opt_attn_map[:, :, :32]
        ## 取TOPK的query
        ## [BSZ, TEXT_SEQ_LEN, TOPK]
        _, argmax_idx = opt_attn_map.float().topk(k=self._query_topk, dim=-1)
        
        return argmax_idx
        
            
    def _retrive_topk_scene_token_from_query(self, qformer_attn_map, qformer_scene_token_xyz, batch_topk_query):
        ## DROP VISUAL PROMPT
        qformer_attn_map = qformer_attn_map[:, :, :, :32,  :]
        ## MEAN IN HEAD
        qformer_attn_map = torch.mean(qformer_attn_map, dim=2)
        ## MEAN IN LAYER
        qformer_attn_map = torch.mean(qformer_attn_map, dim=1)
        ## [BSZ, 32, TOPK]
        _, argmax_idx = qformer_attn_map.float().topk(k=self._scene_token_topk, dim=-1)
        
        batch_select_scene_token_xyz = []
        batch_select_scene_token_ind = []
        for per_k in range(len(argmax_idx[0, 0])):
            idx = argmax_idx[:, :, per_k]
            ## 选出相似度最高的几个SCENE TOKEN
            idx = idx.unsqueeze(1).repeat(1, batch_topk_query.shape[1], 1)
            batch_select_scene_token_ind_perk = torch.gather(idx, -1, batch_topk_query)
            batch_select_scene_token_ind.append(batch_select_scene_token_ind_perk)

            batch_select_scene_token_ind_perk = batch_select_scene_token_ind_perk.unsqueeze(-1).repeat(1, 1, 1, qformer_scene_token_xyz.shape[-1])
            batch_select_scene_token_xyz_perk = torch.gather(qformer_scene_token_xyz.unsqueeze(1).repeat(1, batch_topk_query.shape[1], 1, 3), -2, batch_select_scene_token_ind_perk)
            batch_select_scene_token_xyz.append(batch_select_scene_token_xyz_perk)
        
        batch_select_scene_token_xyz = torch.cat(batch_select_scene_token_xyz, dim=-2)
        batch_select_scene_token_ind = torch.cat(batch_select_scene_token_ind, dim=-1)

        return batch_select_scene_token_xyz, batch_select_scene_token_ind

    
    def _get_batch_scan_dense_pcd(self, batch_scan_name):
        batch_pcd = []
        batch_kd_tree = []
        for scan_name in batch_scan_name:
            scan_name = scan_name.split('_')[0]
            # if scan_name in self.pcd_dict.keys():
            batch_pcd.append(self.pcd_dict[scan_name][0])
            batch_kd_tree.append(self.pcd_dict[scan_name][1])
            # else:
            #     mesh_vertices = np.load(os.path.join(self._dense_pcd_dir, scan_name) + "_aligned_vert.npy")
            #     point_cloud = mesh_vertices[:, 0:6]
                
            #     point_cloud[:, 3:] = (point_cloud[:, 3:] - np.array([109.8, 97.2, 83.8])) / 256.0
                
            #     normals = mesh_vertices[:,6:9]
            #     point_cloud = np.concatenate([point_cloud, normals], 1)
                
            #     floor_height = np.percentile(point_cloud[:, 2], 0.99)
            #     height = point_cloud[:, 2] - floor_height
            #     point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
                
            #     pcd = torch.FloatTensor(point_cloud).half()
            #     self.pcd_dict[scan_name] = pcd
            #     batch_pcd.append(pcd)
            
        return batch_pcd, batch_kd_tree
    
    def _preload_dense_pcd(self):
        import glob
        from tqdm import tqdm
        from scipy.spatial import cKDTree
        
        all_scan_files = glob.glob(f'{self._dense_pcd_dir}/*_aligned_vert.npy')
        for fn in tqdm(all_scan_files, desc='PRELOAD DENSE PCDS'):
            mesh_vertices = np.load(fn)

            instance_labels = np.load(fn.replace('_aligned_vert.npy', '_ins_label.npy'))

            scan_name = fn.split('/')[-1].split('_')[0]
            point_cloud = mesh_vertices[:, 0:6]

            point_cloud[:, 3:] = (point_cloud[:, 3:] - np.array([109.8, 97.2, 83.8])) / 256.0

            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)

            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)
            
            kd_tree = cKDTree(point_cloud[:, :3])
            pcd = torch.FloatTensor(point_cloud)

            self.pcd_dict[scan_name] = (pcd, kd_tree, instance_labels)

    def _break_up_pc(self, pc):
        # pc may contain color/normals.
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features
    
    def _run_encoder(self, point_clouds, inds=None):
        xyz, features = self._break_up_pc(point_clouds)
        
        ## pointcloud tokenization
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.tokenizer(xyz, features, inds)

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        pre_enc_features = pre_enc_features.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        if enc_inds is None:
            # encoder does not perform any downsampling
            enc_inds = pre_enc_inds
        else:
            # use gather here to ensure that it works for both FPS and random sampling
            enc_inds = torch.gather(pre_enc_inds, 1, enc_inds.long())
        return enc_features
    
    def _sample_nearest_point_from_xyz(self, batch_scene_xyz, batch_kd_tree, batch_sample_xyz):
        
        batch_sample_pcd = []
        for scene_xyz, sample_xyz in zip(batch_scene_xyz, batch_sample_xyz):
            # scene_xyz [N, 3] sample_xyz [SEQ_LEN, TOPK, 3]
            scene_xyz = scene_xyz.cuda()
            scene_fts = scene_xyz[..., 3:].unsqueeze(0)
            scene_xyz = scene_xyz[..., :3].unsqueeze(0)
            seq_len, topk, _ = sample_xyz.shape
            sample_xyz = sample_xyz.view(-1, 3).unsqueeze(0)
            idx = ball_query(
                    self._sample_nearest_pcd_radius, 
                    self._sample_nearest_pcd_npoint, 
                    scene_xyz.float().contiguous(), 
                    sample_xyz.float().contiguous()
                    ).long().squeeze(0)
            
        ## USE KDTREE
        # for scene_xyz, scene_kdt, sample_xyz in zip(batch_scene_xyz, batch_kd_tree, batch_sample_xyz):
        #     seq_len, topk, _ = sample_xyz.shape
        #     scene_fts = scene_xyz[..., 3:].unsqueeze(0).cuda()
        #     scene_xyz = scene_xyz[..., :3].unsqueeze(0).cuda()
        #     query_point = sample_xyz.view(-1, 3).cpu().numpy()
        #     indexes_list = scene_kdt.query_ball_point(query_point, self._sample_nearest_pcd_radius)
            
        #     sampled_indexes_list = []
        #     non_zero_index = [i for i in range(len(indexes_list)) if len(indexes_list[i]) > 0]
        #     for indexes in indexes_list:
        #         num_points_in_neighborhood = len(indexes)
        #         if num_points_in_neighborhood == 0:
        #             import random
        #             indexes = indexes_list[random.choice(non_zero_index)]
        #             num_points_in_neighborhood = len(indexes)
        #         if num_points_in_neighborhood >= self._sample_nearest_pcd_npoint:
        #             sampled_indexes = np.random.choice(indexes, self._sample_nearest_pcd_npoint, replace=False)
        #         else:
        #             sampled_indexes = np.random.choice(indexes, self._sample_nearest_pcd_npoint, replace=True)
                    
                    
        #         sampled_indexes_list.append(sampled_indexes)
        #     idx = torch.LongTensor(np.array(sampled_indexes_list)).cuda()
            # idx = idx.view(1, seq_len, topk, self._sample_nearest_pcd_npoint)
            
            ## 均匀采样
            # unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            # for i_batch in range(idx.shape[0]):
            #     for i_region in range(idx.shape[1]):
            #         unique_ind = torch.unique(idx[i_batch, i_region, :])
            #         num_unique = unique_ind.shape[0]
            #         unique_cnt[i_batch, i_region] = num_unique
            #         sample_ind = torch.randint(0, num_unique, (self._sample_nearest_pcd_npoint - num_unique,), dtype=torch.long)
            #         all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
            #         idx[i_batch, i_region, :] = all_ind
            # idx.squeeze_(1)
            
            sample_xyz = torch.gather(scene_xyz.repeat(idx.shape[0], 1, 1), 1, idx.unsqueeze(-1).repeat(1, 1, scene_xyz.shape[-1]))
            sample_fts = torch.gather(scene_fts.repeat(idx.shape[0], 1, 1), 1, idx.unsqueeze(-1).repeat(1, 1, scene_fts.shape[-1]))

            sample_xyz = sample_xyz.view(seq_len, -1, self._sample_nearest_pcd_npoint, sample_xyz.shape[-1])
            sample_fts = sample_fts.view(seq_len, -1, self._sample_nearest_pcd_npoint, sample_fts.shape[-1])
            sample_pcd = torch.cat((sample_xyz, sample_fts), -1)
            batch_sample_pcd.append(sample_pcd)
        
        ## [BSZ, SEQ_LEN, TOPK, NPOINT, 10]
        batch_sample_pcd = torch.stack(batch_sample_pcd, 0)
        
        return batch_sample_pcd
    
    @torch.no_grad()
    def forward(self, opt_attn_map, qformer_attn_map, qformer_scene_token_xyz, scan_name, flex_gt_dense_token=None):
        if os.getenv('use_gt_dense_token','False') == 'True' and self.training:
            seq_len = opt_attn_map.shape[-2] - 32
            ## flex_gt_dense_token : [BS, self._query_topk * self._scene_token_topk*self._preenc_npoints, 1, 256]
            enc_features = flex_gt_dense_token.half().repeat(1, seq_len, 1, 1)
            enc_features = enc_features.view(opt_attn_map.shape[0], seq_len, self._query_topk*self._scene_token_topk, 256)
        else:
            ## [BSZ, TEXT_SEQ_LEN, TOPK]
            batch_topk_query = self._retrive_topk_query_from_text(opt_attn_map)
            ## batch_select_scene_token_ind : [BSZ, SEQ_LEN, self._scene_token_topk * self._query_topk]
            batch_topk_select_scene_token_xyz, batch_select_scene_token_ind = self._retrive_topk_scene_token_from_query(qformer_attn_map, qformer_scene_token_xyz, batch_topk_query)
            ## LOAD PRECOMPUTE DATA HETE
            cache_dir = '/mnt/nfs/share/Adaptive/LL3DA-FLEX/0501_ALL_LL3DA_TOKEN'
            enc_xyz = []
            enc_features = []
            for bsz in range(len(scan_name)):
                sn = scan_name[bsz].split('_')[0]
                info = torch.load(os.path.join(cache_dir, f'{sn}.pt'), map_location=batch_topk_query.device)
                enc_features.append(info['region_features'])
            enc_features = torch.stack(enc_features, dim=0)

            seq_len = opt_attn_map.shape[-2] - 32
            # [BSZ, SEQ_LEN, 1024, 256]
            repeat_enc_features = enc_features.unsqueeze(1).repeat(1, seq_len, 1, 1)
            batch_select_scene_token_ind = batch_select_scene_token_ind.unsqueeze(-1).repeat(1, 1, 1, 256)
            enc_features = torch.gather(repeat_enc_features, 2, batch_select_scene_token_ind).half().contiguous()

        ''' 旧实现 实时采样编码特征
        else:
            batch_pcd, batch_kd_tree = self._get_batch_scan_dense_pcd(scan_name)
            ## [BSZ, TEXT_SEQ_LEN, TOPK]
            # time0 = time.time()
            batch_topk_query = self._retrive_topk_query_from_text(opt_attn_map)
            # time1 = time.time()
            batch_topk_select_scene_token_xyz = self._retrive_topk_scene_token_from_query(qformer_attn_map, qformer_scene_token_xyz, batch_topk_query)
            # time2 = time.time()
            batch_sample_pcd = self._sample_nearest_point_from_xyz(batch_pcd, batch_kd_tree, batch_topk_select_scene_token_xyz)
            bsz, seq_len, topk, npoint, ft_dim = batch_sample_pcd.shape
            ## 为了降低显存只能一个个来
            # batch_sample_pcd = batch_sample_pcd.view(bsz*seq_len, topk, npoint, ft_dim).contiguous()
            # enc_features = []
            # for sample_pcd in batch_sample_pcd:
            #     enc_features.append(self._run_encoder(sample_pcd.float()).half().permute(1, 0, 2))
            # enc_features = torch.stack(enc_features, 0)
            
            ## BATCH 实现
            batch_sample_pcd = batch_sample_pcd.view(bsz*seq_len*topk, npoint, ft_dim).contiguous()
            # time3 = time.time()
            enc_features = self._run_encoder(batch_sample_pcd).permute(1, 0, 2)
            # print(time1 - time0, time2 - time1, time3 - time2, time.time() - time3)
            ## [BSZ, 128, self._query_topk * self._scene_token_topk, self._preenc_npoints, 256]
            enc_features = enc_features.view(bsz, seq_len, topk, enc_features.shape[-2], enc_features.shape[-1]).contiguous()
            # enc_xyz = enc_xyz.view(bsz, seq_len, topk, enc_xyz.shape[-2], enc_xyz.shape[-1]).contiguous()
            # enc_inds = enc_inds.view(bsz, seq_len, topk, enc_inds.shape[-1]).contiguous()
            torch.cuda.empty_cache()
        '''
        return enc_features

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
        reg_features = None
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
            reg_features=reg_features,
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
        
        if os.getenv('use_flex_attn') == 'True':
            self.dense_token_selection = DenseTokenSelection()

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
        flex_attn_info = None
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

        reg_features = None
        
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
                    reg_features
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    reg_features=reg_features
                )

            hidden_states = layer_outputs[0]

            ## TODO：LAYER IDX判断
            if os.getenv('use_flex_attn') == 'True' and idx >= self.config.num_hidden_layers- int(os.getenv('num_finetune_hidden_layers')) - 1 and os.getenv('finetune_flex_self_attn','False') == 'True':
                reg_features = self.dense_token_selection(
                                        opt_attn_map = layer_outputs[1], 
                                        qformer_attn_map = flex_attn_info['qformer_batch_x_attn'],
                                        qformer_scene_token_xyz = flex_attn_info['qformer_scene_token_xyz'],
                                        scan_name = flex_attn_info['scan_name'],
                                        flex_gt_dense_token = flex_attn_info['flex_gt_dense_token']
                                        )
            
            
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
        flex_attn_info = None
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
            flex_attn_info=flex_attn_info
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
        flex_attn_info = None
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
            flex_attn_info=flex_attn_info
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
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        hr_key_value_states: Optional[torch.Tensor] = None,
        top_indices: Optional[torch.Tensor] = None
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
            if hr_key_value_states is not None and hasattr(self,"k_hr_proj"):
                ## TODO: inference
                
                # hr_key_value_states.requires_grad_(True)
                ## hr_key_value_states [bs, text_seq_len, dense_token_num, dim]
                key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
                value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
                dense_token_num = hr_key_value_states.shape[2]
                
                hr_attention_mask = torch.ones((bsz, hr_key_value_states.shape[1], hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device, dtype=hr_key_value_states.dtype)*torch.finfo(hr_key_value_states.dtype).min
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
                # scene_token_attention_mask = torch.zeros((bsz, self.config.scene_token_num, hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device)
                        
                ## 每个scene token只能和自己选出来的dense token交互
                scene_token_attention_mask = torch.ones((bsz, self.config.scene_token_num, hr_key_value_states.shape[1]*hr_key_value_states.shape[2]), device=hr_key_value_states.device, dtype=hr_key_value_states.dtype)*torch.finfo(hr_key_value_states.dtype).min
                for bs in range(bsz):
                    top_indices_bs = top_indices[bs].view(-1)
                    for dj in range(len(top_indices_bs)):
                        id = top_indices_bs[dj]
                        scene_token_attention_mask[bs, id, dj*10:(dj+1)*10] = 0.
                
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
        attn_weights_bf_sm = attn_weights_bf_sm[:, :, :tgt_len]
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
        
        ## drop hr token here
        ## attn_weights [32, tgt_len, src_len]
        attn_weights = attn_weights[:, :, :tgt_len]
        src_len = tgt_len
        
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
        
        multi_head_attn_weights = attn_weights_bf_sm.view(bsz, self.num_heads, tgt_len, src_len).contiguous()
        global_attention_map = torch.sum(multi_head_attn_weights, dim=1)
        scene_token_hidden_state = global_attention_map[:, self.config.scene_token_num:, :][:, :, :self.config.scene_token_num].contiguous() #* self.config.attn_softmax_scale
        scene_token_hidden_state = nn.functional.softmax(scene_token_hidden_state, dim=-1)
        
        ## text token中会有pad token，根据attn mask处理
        # batch_valid_len_w_eos = (attention_mask.squeeze(1)[:, -1, :] == 0).sum(dim=1) - self.config.scene_token_num
        # scene_token_hidden_state = [hs[:batch_valid_len_w_eos[hi].item()] for hi,hs in enumerate(scene_token_hidden_state)]
        
        return attn_output, attn_weights_reshaped, past_key_value, scene_token_hidden_state


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
        return batch_select_dense_region_tokens, top_indices ## [bs, text_token_num, 4]

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

        # self.dense_region_selector = Dense_Region_Selector(config)
    
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        hr_key_value_states: Optional[torch.Tensor] = None,
        top_indices: Optional[torch.Tensor] = None,
        dense_pcd_info = None,
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
            top_indices=top_indices
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
            
        # hr_key_value_states, top_indices = self.dense_region_selector(scene_token_hidden_state, dense_pcd_info['dense_region_tokens'])
        hr_key_value_states = None
        top_indices = None
        
        outputs += (hr_key_value_states,)
        outputs += (top_indices,)
        
        
        return outputs
    

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

    
        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.flex_layers = nn.ModuleList([FlexOPTDecoderLayer(config) for _ in range(config.num_finetune_hidden_layers)])
        
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
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
        
        
        ## causal_attention_mask : [bs, 1, seq len, seq len]  
        ## mask_for_visual_token : [bs , 512 ,512] True in the mask do not contribute to self-attention
        ## scene token只能跟一点范围内的token交互
        ## mask=1 是mask
        # mask_for_visual_token, dist = self.compute_mask_for_scene_tokens(dense_pcd_info['scene_xyz'])
        # mask_for_visual_token = mask_for_visual_token.int()
        # mask_for_visual_token = mask_for_visual_token  # * torch.finfo(causal_attention_mask.dtype).min
        
        ## 这里计算instance mask有些任务只需要激活特定的token，所以需要mask掉其他token 
        ## 这里相当于ll3da的visual prompt 
        ## 这里的方法是只激活对应的instance的token
        ## 这里的mask是用来控制对应的token是否激活 最后输出为0是激活
        scene_token_num = 256
        if os.getenv('token_instance_mask', 'False')  == 'True':
            non_scene_token_num = causal_attention_mask.shape[-1] - scene_token_num
            ## [bs, scene_token_num]
            ## 这里的mask是1是激活 所以需要取反
            token_instance_mask = 1 - dense_pcd_info['token_instance_mask'].int()
            
            ## 1. 计算q的scene token部分与k的scene token部分的attention
            ### 禁止q token跟 k token 交互
            col_token_instance_mask = token_instance_mask.unsqueeze(1)
            col_token_instance_mask = col_token_instance_mask.expand(-1, token_instance_mask.shape[-1], -1)
            ### 禁止k token跟 q token 交互
            row_token_instance_mask = token_instance_mask.unsqueeze(-1)
            row_token_instance_mask = row_token_instance_mask.expand(batch_size, -1, token_instance_mask.shape[-1])
            scene_scene_token_instance_attn_mask = col_token_instance_mask | row_token_instance_mask
            
            ## 2.计算 q的text token部分与k的scene token部分的attention
            ### 禁止q token跟 k token 交互
            col_token_instance_mask = token_instance_mask.unsqueeze(1)
            text_scene_token_instance_attn_mask = col_token_instance_mask.expand(-1, non_scene_token_num, -1)
            ## [bs, whole seq len ,scene_token_num]
            token_instance_attn_mask = torch.cat([scene_scene_token_instance_attn_mask, text_scene_token_instance_attn_mask], dim=-2)
            
            token_instance_attn_mask[:, :scene_token_num, : ] = token_instance_attn_mask[:, :scene_token_num, : ] #| mask_for_visual_token
            final_attn_mask = token_instance_attn_mask.half()
            final_attn_mask = final_attn_mask * torch.finfo(causal_attention_mask.dtype).min
            final_attn_mask = final_attn_mask.unsqueeze(1)
        
        ## causal_attention_mask是0为激活 torch.finfo(causal_attention_mask.dtype).min为不激活

        ## scene token 不应该是causal的
        causal_attention_mask[:,:, :scene_token_num, :scene_token_num] = 0.
        if os.getenv('token_instance_mask', 'False')  == 'True':
            causal_attention_mask[..., :, :scene_token_num] = final_attn_mask
        
        if os.getenv('token_instance_mask', 'False')  == 'True':
            instance_attention_mask = torch.clone(attention_mask)
            instance_attention_mask[:, :dense_pcd_info['token_instance_mask'].shape[1]] = dense_pcd_info['token_instance_mask']
            pos_embeds = self.embed_positions(attention_mask, past_key_values_length)
        else:
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
                
                
        # hidden_states.requires_grad_(True)
        # for k ,v in dense_pcd_info.items():
        #     if v.dtype == torch.float32:
        #         v.requires_grad_(True)        
        for idx, decoder_layer in enumerate(self.flex_layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training :
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                    None if idx == 0 else hr_key_value_states,
                    None if idx == 0 else top_indices,
                    dense_pcd_info,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    hr_key_value_states=None if idx == 0 else hr_key_value_states,
                    top_indices=None if idx == 0 else top_indices,
                    dense_pcd_info=dense_pcd_info,
                )

            
            hr_key_value_states = layer_outputs[-2]
            top_indices = layer_outputs[-1]
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
        self.visual_nquery = 8
        self.hidden_size = 2048
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.visual_nquery*self.hidden_size),
        )
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=256, 
            pos_type='fourier', 
            normalize=True
        )
        
    def expand_prompt_representation(self, prompt_feature: Tensor, prompt_mask: Tensor=None):
        # input:
        #   prompt_feature: batch x nprompt x (ntkn x channel)
        #   prompt_mask: batch x nprompt
        # output:
        #   prompt_feature: batch x (nprompt x ntkn) x channel
        #   prompt_mask: batch x (nprompt x ntkn)
        batch_size, nprompt = prompt_feature.shape[:2]
        if prompt_mask is None:
            prompt_mask = torch.ones_like(prompt_feature[..., 0])
        prompt_mask = prompt_mask.unsqueeze(-1).repeat(1, 1, self.visual_nquery)
        prompt_mask = prompt_mask.reshape(batch_size, nprompt * self.visual_nquery)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt, self.visual_nquery, self.hidden_size)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt * self.visual_nquery, self.hidden_size)
        return prompt_feature, prompt_mask
    
    def forward(self, 
        point_cloud_dims,
        click_query=None,
        click_qmask=None
    ):
        
        click_xyz = click_query     # batch x nquery x 3
        click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)
        click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))
        click_prompt, click_qmask = self.expand_prompt_representation(click_prompt, click_qmask)

        return click_prompt, click_qmask
    

class FlexOPTForCausalLM(OPTForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = FlexOPTModel(config)
        
        # self.prompt_encoder = PromptEncoder()
        
        self.tokenizer = AutoTokenizer.from_pretrained('ckpts/opt-model')

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        in_channel = 768 
        hidden_size = config.word_embed_proj_dim
        # self.encoder = self._build_mask_transformer_encoder(config)

        self.encoder_to_llm_projection = nn.Sequential(
            nn.Linear(in_channel, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.post_init()
    
    def _build_mask_transformer_encoder(self, config):
        import math
        encoder_layer = TransformerEncoderLayer(
            d_model=768,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            activation='relu',
        )
        
        masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
        encoder = MaskedTransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=3,
            masking_radius=masking_radius,
            interim_downsampling=None
        )
        
        return encoder
    
    def _run_mask_tranformer_encoder(self, scene_tokens, xyz):
        
        # pre_enc_features = self.scene_token_in_head(scene_tokens)
        ## expects npoints x batch x channel features
        # pre_enc_features = scene_tokens.permute(1, 0, 2)
        # enc_xyz, enc_features, enc_inds = self.encoder(
        #     pre_enc_features, xyz=xyz
        # )
        # enc_features = enc_features.permute(1, 0, 2)
        enc_features = self.encoder_to_llm_projection(scene_tokens)
        return enc_features
    
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
        xyz = batch_data_label['scene_xyz'] if 'scene_xyz' in batch_data_label else torch.zeros(1).to(inputs_embeds.device)
        pre_enc_features = self._run_mask_tranformer_encoder(batch_data_label['scene_tokens'], xyz)
        
        # batch_data_label['dense_region_tokens'] = self.scene_token_in_head(batch_data_label['dense_region_tokens'])
        
        # batch_data_label['dense_region_tokens'] = self._run_mask_tranformer_encoder(batch_data_label['dense_region_tokens'].view(-1, 10, 771), batch_data_label['dense_region_xyz'].view(-1, 10, 3))
        # batch_data_label['dense_region_tokens'] = batch_data_label['dense_region_tokens'].view(-1, 512, 10 ,2048)
        
        # features = batch_data_label['openscene_sparse_fts']
        # features = features.transpose(1, 2).contiguous()
        # xyz = batch_data_label['openscene_sparse_pcd']
        # pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pcd_tokenizer(xyz, features)
        # pre_enc_features = pre_enc_features.transpose(1, 2).contiguous()
        pre_enc_features_mask = torch.ones_like(pre_enc_features[..., 0])
        
        if past_key_values is None :
            # click_query = batch_data_label['click_query']
            # click_mask = batch_data_label['click_mask']
            # point_cloud_dims = [
            #     batch_data_label["point_cloud_dims_min"],
            #     batch_data_label["point_cloud_dims_max"],
            # ]
            # prompt_embeds, prompt_mask = self.prompt_encoder(point_cloud_dims, click_query, click_mask)
            
            # inputs_embeds = torch.cat([pre_enc_features, prompt_embeds, inputs_embeds], dim=1)
            # attention_mask = torch.cat([pre_enc_features_mask, prompt_mask, attention_mask], dim=1)
            # self.prefix_len = pre_enc_features.shape[1] + prompt_embeds.shape[1]
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
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            dense_pcd_info={
                'dense_region_tokens':batch_data_label['dense_region_tokens'] if 'dense_region_tokens' in batch_data_label else torch.zeros(1).to(inputs_embeds.device),
                'token_instance_mask': batch_data_label['token_instance_mask']
            }
        )

        # save attn results
        # [layer_num(24), num_head(32), seq_len(512+n), seq_len(512+n)]
        # if self.new_episode:
        #     self.new_episode = False
        #     self.new_token_idx = 0
        # assert outputs['attentions'][0].shape[0] == 1
        # attn = torch.cat(outputs['attentions'], dim=0)
        # # attn = attn[:, :, :, :]
        # task_name = batch_data_label['task_name']
        # attn_dict = {
        #     'attn_weight' : attn,
        #     'xyz' : batch_data_label['scene_xyz'],
        #     'scan_idx' : batch_data_label['scan_idx'],
        #     'scan_name': batch_data_label['scan_name']
        # }
        # scan_idx = batch_data_label['scan_idx'].item()
        # op_path = f'results/attn_vis_flex/0423-EncoderMLP-WoVisualPrpmpt-ST128WoXYZ-OPT1_3bmWoCausalMask/10k/{task_name}/{scan_idx}'
        # if not os.path.exists(op_path):
        #     os.makedirs(op_path)
        # scan_idx = batch_data_label['scan_idx']
        # torch.save(attn_dict, f'{op_path}/{self.new_token_idx}.pt')
        # self.new_token_idx += 1
        
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
        p = '/mnt/nfs/share/Adaptive/0423_openscene_scene_tokens_axis_align_w_pcd_info_w_token_instance_label_s_128_0.4_512'
        os.makedirs(p, exist_ok=True)
        self.scan_name_list = os.listdir(p)
        scan_name = batch_data_label['scan_name'][0]
        ot_dir = f'{p}/{scan_name}'
        os.makedirs(ot_dir, exist_ok=True)
        from third_party.pointnet2.pointnet2_modules import  PointnetSAModuleVotes_WoMlp
        from third_party.pointnet2.pointnet2_utils import ball_query
        from tqdm import tqdm
        
        
        if scan_name not in self.scan_name_list:
            self.scan_name_list.append(scan_name)
        else:
            return 
        
        scene_tokenizer = PointnetSAModuleVotes_WoMlp(
            radius=0.4,
            nsample=512,
            npoint=128,
            normalize_xyz=False,
            use_xyz=False
        )
        # region_tokenizer = PointnetSAModuleVotes_WoMlp(
        #     radius=self.config.region_token_sample_radius,
        #     nsample=self.config.region_token_sample_num_per_ball,
        #     npoint=self.config.region_token_sample_num,
        #     normalize_xyz=True,
        #     use_xyz=False
        # )
        # region_radius = self.config.region_radius
        # region_nsample = self.config.region_sample_num
        

        ## concat xyz with fts
        features = batch_data_label['openscene_sparse_fts']
        dense_scene_xyz = batch_data_label['openscene_dense_pcd']
        dense_scene_fts = batch_data_label['openscene_dense_fts']
        instance_labels = batch_data_label['instance_labels']
        
        ## get sparse scene object tokens from openscene_sparse_fts by set abstract layer here
        features = features.transpose(1, 2).contiguous()
        xyz = batch_data_label['openscene_sparse_pcd']
        
        
        pre_enc_xyz, pre_enc_features, pre_enc_inds = scene_tokenizer(xyz, features)
        pre_enc_features = pre_enc_features.transpose(1, 2).contiguous()
        
        token_instance_label = instance_labels[:, pre_enc_inds[0].long()]
        other_pcd_info = {
            'instance_labels' : instance_labels ,
            'point_clouds': xyz, 
            'token_instance_label': token_instance_label
        }
        
        torch.save(pre_enc_inds, f'{ot_dir}/enc_inds.pt')
        torch.save(pre_enc_xyz, f'{ot_dir}/enc_xyz.pt')
        torch.save(pre_enc_features, f'{ot_dir}/enc_features.pt')
        torch.save(other_pcd_info, f'{ot_dir}/other_pcd_info.pt')
        
        
        ## visualization
        # import open3d
        # import numpy as np
        # sparse_vis_pcd = open3d.geometry.PointCloud()
        # sparse_vis_pcd.points = open3d.utility.Vector3dVector(xyz[0].cpu().numpy())
        # colors = np.ones_like(xyz[0].cpu().numpy()) * 0.8
        # colors[pre_enc_inds[0].cpu().numpy()] = [1., 0., 0.]
        # sparse_vis_pcd.colors = open3d.utility.Vector3dVector(colors)
        # open3d.visualization.draw_geometries([sparse_vis_pcd])
        
        # for ri, r_xyz in tqdm(enumerate(pre_enc_xyz.squeeze(0))):
        #     idx = ball_query(region_radius, region_nsample, dense_scene_xyz, r_xyz.unsqueeze(0).unsqueeze(0)).long()
        #     unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
        #     for i_batch in range(idx.shape[0]):
        #         for i_region in range(idx.shape[1]):
        #             unique_ind = torch.unique(idx[i_batch, i_region, :])
        #             num_unique = unique_ind.shape[0]
        #             unique_cnt[i_batch, i_region] = num_unique
        #             sample_ind = torch.randint(0, num_unique, (region_nsample - num_unique,), dtype=torch.long)
        #             all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
        #             idx[i_batch, i_region, :] = all_ind
        #     idx.squeeze_(1)
            
        #     interest_dense_scene_xyz = torch.stack([dense_scene_xyz[i, idx[i], :] for i in range(len(idx))], dim=0)
        #     interest_dense_scene_fts = torch.stack([dense_scene_fts[i, idx[i], :] for i in range(len(idx))], dim=0)
        #     interest_dense_scene_fts = interest_dense_scene_fts.transpose(1, 2).contiguous()
        #     pre_reg_xyz, pre_reg_features, pre_reg_inds = region_tokenizer(interest_dense_scene_xyz, interest_dense_scene_fts)
            
        #     torch.save(pre_reg_inds, f'{ot_dir}/region_inds_{ri}.pt')
        #     torch.save(pre_reg_xyz, f'{ot_dir}/region_xyz_{ri}.pt')
        #     torch.save(pre_reg_features.transpose(1,2), f'{ot_dir}/region_features_{ri}.pt')
            
        # assert len(os.listdir(ot_dir)) == 512 * 3 + 3
            
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


class Shell_Model(nn.Module):
    def __init__(self, config) -> None:
        super(Shell_Model, self).__init__()
        self.model = FlexOPTForCausalLM.from_pretrained('ckpts/opt-model', config=config, torch_dtype=torch.float16)
        
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
            max_length = 64
            output_ids_list = torch.ones(input_ids.shape[0], max_length).long().to(input_ids.device)
            output_ids_list = output_ids_list * caption_config['eos_token_id']
            for batch_id in range(input_ids.shape[0]):
                data_label = {
                    # 'dense_region_tokens': batch_data_label['dense_region_tokens'][batch_id].unsqueeze(0),
                    'click_mask': batch_data_label['click_mask'][batch_id].unsqueeze(0),
                    'click_query': batch_data_label['click_query'][batch_id].unsqueeze(0),
                    'scene_tokens': batch_data_label['scene_tokens'][batch_id].unsqueeze(0),
                    'scene_xyz': batch_data_label['scene_xyz'][batch_id].unsqueeze(0),
                    # 'dense_region_tokens': batch_data_label['dense_region_tokens'][batch_id].unsqueeze(0),
                    # 'dense_region_xyz': batch_data_label['dense_region_xyz'][batch_id].unsqueeze(0),
                    'point_cloud_dims_min': batch_data_label['point_cloud_dims_min'][batch_id].unsqueeze(0),
                    'point_cloud_dims_max': batch_data_label['point_cloud_dims_max'][batch_id].unsqueeze(0),
                    'task_name': 'qa',
                    'scan_idx': batch_data_label['scan_idx'],
                    'scan_name': batch_data_label['scan_name'],
                    'token_instance_mask': batch_data_label['token_instance_mask'][batch_id].unsqueeze(0),
                    }
                self.model.set_batch_data_label_cache(data_label)
                self.model.new_episode = True
                output_ids = self.model.generate(input_ids = (input_ids[batch_id].unsqueeze(0)[attention_mask[batch_id].unsqueeze(0)==1].unsqueeze(0)), max_length=max_length)    
                output_ids = output_ids[:, len(input_ids[batch_id].unsqueeze(0)[attention_mask[batch_id].unsqueeze(0)==1]):][0] 
                output_ids_list[batch_id][:len(output_ids)] = output_ids   
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