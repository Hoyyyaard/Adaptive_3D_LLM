import copy, math, importlib
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict

from collections import OrderedDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)
from models.ll3da.generation_utils import generation
from models.ll3da.position_embedding import PositionEmbeddingCoordsSine

from utils.box_util import box3d_iou_batch_tensor



def proposal_dimension_select(features: Tensor, indices: Tensor) -> Tensor:
    '''
    
    Parameters
    ----------
    features : Tensor, with size [batch x nsrc x ...]
        Data bank, from which to gather information.
    indices : Tensor, with size [batch x ntgt]
        Indices for gathering information from data bank.

    Returns
    -------
    Tensor, with size [batch x ntgt x ...]
        Gathers features in proposal dimension.
    
    '''
    return torch.gather(
        features, 1, 
        indices.reshape(
            *(indices.shape + tuple(1 for _ in features.shape[2:]))
        ).repeat(
            *((1, 1) + features.shape[2:])
        )
    )


def select_proposal_feature(
    prop_features: Tensor, prop_box_corners: Tensor, prop_sem_mask: Tensor, box_query: Tensor
) -> Tensor:
    '''
    
    Parameters
    ----------
    prop_features : Tensor, with size [batch x nproposal x n_embd]
    prop_box_corners : Tensor, with size [batch x nproposal x 8 x 3]
    prop_sem_mask : Tensor, with size [batch x nproposal], 0 for background
    box_query : Tensor, with size [batch x nquery x 8 x 3]

    Returns
    -------
    Tensor, with size [batch x nquery x n_embd]
        Gathers features in proposal dimension.
    
    '''
    # prop_features
    batch_size, nproposal, _, _ = prop_box_corners.shape
    nquery = box_query.shape[1]
    
    matched_box_iou = box3d_iou_batch_tensor(
        prop_box_corners.unsqueeze(1).repeat(1, nquery, 1, 1, 1).reshape(-1, 8, 3),
        box_query.unsqueeze(2).repeat(1, 1, nproposal, 1, 1).reshape(-1, 8, 3)
    )
    matched_box_iou = matched_box_iou.reshape(batch_size, nquery, nproposal)
    matched_box_iou = matched_box_iou * prop_sem_mask.unsqueeze(1)
    
    matched_indices = matched_box_iou.argmax(-1)    # batch x nquery
    return proposal_dimension_select(prop_features, matched_indices)


class PromptEncoder(nn.Module):
    
    def __init__(self, encoder_hidden_size, visual_nquery, qformer_hidden_size, n_embd):
        super(PromptEncoder, self).__init__()
        self.n_embd = n_embd
        self.visual_nquery = visual_nquery
        self.qformer_hidden_size = qformer_hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        
        self.box_prompt_projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, visual_nquery * qformer_hidden_size),
        )
        self.click_prompt_projector = nn.Sequential(
            nn.Linear(encoder_hidden_size, qformer_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_hidden_size, visual_nquery * qformer_hidden_size),
        )
        self.pos_emb3d = PositionEmbeddingCoordsSine(
            d_pos=encoder_hidden_size, 
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
        prompt_feature = prompt_feature.reshape(batch_size, nprompt, self.visual_nquery, self.qformer_hidden_size)
        prompt_feature = prompt_feature.reshape(batch_size, nprompt * self.visual_nquery, self.qformer_hidden_size)
        return prompt_feature, prompt_mask
    
    def forward(self, 
        detector_output, 
        point_cloud_dims,
        box_query=None, 
        box_qmask=None,
        click_query=None,
        click_qmask=None
    ):
        sem_cls_logits = detector_output['sem_cls_logits']  ## [bs, 256, box_num]
        prop_sem_mask = (sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)).float()
        
        net_device = sem_cls_logits.device
        batch_size = sem_cls_logits.shape[0]
        
        ### prompt encoding
        # box prompt encoding
        visual_prompt = [torch.zeros(batch_size, 0, self.qformer_hidden_size).to(net_device)]
        visual_mask = [torch.zeros(batch_size, 0).to(net_device)]
        if box_query is not None:
            box_prompt = select_proposal_feature(               ## [bs,1,256]
                detector_output['prop_features'][-1], 
                detector_output['box_corners'], 
                prop_sem_mask, 
                box_query
            )
            box_prompt = self.box_prompt_projector(box_prompt)
            box_prompt, box_qmask = self.expand_prompt_representation(box_prompt, box_qmask)
            visual_prompt.append(box_prompt)
            visual_mask.append(box_qmask)
            
        # click prompt encoding: batch x nquery x nproposal
        if click_query is not None:
            click_xyz = click_query     # batch x nquery x 3
            click_prompt = self.pos_emb3d(click_xyz, input_range=point_cloud_dims)
            click_prompt = self.click_prompt_projector(click_prompt.permute(0, 2, 1))
            click_prompt, click_qmask = self.expand_prompt_representation(click_prompt, click_qmask)
            visual_prompt.append(click_prompt)
            visual_mask.append(click_qmask)
        
        ## concat box and click prompts as well as prompt masks
        prompt_feature = torch.cat(visual_prompt, dim=1)   # batch x (2 x ntoken) x channel
        prompt_mask = torch.cat(visual_mask, dim=1)        # batch x (2 x ntoken)
        
        return prompt_feature, prompt_mask


class captioner(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_llm is True:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        return self
    
    def __init__(self, args, train_dataset):
        super(captioner, self).__init__()
        
        self.encoder_hidden_size = 256
        self.dtype = torch.float16
        self.visual_nquery = 8
        self.nlatent_query = 32
        self.freeze_llm = args.freeze_llm
        
        ## initialize tokenizer for batch decoding
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        self.nvocabs = len(self.tokenizer)
        
        ## caption generation cores
        # from src.modeling_opt_flex import FlexOPTForCausalLM
        # self.transformer = AutoModelForCausalLM.from_pretrained('ckpts/opt-model')
        self.transformer = AutoModelForCausalLM.from_pretrained(
            args.vocab,
            torch_dtype=self.dtype,
        )
        self.n_embd = self.transformer.config.hidden_size
        
        ## Multi-modality Transformer
        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=6,
            encoder_hidden_size=self.encoder_hidden_size
        )
        self.qformer = InstructBlipQFormerModel.from_pretrained(
            args.qformer_vocab, 
            config=qformer_config
        )
        self.qformer_hidden_size = qformer_config.hidden_size
        
        
        ## for prompt feature projection
        self.encoder_to_qformer_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_config.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
        )
        self.prompt_encoder = PromptEncoder(
            self.encoder_hidden_size, 
            self.visual_nquery, 
            self.qformer_hidden_size, 
            self.n_embd
        )
        self.latent_query = nn.Embedding(self.nlatent_query, self.qformer_hidden_size)
        self.qformer_to_language_projection = nn.Linear(self.qformer_hidden_size, self.n_embd)
        
        
        self.max_gen_per_iter = 8
        ## ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'num_beams': 4 if args.use_beam_search is True else None,
        }
        self.train()
    
    
    def _get_instruction_response(self, 
            detector_output: dict, 
            inputs: dict, 
            box_query: Tensor=None,
            box_qmask: Tensor=None,
            click_query: Tensor=None,
            click_qmask: Tensor=None
        ) -> dict:
        
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        net_device = inputs["point_clouds"].device
        batch_size = inputs["point_clouds"].shape[0]
        encoder_hidden_states = detector_output['enc_features']
        
        ## prompt encoding
        prompt_feature, prompt_mask = self.prompt_encoder(
            detector_output, 
            point_cloud_dims,
            box_query=box_query, 
            box_qmask=box_qmask,
            click_query=click_query, 
            click_qmask=click_qmask
        )
        
        ## gather query feature for qformer: batch x (n_query + n_tokens) x n_embd
        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_tokens = torch.cat((query_tokens, prompt_feature), dim=1)
        query_attention_mask = torch.cat(
            (torch.ones(batch_size, self.nlatent_query).to(net_device), prompt_mask), dim=1)
        
        # prepare qformer inputs: batch x ntoken x n_embd
        query_attention_mask = torch.cat((query_attention_mask, inputs['qformer_attention_mask']), dim=1)
        
        query_outputs = self.qformer(
            input_ids=inputs['qformer_input_ids'],
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=self.encoder_to_qformer_projection(encoder_hidden_states),
            # output_attentions=True
        )
        
        query_outputs_latent = query_outputs[0][:, : self.nlatent_query, :]
        prefix_feature = self.qformer_to_language_projection(query_outputs_latent)
        
        ## save x-attn here
        # import os
        # assert detector_output['enc_xyz'].shape[0] == 1
        # task_name = inputs['task_name']
        # x_attn_weight = torch.stack(query_outputs['cross_attentions'], dim=0)
        # attn_dict = {
        #     'x_attn_weight' : x_attn_weight,
        #     'xyz' : detector_output['enc_xyz'],
        #     'scan_idx' : inputs['scan_idx'],
        #     'scan_name': inputs['scan_name']
        # }
        # op_path = f'results/attn_vis/{task_name}'
        # if not os.path.exists(op_path):
        #     os.makedirs(op_path)
        # scan_idx = inputs['scan_idx']
        # torch.save(attn_dict, f'{op_path}/{scan_idx.item()}.pt')
        
        return prefix_feature
        
    
    def forward(self, detector_output: dict, inputs: dict, is_eval: bool=False, task_name: str='qa') -> dict:
        
        if is_eval is False:
            return self.forward_training(detector_output, inputs)
        
        response_config = {
            'ov-det': 64,
            'vg': 64,
            'dense-cap': 48,
            'object_caption': 48,
            'qa': 16,
            'chat': 512,
        }
        max_gen_length = response_config[task_name]
        
        if task_name in {'ov-det', 'dense-cap'}:
            return self.predict_densecap(detector_output, inputs, task_name, max_gen_length=max_gen_length)
        elif task_name == 'qa':
            return self.predict_answer(detector_output, inputs, max_gen_length=max_gen_length)
        elif task_name == 'object_caption':
            return self.predict_object_caption(detector_output, inputs, max_gen_length=max_gen_length)
        elif task_name == 'vg':
            return self.predict_vg(detector_output, inputs, max_gen_length=max_gen_length)
        else:
            return self.predict_chat(detector_output, inputs, max_gen_length=max_gen_length)
    
    
    def forward_training(self, detector_output: Dict, inputs: Dict) -> Dict:
        # get word embeddings, NOTE: captioner does not predict <bos> token
        input_ids = inputs['input_ids']         # batch x ntokens
        input_mask = inputs['attention_mask']   # batch x ntokens
        gradient_mask = inputs['gradient_mask'] # batch x ntokens
        
        box_query = inputs.get('box_query', None)       # batch x nquery x 8 x 3
        box_qmask = inputs.get('box_mask', None)        # batch x nquery
        click_query = inputs.get('click_query', None)   # batch x nquery x 3
        click_qmask = inputs.get('click_mask', None)    # batch x nquery
        
        embedding_layer = self.transformer.get_input_embeddings()
        
        # ---- batch x ntoken x n_embd
        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output, 
            inputs=inputs, 
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )
        prefix_mask = torch.ones_like(prefix_tokens[..., 0])
        # ---- batch x (ntoken + nword) x n_embd
        inputs_embeds = torch.cat((prefix_tokens, embedding_layer(input_ids)), dim=1)
        attention_mask = torch.cat((prefix_mask, input_mask), dim=1)
        
        # ---- calculate transformer loss
        outputs = self.transformer(
            inputs_embeds=inputs_embeds.to(self.dtype),
            attention_mask=attention_mask.to(self.dtype),
        )
        
        detector_output['loss'] += self.loss_caption(
            logits = outputs.logits[:, prefix_tokens.shape[1] - 1: -1],
            target = input_ids,
            mask = gradient_mask.to(self.dtype),
        )
        return detector_output

    def loss_caption(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss_per_word = nnf.cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            target,
            reduction='none',
        )
        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)
        # parameter activation for multi-gpu training
        for param in self.parameters():
            if param.requires_grad:
                final_loss += 0 * torch.sum(param.to(final_loss.dtype) ** 2)
        return final_loss
    
    def predict_densecap(self, detector_output: Dict, inputs: Dict, task_name: str, max_gen_length: int=64) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        batch_size, nproposals, _, _ = detector_output['box_corners'].shape
        # ---- to store llm outputs
        output_ids = torch.ones(batch_size, nproposals, max_gen_length).long().to(net_device)
        output_ids = output_ids * self.tokenizer.eos_token_id
        
        # ---- llm input preparation
        instruction = inputs['instruction'][0]              # ntoken
        instruction_mask = inputs['instruction_mask'][0]    # ntoken
        instruction_id = instruction[instruction_mask == 1] # ntoken
        instruction_id = instruction_id[None, :].repeat(batch_size, 1)
        instruction_embedding = embedding_layer(instruction_id) # batch x ntoken x n_embd
        
        ## USer
        # viusualization_preict_bbox(inputs['point_clouds'][0], detector_output['box_corners'][0], inputs['gt_box_corners'][0])
        
        prefix_tokens = []
        for proposal_id in range(nproposals):
            box_query=detector_output['box_corners'][:, [proposal_id]]  # batch x 1 x 8 x 3

            click_query=None
            if task_name == 'ov-det':
                click_query=detector_output['query_xyz'][:, [proposal_id]]  # batch x 1 x 3
            
            instruct_prefix_feature=self._get_instruction_response(     # batch x ntoken x n_embd
                detector_output=detector_output,
                inputs=inputs,
                box_query=box_query,        # batch x 1 x 8 x 3
                click_query=click_query,
            )
            instruct_prefix_feature = torch.cat((instruct_prefix_feature, instruction_embedding), dim=1)
            prefix_tokens.append(instruct_prefix_feature.unsqueeze(1))
        # batch x nproposal x 1 x n_embd
        prefix_tokens = torch.cat(prefix_tokens, dim=1).to(self.dtype)
        
        ## filter and rank the queries
        sem_cls_logits = detector_output["sem_cls_logits"]
        objectness_mask = sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)
        
        ## limit the proposals for generating captions
        candidate_prefix = prefix_tokens[objectness_mask].to(self.dtype)

        gather_output_ids = []
        for start_idx in range(0, candidate_prefix.shape[0], self.max_gen_per_iter):
            prefix = candidate_prefix[start_idx: start_idx + self.max_gen_per_iter]
            scene_cap_output = generation(
                self.transformer, 
                inputs_embeds=prefix,
                max_length=max_gen_length,
                **self.caption_config
            )
            gather_output_ids.append(scene_cap_output['output_ids'])
        gather_output_ids = torch.cat(gather_output_ids, dim=0)
        
        output_ids[objectness_mask] = gather_output_ids
        detector_output['output_ids'] = output_ids
        
        return detector_output
    
    
    def predict_answer(self, detector_output: Dict, inputs: Dict, max_gen_length: int=8) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []
        
        # ---- llm input preparation
        instruction = inputs['instruction']             # ntoken
        instruction_mask = inputs['instruction_mask']   # ntoken

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)
        
        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]     
            sample_mask = instruction_mask[batch_id]     # ntoken
            
            output = generation(
                self.transformer, 
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),   # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_length=max_gen_length,
                **self.caption_config
            )
            output_ids.append(output['output_ids'])
        
        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        
        return detector_output
    
    def predict_chat(self, detector_output: Dict, inputs: Dict, max_gen_length: int=512) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []
        
        # ---- llm input preparation
        instruction = inputs['instruction']             # ntoken
        instruction_mask = inputs['instruction_mask']   # ntoken

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)
        
        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]     
            sample_mask = instruction_mask[batch_id]     # ntoken
            
            output = self.transformer.generate(
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),   # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_new_tokens=max_gen_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
            )   # 1 x max_gen_length
            output = output.squeeze(0)
            placeholder = torch.ones(max_gen_length).to(net_device) * self.tokenizer.eos_token_id
            output = output[:min(max_gen_length, output.shape[0])]
            placeholder[:output.shape[0]] = output
            
            output_ids.append(placeholder.unsqueeze(0).long())
        
        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        
        return detector_output
    

    def predict_object_caption(self, detector_output: Dict, inputs: Dict, max_gen_length: int=8) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []
        
        # ---- llm input preparation
        instruction = inputs['instruction']             # ntoken
        instruction_mask = inputs['instruction_mask']   # ntoken


        box_query = inputs.get('box_query', None)       # batch x nquery x 8 x 3
        box_qmask = inputs.get('box_mask', None)        # batch x nquery
        click_query = inputs.get('click_query', None)   # batch x nquery x 3
        click_qmask = inputs.get('click_mask', None)    # batch x nquery
        
        assert (box_query is not None) or (click_query is not None)
        
        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )
        prefix_tokens = prefix_tokens.to(self.dtype)
        
        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]     
            sample_mask = instruction_mask[batch_id]     # ntoken
            
            output = generation(
                self.transformer, 
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),   # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_length=max_gen_length,
                **self.caption_config
            )
            output_ids.append(output['output_ids'])
        
        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        
        return detector_output
    
    def predict_vg(self, detector_output: Dict, inputs: Dict, max_gen_length: int=8) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []
        
        # ---- llm input preparation
        instruction = inputs['instruction']             # ntoken
        instruction_mask = inputs['instruction_mask']   # ntoken
        box_query = inputs['box_query']
        click_query = inputs['click_query']

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
            box_query=box_query,
            click_query=click_query,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)
        
        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]     
            sample_mask = instruction_mask[batch_id]     # ntoken
            
            output = generation(
                self.transformer, 
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),   # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_length=max_gen_length,
                **self.caption_config
            )
            output_ids.append(output['output_ids'])
        
        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        
        return detector_output

        #  # ---- necessary elements
        # embedding_layer = self.transformer.get_input_embeddings()
        # net_device = next(self.parameters()).device
        # batch_size, nproposals, _, _ = detector_output['box_corners'].shape

        # # ---- to store llm outputs
        # output_ids = torch.ones(batch_size, nproposals, max_gen_length).long().to(net_device)
        # output_ids = output_ids * self.tokenizer.eos_token_id
        
        # # ---- llm input preparation
        # instruction = inputs['instruction'][0]              # ntoken
        # instruction_mask = inputs['instruction_mask'][0]    # ntoken
        # instruction_id = instruction[instruction_mask == 1] # ntoken
        # instruction_id = instruction_id[None, :].repeat(batch_size, 1)
        # instruction_embedding = embedding_layer(instruction_id) # batch x ntoken x n_embd
        
        # prefix_tokens = []
        # for proposal_id in range(nproposals):
        #     # prepare the visual prompt
        #     # box_query are poposal features
        #     box_query=detector_output['box_corners'][:, [proposal_id]]  # batch x 1 x 8 x 3
        #     # point query for localization
        #     click_query=None
        #     click_query=detector_output['query_xyz'][:, [proposal_id]]  # batch x 1 x 3
            
        #     instruct_prefix_feature=self._get_instruction_response(     # batch x ntoken x n_embd
        #         detector_output=detector_output,
        #         inputs=inputs,
        #         box_query=box_query,        # batch x 1 x 8 x 3
        #         click_query=click_query,
        #     )
        #     instruct_prefix_feature = torch.cat((instruct_prefix_feature, instruction_embedding), dim=1)
        #     prefix_tokens.append(instruct_prefix_feature.unsqueeze(1))
        # # batch x nproposal x 1 x n_embd
        # prefix_tokens = torch.cat(prefix_tokens, dim=1).to(self.dtype)
        
        # ## filter and rank the queries
        # sem_cls_logits = detector_output["sem_cls_logits"]
        # objectness_mask = sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)

        # ###################match the proposal for the input query box########################
        # # prefix_tokens： (B, num_proposal, L, D)
        # # torch.gather(prefix_tokens, 1, indices.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 50, 1048))
        # B, num_proposal, L, D = prefix_tokens.shape
        # _, num_query, _, _ = inputs['box_query'].shape
        # assert num_query == 1

        # # FIXME: check the select process is correct for multi-box query, the code are for debug only
        # # (B, num_query), (B, num_query, num_proposal, 1)
        # matched_index, matched_iou = select_proposal_index(prop_box_corners=detector_output['box_corners'], prop_sem_mask=objectness_mask, box_query=inputs['box_query'])
        # # (B, num_query, L, D)
        # matched_prefix_tokens = torch.gather(prefix_tokens, 1, matched_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, L, D))
        # matched_prefix_tokens = matched_prefix_tokens.unsqueeze(0)

        # pass
        
        # ## limit the proposals for generating captions
        # candidate_prefix = prefix_tokens[objectness_mask].to(self.dtype)

        # gather_output_ids = []
        # for start_idx in range(0, candidate_prefix.shape[0], self.max_gen_per_iter):
        #     prefix = candidate_prefix[start_idx: start_idx + self.max_gen_per_iter]
        #     scene_cap_output = generation(
        #         self.transformer, 
        #         inputs_embeds=prefix,
        #         max_length=max_gen_length,
        #         **self.caption_config
        #     )
        #     gather_output_ids.append(scene_cap_output['output_ids'])
        # gather_output_ids = torch.cat(gather_output_ids, dim=0)
        
        # output_ids[objectness_mask] = gather_output_ids
        # detector_output['output_ids'] = output_ids
        
        # return detector_output

def viusualization_preict_bbox(pcd, pred_box_corners, gt_box_corners):
    from utils.box_util import flip_axis_to_camera_np
    import open3d
    pcd = pcd.cpu().numpy()
    pred_box_corners = pred_box_corners.cpu().numpy()
    gt_box_corners = gt_box_corners.cpu().numpy()
    pcd[:, 0:3] = flip_axis_to_camera_np(pcd[:, 0:3])
    # 定义边界框的边，即点之间的连接
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    # 根据点和线创建线段集合
    colors = [[1, 0, 0] for i in range(len(lines))]  # 定义所有线条的颜色，这里设置为红色
    
    pred_bbox_line_set_list = []
    for p_bbox_corner in gt_box_corners:
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(p_bbox_corner),
            lines=open3d.utility.Vector2iVector(lines),
        )
        line_set.colors = open3d.utility.Vector3dVector(colors)
        pred_bbox_line_set_list.append(line_set)
    
    objects_pcd = open3d.geometry.PointCloud()
    objects_pcd.points= open3d.utility.Vector3dVector(pcd[:,:3])
    objects_pcd.colors= open3d.utility.Vector3dVector(pcd[:,3:6])
    # open3d.visualization.draw_geometries([objects_pcd, line_set])
    vis_list = [objects_pcd]
    vis_list.extend(pred_bbox_line_set_list)
    open3d.visualization.draw_geometries(vis_list)
    
