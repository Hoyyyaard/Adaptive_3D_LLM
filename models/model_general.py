import torch
import importlib
from torch import nn
import os

class CaptionNet(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_detector is True:
            self.detector.eval()
            for param in self.detector.parameters():
                param.requires_grad = False
        return self
    
    def pretrained_parameters(self):
        if hasattr(self.captioner, 'pretrained_parameters'):
            return self.captioner.pretrained_parameters()
        else:
            return []
    
    def __init__(self, args, dataset_config, train_dataset):
        super(CaptionNet, self).__init__()
        
        self.freeze_detector = args.freeze_detector
        self.detector = None
        self.captioner = None
        
        if args.abl_ll3da_w_openscene_token:
            self.openscene2ll3da_head = nn.Linear(768, 256)
            self.xyz_head = nn.Linear(3, 128)
        
        if args.detector is not None:
            detector_module = importlib.import_module(
                f'models.{args.detector}.detector'
            )
            self.detector = detector_module.detector(args, dataset_config)
        
        if args.captioner is not None:
            captioner_module = importlib.import_module(
                f'models.{args.captioner}.captioner'
            )
            self.captioner = captioner_module.captioner(args, train_dataset)
        
        self.train()
        
    def forward(self, batch_data_label: dict, is_eval: bool=False, task_name: str=None, train_encoder: bool=False) -> dict:
        
        outputs = {'loss': torch.zeros(1)[0].cuda()}
        
        abl_ll3da_w_openscene_token = os.getenv('abl_ll3da_w_openscene_token', 'False')
        if self.detector is not None and not abl_ll3da_w_openscene_token == 'True':
            if self.freeze_detector is True:
                outputs = self.detector(batch_data_label, is_eval=True)
            else:
                outputs = self.detector(batch_data_label, is_eval=is_eval)
        else:
            outputs['enc_features'] = self.openscene2ll3da_head(batch_data_label['scene_tokens'])
            xyz = self.xyz_head(batch_data_label['scene_xyz'])
            outputs['enc_features'] = torch.cat(outputs['enc_features'], xyz, dim=-1)
            outputs['enc_xyz'] = batch_data_label['scene_xyz']
            outputs['sem_cls_logits'] = torch.zeros((batch_data_label['enc_xyz'].shape[0], 256, 128)).to(batch_data_label['enc_xyz'].device)
            batch_data_label['box_query'] = None
            
        if task_name == 'preprocess':
            return outputs
        
        if train_encoder is True:
            return outputs
        
        if self.freeze_detector is True:
            outputs['loss'] = torch.zeros(1)[0].cuda()
        # elif self.freeze_detector is False and os.getenv('adaptive_pcd_input') == 'True':
        #     outputs['loss'] = torch.zeros(1)[0].cuda()
        
        
        if self.captioner is not None:
            outputs = self.captioner(
                outputs, 
                batch_data_label, 
                is_eval=is_eval, 
                task_name=task_name
            )
        else:
            batch, nproposals, _, _ = outputs['box_corners'].shape
            outputs['lang_cap'] = [
                ["this is a valid match!"] * nproposals
            ] * batch
        return outputs
