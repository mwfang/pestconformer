# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# ConvMAE: https://github.com/Alpha-VL/ConvMAE
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
import pdb
import vision_transformer


class ConvViT(vision_transformer.ConvViT):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(ConvViT, self).__init__(**kwargs)
        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim[-1])

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
               
        stage1_embed = x      
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)           
        
        stage2_embed = x
               
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = self.patch_embed4(x)
        x = x + self.pos_embed
     
        for blk in self.blocks3:
            x = blk(x)
        
        stage3_embed = x.permute(0, 2, 1).view(-1, 768, 14, 14)
       
        features = (stage1_embed, stage2_embed, stage3_embed)
      
        return features
            
    def forward(self, x):
        features = self.forward_features(x)
        xf = {}
        logits = {}
        feature1, feature2, feature3 = features
        #print("feature1.shape",feature1.shape)
        
        f3, f4, f5 = self.fpn([feature1, feature2, feature3])
        a3, a4, a5 = self.apn([f3, f4, f5])
        xf["layer1"] = a3
        xf['layer2'] = a4
        xf['layer3'] = a5
     
        self.fpn_predict(xf, logits)
        selects = self.selector(xf, logits)
        comb_outs = self.combiner(selects)
        logits['comb_outs'] = comb_outs
        
        return logits


def convvit_base_patch16(**kwargs):
    model = ConvViT(
        img_size=[224, 56, 28], patch_size=[4, 2, 2], embed_dim=[256, 384, 768], depth=[2, 2, 11], num_heads=12, mlp_ratio=[4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

