# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict
import math
from timm.models.vision_transformer import VisionTransformer, _cfg, Block, Attention
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
# from fasterkan import FasterKAN as KAN
from KAN import KAN

class GlobalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class kanBlock(nn.Module):

    def __init__(self, dim, hdim_kan=192,
                 drop_path=0., norm_layer=nn.LayerNorm, h=14, w=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # mlp_hidden_dim = int(dim * mlp_ratio)
        self.kan = KAN([dim, hdim_kan, dim])

    def forward(self, x):
        x = x + self.drop_path(self.kan(self.norm2(self.filter(self.norm1(x)))))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768): #768 = 3 * 16 * 16
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        #(B, embed_dim, H/patch_size, W/patch_size) => (B, embed_dim, num_patches) => (B, num_patches, embed_dim)
        return x

class GFNet_KAN_Boosted(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 dropcls=0, hdim_kan=192):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.hdim_kan = hdim_kan
        self.num_classes = num_classes
        self.n_blocks = depth
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        h = img_size // patch_size
        w = h // 2 + 1

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        
        # 设置中间头 intermediate_heads
        self.intermediate_heads = nn.ModuleList([
            nn.Linear(self.num_features, num_classes)
        for i in range(depth-1)])

        self.blocks = nn.ModuleList([
            kanBlock(
                dim=embed_dim, hdim_kan=self.hdim_kan,
                drop_path=dpr[i], norm_layer=norm_layer, h=h, w=w)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.ensemble_reweight = [0.5] * depth
        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()
        
        # 使用截断正态分布（truncated normal distribution）来初始化 self.pos_embed 参数。
        trunc_normal_(self.pos_embed, std=.02)
        
        # 将 _init_weights 方法递归地应用到模型的所有子模块上。
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        intermediate_preds = []
        for itrr ,blk in enumerate(self.blocks):
            x = blk(x)
            if itrr < len(self.intermediate_heads):
                normlized = self.norm(x).mean(1)
                intermediate_preds.append(normlized)

        x = self.norm(x).mean(1)
        return x, intermediate_preds

    def forward(self, x):
        x, inter_preds = self.forward_features(x)
        
        for stage in range(len(inter_preds)):
            inter_preds[stage] = self.final_dropout(inter_preds[stage])
            inter_preds[stage] = self.intermediate_heads[stage](inter_preds[stage])
        x = self.final_dropout(x)
        x = self.head(x)
        return x, inter_preds
    def forward_all(self, x, stage=None):
        final_pred, inter_preds = self.forward(x)
        inter_preds.append(final_pred)
        preds = [0]
        for iii in range(len(inter_preds) + 1):
            pred = (inter_preds[iii] + preds[-1]) * self.ensemble_reweight[iii]
            preds.append(pred)
            if iii == stage:
                break
        return inter_preds, preds
    def forward_inference(self, x):
        final_pred, inter_preds = self.forward(x)
        inter_preds.append(final_pred)
        preds = [0]
        for iii in range(len(inter_preds) + 1):
            pred = (inter_preds[iii] + preds[-1]) * self.ensemble_reweight[iii]
            preds.append(pred)
        preds = preds[1:]
        return preds




# class VisionKAN(VisionTransformer):
#     def __init__(self, *args, num_heads=8, batch_size=16, **kwargs):
#         if 'hdim_kan' in kwargs:
#             self.hdim_kan = kwargs['hdim_kan']
#             del kwargs['hdim_kan']
#         else:
#             self.hdim_kan = 192
        
#         super().__init__(*args, **kwargs)
#         self.num_heads = num_heads
#         # For newer version timm they don't save the depth to self.depth, so we need to check it
#         try:
#             self.depth
#         except AttributeError:
#             if 'depth' in kwargs:
#                 self.depth = kwargs['depth']
#             else:
#                 self.depth = 12

#         block_list = [
#             kanBlock(dim=self.embed_dim, num_heads=self.num_heads, hdim_kan=self.hdim_kan)
#             for i in range(self.depth)
#         ]
#         # check the origin type of the block is torch.nn.modules.container.Sequential
#         # if the origin type is torch.nn.modules.container.Sequential, then we need to convert it to a list
#         if type(self.blocks) == nn.Sequential:
#             self.blocks = nn.Sequential(*block_list)
#         elif type(self.blocks) == nn.ModuleList:
#             self.blocks = nn.ModuleList(block_list)



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


'''
def create_kan(model_name, pretrained, **kwargs):

    if model_name == 'deit_tiny_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_name == 'deit_small_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_224_KAN':
        model = VisionKAN(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_384_KAN':
        model = VisionKAN(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_tiny_distilled_patch16_224_KAN':
        raise RuntimeError('Distilled models are not yet implmented in KAN')

    elif model_name == 'deit_small_distilled_patch16_224_KAN':
        raise RuntimeError('Distilled models are not yet implmented in KAN')

    elif model_name == 'deit_base_distilled_patch16_224_KAN':
        raise RuntimeError('Distilled models are not yet implmented in KAN')

def create_ViT(model_name, pretrained, **kwargs):
    if 'batch_size' in kwargs:
        del kwargs['batch_size']
    if model_name == 'deit_base_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_name == 'deit_small_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_224_ViT':
        model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_patch16_384_ViT':
        model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_tiny_distilled_patch16_224_ViT':
        model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model
    
    elif model_name == 'deit_small_distilled_patch16_224_ViT':
        model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_distilled_patch16_224_ViT':
        model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model

    elif model_name == 'deit_base_distilled_patch16_384_ViT':
        model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
        model.default_cfg = _cfg()
        if pretrained:
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
                map_location="cpu", check_hash=True
            )
            model.load_state_dict(checkpoint["model"])
        return model


def create_model(model_name,**kwargs):
    pretrained = kwargs['pretrained'] if 'pretrained' in kwargs else False
    if 'pretrained' in kwargs:
        del kwargs['pretrained']
    print(kwargs)
    if model_name in __all__KAN:
        model = create_kan(model_name, pretrained, **kwargs)
        model.default_cfg = _cfg()
        return model
    elif model_name in __all__ViT:
        model = create_ViT(model_name, pretrained, **kwargs)
        model.default_cfg = _cfg()
        return model
    else:
        raise RuntimeError('Unknown model (%s)' % model_name)

# if __name__ == '__main__':
#     model = deit_tiny_patch16_224().cuda()
#     img = torch.randn(5, 3, 224, 224).cuda()
#     out = model(img)
#     print(out.shape)

'''