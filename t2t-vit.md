# T2T-ViT 模型
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/3912bcb6-cb35-463a-a13a-c3b2b8f6f407)  
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/cddf7a91-7944-4c52-8ed1-9d8f993edd37)  



class T2T_module(nn.Layer):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self,
                 img_size=224,
                 tokens_type='performer',
                 in_chans=3,
                 embed_dim=768,
                 token_dim=64):
        super().__init__()   
    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose([0, 2, 1])

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape

        x = x.transpose([0, 2, 1]).reshape(
            [B, C, int(np.sqrt(new_HW)),
             int(np.sqrt(new_HW))])
        # iteration1: soft split
        x = self.soft_split1(x).transpose([0, 2, 1])

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose([0, 2, 1]).reshape(
            [B, C, int(np.sqrt(new_HW)),
             int(np.sqrt(new_HW))])
        # iteration2: soft split
        x = self.soft_split2(x).transpose([0, 2, 1])

        # final tokens
        x = self.project(x)

        return x


