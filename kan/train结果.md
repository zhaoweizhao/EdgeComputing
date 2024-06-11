# KAN+GFNet+BoostedNet
## 1.Replace the MLP module in [GFNet](https://github.com/raoyongming/GFNet) with a KAN module, and use the Early-Exit method from BoostedNet to add exit heads to each layer.
Node:This article uses the KAN from [Efficient KAN](https://github.com/Blealtan/efficient-kan), which is an Efficient Implementation of Kolmogorov-Arnold Network.

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/902abf2e-e2b6-45c4-bcd5-d087d6b5712e)

## 2.The model has not finished training yet, and the convergence is very slow.
Node:This experiment uses the ImageNet2012 dataset and conducts distributed training on two A100 GPUs.

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/e1d50e6f-f827-4276-b6bd-3390ac61d426)
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/190786a3-09e9-449d-b8ca-551e4d074f89)
