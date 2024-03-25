## netron
利用[netron](https://netron.app/)查看模型71.7_T2T_ViT_7.pth架构
## Running the code(JEI-DNN)
1.Install all requirements `pip install -r requirements.txt`  
2.To transfer learn T2T-ViT-7 to CIFAR-10:  
` python transfer_learning.py --lr 0.05 --dataset cifar10 --weights-path model_weights/71.7_T2T_ViT_7.pth.tar`
