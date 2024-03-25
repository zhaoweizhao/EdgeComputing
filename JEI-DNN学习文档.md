## Netron
利用[netron](https://netron.app/)查看模型71.7_T2T_ViT_7.pth架构
## Running the code(JEI-DNN)
1.Install all requirements `pip install -r requirements.txt`  
2.To transfer learn T2T-ViT-7 to CIFAR-10:  
` python transfer_learning.py --lr 0.05 --dataset cifar10 --weights-path model_weights/71.7_T2T_ViT_7.pth.tar`
## The training scheme when training gates.
1.默认方案（Default:1）：训练最优门以退出，同时强制其他门不退出。  
2.忽略后续方案（Ignore subsequent:2）：训练最优门以退出，先前的门不退出，并忽略后面（更深层）的门。  
3.退出后续方案（Exit subsequent:3）：训练最优门以退出，所有后续的门也退出，而先前的门被训练为不退出。  
