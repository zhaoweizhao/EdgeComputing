This article introduces two methods for implementing early exit: the early exit method proposed in [JEL-DNN(ICLR)](https://arxiv.org/abs/2310.09163) and the one in [BoostedNet(AAAI)](https://arxiv.org/abs/2211.16726). 
Both methods are integrated with the [STTS](https://arxiv.org/abs/2111.11591) video recognition model.

STTS is a Transformer-based video model, and the version with 16 Transformer blocks is chosen for this study. The dataset used in this experiment is Kinetics-400.

## JEL-DNN
1.JEL-DNN utilizes joint training to train both the the gating mechanism (GM) and the intermediate inference modules (IMs), thereby reducing the training gap between IM and GM and resulting in better performance.

2.The architecture produces reliable uncertainty characterization in the form of conformal intervals and well-calibrated predicted probabilities.

3.Training algorithm(JEL-DNN)：

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/eb427ae1-7b0c-4ec2-b318-d1102b7b5779)

4.Training algorithm(BoostedNet)：

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/fbd18afd-3a10-4ab6-b903-89db03044ce1)

## Experimental result
The first figure shows the average accuracy and reduction in inference cost at different thresholds of GM when using the early exit method from **JEL-DNN** combined with the STTS model.

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/ceb707c3-84ff-4fa6-a60a-f1c8b98aa826)

The second figure shows the average accuracy and reduction in inference cost at different thresholds of IM when using the early exit method from **Boosted** combined with the STTS model.

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/d4ab68c1-8628-463b-bfdc-e6a0791264bf)

The third figure shows the classification accuracy of each IM on the entire dataset.

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/420cebdf-d560-4a6c-9063-875da4bdd392)
