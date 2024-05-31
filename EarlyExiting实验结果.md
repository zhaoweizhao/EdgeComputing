This article introduces two methods for implementing early exit: the early exit method proposed in [JEL-DNN(ICLR)](https://arxiv.org/abs/2310.09163)和[BoostedNet(AAAI)](https://arxiv.org/abs/2211.16726) and the one in [BoostedNet(AAAI)](https://arxiv.org/abs/2211.16726). 
Both methods are integrated with the [STTS](https://arxiv.org/abs/2111.11591) video recognition model.

STTS is a Transformer-based video model, and the version with 16 Transformer blocks is chosen for this study. The dataset used in this experiment is Kinetics-400.

## JEL-DNN
JEL-DNN utilizes joint training to train both the GM and IM, thereby reducing the training gap between IM and GM and resulting in better performance.

The architecture produces reliable uncertainty characterization in the form of conformal intervals and well-calibrated predicted probabilities;

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/eb427ae1-7b0c-4ec2-b318-d1102b7b5779)

### Experimental result
The first figure shows the average accuracy and reduction in inference cost at different thresholds when using the early exit method from JEL-DNN combined with the STTS model.

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/7198eb70-fd2f-4a1b-b6c2-c3f1b5af4e6a)

