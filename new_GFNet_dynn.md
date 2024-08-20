## NWPU
```javascript
****************0.5****************
Exiting Layer0:[0.0/0]
Exiting Layer1:[0.0/0]
Exiting Layer2:[8.571428298950195/99.44444274902344]
Exiting Layer3:[1.6031745672225952/99.00990295410156]
Exiting Layer4:[12.365078926086426/99.6148910522461]
Exiting Layer5:[31.238094329833984/98.78048706054688]
Exiting Layer6:[13.460317611694336/96.9339599609375]
Exiting Layer7:[22.650793075561523/96.07568359375]
Exiting Layer8:[10.11111068725586/74.09732818603516]
Exiting Layer9:[0.0/0]
Exiting Layer10:[0.0/0]
Exiting Layer11:[0.0/0]
acc_val=95.58730158730158,total_GFLOPs=217.8188934326172G
****************0.5****************
threshold:0.5-->expected_acc:95.58730158730158<=>expected_GFLOPs217.8188934326172G
Val:  [ 0/66]  eta: 0:01:09    time: 1.0599  data: 0.9193  max mem: 512
Val:  [10/66]  eta: 0:00:15    time: 0.2759  data: 0.0961  max mem: 512
Val:  [20/66]  eta: 0:00:10    time: 0.1901  data: 0.0128  max mem: 512
Val:  [30/66]  eta: 0:00:07    time: 0.1884  data: 0.0116  max mem: 512
Val:  [40/66]  eta: 0:00:05    time: 0.1849  data: 0.0111  max mem: 512
Val:  [50/66]  eta: 0:00:03    time: 0.1747  data: 0.0103  max mem: 512
Val:  [60/66]  eta: 0:00:01    time: 0.1756  data: 0.0101  max mem: 512
Val:  [65/66]  eta: 0:00:00    time: 0.1739  data: 0.0098  max mem: 512
Val: Total time: 0:00:13 (0.1975 s / it)
6300
6034
****************0.55****************
Exiting Layer0:[0.0/0]
Exiting Layer1:[0.0/0]
Exiting Layer2:[6.8412699699401855/99.76798248291016]
Exiting Layer3:[0.8571428656578064/100.0]
Exiting Layer4:[10.44444465637207/99.8480224609375]
Exiting Layer5:[30.73015785217285/99.01859283447266]
Exiting Layer6:[12.095237731933594/98.4251937866211]
Exiting Layer7:[27.238094329833984/96.96969604492188]
Exiting Layer8:[11.746031761169434/75.8108139038086]
Exiting Layer9:[0.0476190485060215/33.33333206176758]
Exiting Layer10:[0.0/0]
Exiting Layer11:[0.0/0]
acc_val=95.77777777777777,total_GFLOPs=220.03750610351562G
****************0.55****************
```
### UCM
```javascript
****************0.75****************
Exiting Layer0:[0.0/0]
Exiting Layer1:[0.0/0]
Exiting Layer2:[0.0/0]
Exiting Layer3:[0.0/0]
Exiting Layer4:[0.0/0]
Exiting Layer5:[0.0/0]
Exiting Layer6:[26.66666603088379/100.0]
Exiting Layer7:[47.619049072265625/99.5]
Exiting Layer8:[16.904762268066406/85.91548919677734]
Exiting Layer9:[8.809523582458496/81.0810775756836]
Exiting Layer10:[0.0/0]
Exiting Layer11:[0.0/0]
acc_val=95.71428571428572,total_GFLOPs=234.17755126953125G
****************0.75****************
threshold:0.75-->expected_acc:95.71428571428572<=>expected_GFLOPs234.17755126953125G
Val:  [0/5]  eta: 0:00:05    time: 1.1084  data: 0.8492  max mem: 510
Val:  [4/5]  eta: 0:00:00    time: 0.3761  data: 0.1795  max mem: 510
Val: Total time: 0:00:01 (0.3914 s / it)
420
408
****************0.8****************
Exiting Layer0:[0.0/0]
Exiting Layer1:[0.0/0]
Exiting Layer2:[0.0/0]
Exiting Layer3:[0.0/0]
Exiting Layer4:[0.0/0]
Exiting Layer5:[0.0/0]
Exiting Layer6:[0.0/0]
Exiting Layer7:[33.33333206176758/100.0]
Exiting Layer8:[15.952381134033203/100.0]
Exiting Layer9:[36.66666793823242/98.05194854736328]
Exiting Layer10:[13.095237731933594/83.63636016845703]
Exiting Layer11:[0.9523809552192688/100.0]
acc_val=97.14285714285714,total_GFLOPs=247.00169372558594G
****************0.8****************
threshold=0.8
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 100.0, 98.0392156862745, 0.0, 95.08196721311475]
[0, 0, 0, 0, 0, 0, 0, 139, 67, 153, 0, 61]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 33.095238095238095, 15.95238095238095, 36.42857142857142, 0.0, 14.523809523809526]
* Acc@1 98.571 Acc@5 100.000 loss 0.365
Unfreezing classifiers after warmup
warm up ...

testing ...

100%|█████████████████████████████████████████████████████████████████████████████████| 420/420 [00:12<00:00, 34.25it/s]
FPS: 34.248681477199696
elapsed_time_ms: 29.19820433571225
Avg Forward Time per Image: 23.735474972497848 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

Total time: 18.7003 s
File: /home/nvidia/heShaoWei/GFNet/dynn/gfnet_dynn.py
Function: forward at line 529

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   529                                               @profile
   530                                               def forward(self, x):
   531       840       4650.5      5.5      0.0          B = x.shape[0]
   532       840     608144.3    724.0      3.3          x = self.patch_embed(x)
   533       840      62762.9     74.7      0.3          x = x + self.pos_embed
   534       840      49692.7     59.2      0.3          x = self.pos_drop(x)
   535      8076      29747.1      3.7      0.2          for blk_idx, blk in enumerate(self.blocks):
   536      7954   15381674.7   1933.8     82.3              x = blk.forward(x)
   537                                                       # if blk_idx in [8, 9, 10]:
   538      7954      16216.0      2.0      0.1              if blk_idx in [7, 8, 9]: ##UCM
   539                                                       # if blk_idx in [ 9, 10]:##NWPU
   540
   541      1830     384021.1    209.8      2.1                  inter_z = self.norm(x).mean(1)
   542      1830     599577.2    327.6      3.2                  inter_z = inter_z.to('cpu')
   543      1830     205369.6    112.2      1.1                  inter_logit = self.intermediate_heads[blk_idx](inter_z)
   544                                                           # inter_logit = inter_logit.to('cpu')
   545      1830    1165160.4    636.7      6.2                  g = self.gates[blk_idx](inter_logit)
   546      1830      33352.0     18.2      0.2                  g = torch.sigmoid(g)
   547      1830      83713.7     45.7      0.4                  if g >= self.threshold:  # 用torch.jit.script时可以使用.item()
   548       718       1675.2      2.3      0.0                      return inter_logit, blk_idx
   549
   550       122      24217.5    198.5      0.1          x = self.norm(x).mean(1)
   551                                                   # x = x.to('cpu')
   552       122      47990.9    393.4      0.3          x = self.head(x)
   553       122       2366.4     19.4      0.0          return x, len(self.blocks) - 1
```
### CIFAR10
```javascript
****************0.75****************
Exiting Layer0:[0.0/0]
Exiting Layer1:[0.0/0]
Exiting Layer2:[0.0/0]
Exiting Layer3:[0.0/0]
Exiting Layer4:[0.0/0]
Exiting Layer5:[18.0/99.55555725097656]
Exiting Layer6:[4.440000057220459/98.87387084960938]
Exiting Layer7:[15.979999542236328/99.74968719482422]
Exiting Layer8:[31.200000762939453/98.94230651855469]
Exiting Layer9:[23.469999313354492/98.08265686035156]
Exiting Layer10:[6.909999847412109/84.37047576904297]
Exiting Layer11:[0.0/0]
acc_val=97.97,total_GFLOPs=187.98910522460938G
****************0.75****************
threshold:0.75-->expected_acc:97.97<=>expected_GFLOPs187.98910522460938G
Val:  [  0/105]  eta: 0:01:30    time: 0.8634  data: 0.6582  max mem: 425
Val:  [ 10/105]  eta: 0:00:27    time: 0.2850  data: 0.0692  max mem: 425
Val:  [ 20/105]  eta: 0:00:21    time: 0.2196  data: 0.0094  max mem: 425
Val:  [ 30/105]  eta: 0:00:17    time: 0.2123  data: 0.0084  max mem: 425
Val:  [ 40/105]  eta: 0:00:15    time: 0.2169  data: 0.0081  max mem: 425
Val:  [ 50/105]  eta: 0:00:12    time: 0.2088  data: 0.0081  max mem: 425
Val:  [ 60/105]  eta: 0:00:09    time: 0.1964  data: 0.0090  max mem: 425
Val:  [ 70/105]  eta: 0:00:07    time: 0.2120  data: 0.0102  max mem: 425
Val:  [ 80/105]  eta: 0:00:05    time: 0.2162  data: 0.0101  max mem: 425
Val:  [ 90/105]  eta: 0:00:03    time: 0.2151  data: 0.0091  max mem: 425
Val:  [100/105]  eta: 0:00:01    time: 0.2250  data: 0.0086  max mem: 425
Val:  [104/105]  eta: 0:00:00    time: 0.2185  data: 0.0085  max mem: 425
Val: Total time: 0:00:23 (0.2212 s / it)
10000
9831
****************0.8****************
Exiting Layer0:[0.0/0]
Exiting Layer1:[0.0/0]
Exiting Layer2:[0.0/0]
Exiting Layer3:[0.0/0]
Exiting Layer4:[0.0/0]
Exiting Layer5:[5.289999961853027/100.0]
Exiting Layer6:[1.909999966621399/100.0]
Exiting Layer7:[12.920000076293945/100.0]
Exiting Layer8:[35.22999954223633/99.71614837646484]
Exiting Layer9:[31.700000762939453/99.4637222290039]
Exiting Layer10:[12.949999809265137/89.03475189208984]
Exiting Layer11:[0.0/0]
acc_val=98.31,total_GFLOPs=193.60150146484375G
****************0.8****************
```
