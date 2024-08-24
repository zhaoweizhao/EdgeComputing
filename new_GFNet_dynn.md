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
[0.0, 0.0, 0.0, 0.0, 99.78448275862068, 98.64661654135338, 0.0, 96.954986760812, 0.0, 0.0, 0.0, 79.44358578052551]
[0, 0, 0, 0, 1392, 1995, 0, 2266, 0, 0, 0, 647]
[0.0, 0.0, 0.0, 0.0, 22.095238095238095, 31.666666666666664, 0.0, 35.96825396825397, 0.0, 0.0, 0.0, 10.269841269841269]
* Acc@1 96.317 Acc@5 99.714 loss 0.236
Unfreezing classifiers after warmup
warm up ...

testing ...

100%|███████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:29<00:00, 42.13it/s]
FPS: 42.13030808257747
elapsed_time_ms: 23.735881495097328
Avg Forward Time per Image: 19.05335369564238 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

Total time: 222.35 s
File: /home/nvidia/heShaoWei/GFNet/dynn/gfnet_dynn.py
Function: forward at line 529

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   529                                               @profile
   530                                               def forward(self, x):
   531     12600      68754.6      5.5      0.0          B = x.shape[0]
   532     12600    7316785.0    580.7      3.3          x = self.patch_embed(x)
   533     12600     912543.7     72.4      0.4          x = x + self.pos_embed
   534     12600     735237.8     58.4      0.3          x = self.pos_drop(x)
   535     90934     379185.3      4.2      0.2          for blk_idx, blk in enumerate(self.blocks):
   536     89640  174349566.6   1945.0     78.4              x = blk.forward(x)
   537                                                       # if blk_idx in [8, 9, 10]:
   538                                                       # if blk_idx in [7, 8, 9]: ##UCM
   539     89640     175766.0      2.0      0.1              if blk_idx in [4, 5, 7]:##NWPU
   540
   541     28240    5903168.9    209.0      2.7                  inter_z = self.norm(x).mean(1)
   542     28240    9163862.3    324.5      4.1                  inter_z = inter_z.to('cpu')
   543     28240    3368147.8    119.3      1.5                  inter_logit = self.intermediate_heads[blk_idx](inter_z)
   544                                                           # inter_logit = inter_logit.to('cpu')
   545     28240   17656670.2    625.2      7.9                  g = self.gates[blk_idx](inter_logit)
   546     28240     495871.0     17.6      0.2                  g = torch.sigmoid(g)
   547     28240    1259301.6     44.6      0.6                  if g >= self.threshold:  # 用torch.jit.script时可以使用.item()
   548     11306      25409.2      2.2      0.0                      return inter_logit, blk_idx
   549
   550      1294     265058.4    204.8      0.1          x = self.norm(x).mean(1)
   551                                                   # x = x.to('cpu')
   552      1294     250146.7    193.3      0.1          x = self.head(x)
   553      1294      24632.6     19.0      0.0          return x, len(self.blocks) - 1
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

XIT CORRECT RATE:
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 99.72067039106145, 99.46507237256135, 0.0, 90.13867488443759]
[0, 0, 0, 0, 0, 0, 0, 1944, 3580, 3178, 0, 1298]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 19.439999999999998, 35.8, 31.78, 0.0, 12.98]
* Acc@1 98.450 Acc@5 99.980 loss 0.107
Unfreezing classifiers after warmup
warm up ...

testing ...

100%|█████████████████████████████████████████████████████████████████████████████| 10000/10000 [04:19<00:00, 38.55it/s]
FPS: 38.54659334855442
elapsed_time_ms: 25.942629766464233
Avg Forward Time per Image: 23.229271149635316 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

Total time: 436.083 s
File: /home/nvidia/heShaoWei/GFNet/dynn/gfnet_dynn.py
Function: forward at line 529

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   529                                               @profile
   530                                               def forward(self, x):
   531     20000      98227.3      4.9      0.0          B = x.shape[0]
   532     20000   10268332.8    513.4      2.4          x = self.patch_embed(x)
   533     20000    1451448.0     72.6      0.3          x = x + self.pos_embed
   534     20000    1057995.0     52.9      0.2          x = self.pos_drop(x)
   535    192851     673629.3      3.5      0.2          for blk_idx, blk in enumerate(self.blocks):
   536    190255  362592488.6   1905.8     83.1              x = blk.forward(x)
   537
   538    190255     375176.0      2.0      0.1              if blk_idx in [7, 8, 9]: ##UCM CIFAR10
   539                                                       # if blk_idx in [4, 5, 7]:##NWPU
   540
   541     45063    9198764.2    204.1      2.1                  inter_z = self.norm(x).mean(1)
   542     45063   14037028.8    311.5      3.2                  inter_z = inter_z.to('cpu')
   543     45063    4628167.0    102.7      1.1                  inter_logit = self.intermediate_heads[blk_idx](inter_z)
   544                                                           # inter_logit = inter_logit.to('cpu')
   545     45063   27834382.0    617.7      6.4                  g = self.gates[blk_idx](inter_logit)
   546     45063     771882.1     17.1      0.2                  g = torch.sigmoid(g)
   547     45063    2022968.3     44.9      0.5                  if g >= self.threshold:  # 用torch.jit.script时可以使用.item()
   548     17404      40895.1      2.3      0.0                      return inter_logit, blk_idx
   549
   550      2596     519373.2    200.1      0.1          x = self.norm(x).mean(1)
   551                                                   # x = x.to('cpu')
   552      2596     463734.2    178.6      0.1          x = self.head(x)
   553      2596      48450.4     18.7      0.0          return x, len(self.blocks) - 1
```

96.31

| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 15.60M | CIFAR10 | 98.47 | 26.31ms | 136.81mJ/0% |
| GFNet-distil-12-192 | 4.25M | CIFAR10 | 97.10 | 16.44ms | 59.18mJ/56.7% |
| GFNet-dynn-12-384 | 15.65M | CIFAR10 | 98.45 | 23.14ms | **108.75mJ/20.5%** |
| GFNet-dynn-12-192 | 4.27M | CIFAR10 | 96.06 | 15.97ms | **47.92mJ/65.0%** |
| GFNet-12-384(Baseline) | 15.93M | RESISC45 | 96.54 | 26.75ms | 139.10mJ/0% |
| GFNet-distil-12-192 | 4.56M | RESISC45 | 94.97 | 16.75ms | 58.62mJ/57.85% |
| GFNet-dynn-12-384 | 16.12M | RESISC45 | 96.31 | 19.15ms | **90.00mJ/35.29%** |
| GFNet-dynn-12-192 | 4.56M | RESISC45 | 93.89 | 14.09ms | **49.31mJ/64.55%** |
| GFNet-12-384(Baseline) | 15.92M | UMC | 99.52 | 25.88ms | 137.17mJ/0% |
| GFNet-distil-12-192 | 4.55M | UMC | 98.57 | 15.08ms | 52.79mJ/61.51% |
| GFNet-dynn-12-192 | 16.01M | UMC | 98.57 | 23.83ms | **112.00mJ/18.35%** |
| GFNet-dynn-12-192 | 4.64M | UMC | 96.66 | 14.12ms | **42.37mJ/69.11%** |
