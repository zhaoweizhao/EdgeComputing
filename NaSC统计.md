```javascript
NaSC
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.35185185185185, 0.0, 98.80287310454908, 99.48761742100768, 88.50806451612904]
[0, 0, 0, 0, 0, 0, 0, 1080, 0, 1253, 1171, 496]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.0, 0.0, 31.324999999999996, 29.275000000000002, 12.4]
* Acc@1 97.875 Acc@5 100.000 loss 0.115
```
```javascript
PatternNet
[0.0, 0.0, 0.0, 0.0, 0.0, 99.7789240972734, 0.0, 99.34667211106574, 0.0, 97.50120134550697, 0.0, 83.93782383419689]
[0, 0, 0, 0, 0, 1357, 0, 2449, 0, 2081, 0, 193]
[0.0, 0.0, 0.0, 0.0, 0.0, 22.31907894736842, 0.0, 40.2796052631579, 0.0, 34.22697368421053, 0.0, 3.174342105263158]
* Acc@1 98.322 Acc@5 99.885 loss 0.112
```
| **NaSC-dynn-distillation** | Layer7 | Layer9 | Layer10  | Last Layer  |
| --- | --- | --- | --- | --- |
| Exit Rate | 27.0% | 31.32% | 29.28% | 12.4% |
| Exit Accuracy | 99.3 | 98.8 | 99.4 | 88.5 |

| **PatternNet-dynn-distillation** | Layer5 | Layer7 | Layer9  | Last Layer  |
| --- | --- | --- | --- | --- |
| Exit Rate | 22.3% | 40.3% | 34.2% | 3.1% |
| Exit Accuracy | 99.7 | 99.3 | 97.5 | 83.9 |

### NVIDIA Jetson AGX Orin
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 19.47ms | 61.34mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 16.44ms | 37.28mJ | 39.22% |
| Dynn | 14.93M | NaSC | 97.37 | 9.57ms | 29.67mJ | **51.63%** |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 14.02ms | **35.06mJ** | **42.84%** |
| GFNet-12-384(Baseline) | 15.93M | PatternNet | 99.74 | 17.11ms | 117.20mJ | - |
| Distillation | 4.56M | PatternNet | 98.93 | 16.47ms | 56.02mJ | 52.20% |
| Dynn | 16.09M | PatternNet | 99.16 | 12.32ms | 65.29mJ | 44.29% |
| **Distillation+dynn** | **4.72M** | PatternNet | **98.32** | 14.00ms | **44.79mJ** | **61.78%** |

### NVIDIA Jetson Orin Nano
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 28.71ms | 58.88mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 16.53ms | 32.63mJ | 44.58% |
| Dynn | 14.93M | NaSC | 97.37 | 10.87ms | 27.17mJ | **53.10%** |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 16.34ms | **27.77mJ** | **52.83%** |
| GFNet-12-384(Baseline) | 15.93M | PatternNet | 99.74 | 18.77ms | 107.01mJ | - |
| Distillation | 4.56M | PatternNet | 98.93 | 18.74ms | 52.49mJ | 50.94% |
| Dynn | 16.09M | PatternNet | 99.16 | 14.00ms | 60.2mJ | 43.74% |
| **Distillation+dynn** | **4.72M** | PatternNet | **98.32** | 16.30ms | **39.12mJ** | **63.44%** |

```javascript
PatternNet
EXIT CORRECT RATE:
[0.0, 0.0, 99.02459711620017, 0.0, 99.10714285714286, 0.0, 0.0, 99.76905311778292, 0.0, 99.63768115942028, 0.0, 98.04347826086956]
[0, 0, 2358, 0, 1568, 0, 0, 866, 0, 828, 0, 460]
[0.0, 0.0, 38.7828947368421, 0.0, 25.789473684210527, 0.0, 0.0, 14.24342105263158, 0.0, 13.618421052631579, 0.0, 7.565789473684211]
* Acc@1 99.161 Acc@5 99.885 loss 0.258

NaSC
EXIT CORRECT RATE:
[0.0, 99.09836065573771, 0.0, 0.0, 97.7914740626605, 0.0, 0.0, 0.0, 0.0, 94.38902743142144, 0.0, 80.64516129032258]
[0, 1220, 0, 0, 1947, 0, 0, 0, 0, 802, 0, 31]
[0.0, 30.5, 0.0, 0.0, 48.675000000000004, 0.0, 0.0, 0.0, 0.0, 20.05, 0.0, 0.775]
* Acc@1 97.375 Acc@5 100.000 loss 0.182
```

| **PatternNet-dynn-distillation** | Layer2 | Layer4 | Layer7  | Layer9  | Last Layer  |
| --- | --- | --- | --- | --- | --- |
| Exit Rate | 38.78% | 25.78% | 14.24% | 13.61% | 7.56% |
| Exit Accuracy | 99.0 | 99.1 | 99.7 | 99.6 | 98.0 |

| **NaSC-dynn-distillation** | Layer1 | Layer4 | Layer9  | Last Layer  |
| --- | --- | --- | --- | --- |
| Exit Rate | 30.5% | 48.67% | 20.05% | 0.77% |
| Exit Accuracy | 99.0 | 97.7 | 94.3 | 80.6 |

### Raspberry Pi 4 Model B
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 381.6ms | 412.12mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 156.5ms | 151.80mJ | 63.16% |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 137.9ms | **133.76mJ** | **67.54%** |
| GFNet-12-384(Baseline) | 15.93M | PatternNet | 99.74 | 1149.06ms | 1378.87mJ | - |
| Distillation | 4.56M | PatternNet | 98.93 | 495.96ms | 500.91mJ | 63.67% |
| **Distillation+dynn** | **4.72M** | PatternNet | **98.32** | 359.86ms | **363.45mJ** | **73.64%** |


| name | Params | dataset | acc@1 |
| --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 22.82M | AID | 96.15 |
| Distillation | 8.11M | AID | 93.85 |
| **Distillation+dynn** | **3.94M** | AID | **93.80** |

```javascript
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 98.37837837837839, 0.0, 100.0, 0.0, 89.29219600725953]
[0, 0, 0, 0, 0, 0, 0, 370, 0, 528, 0, 1102]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 18.5, 0.0, 26.400000000000002, 0.0, 55.1]
* Acc@1 93.800 Acc@5 99.350 loss 0.291
Avg Forward Time per Image: 42.33863866329193 ms
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   296                                               @profile
   297                                               def forward(self, x):
   298      4000      38976.5      9.7      0.0          B = x.shape[0]
   299      4000    5305096.0   1326.3      3.5          x = self.patch_embed(x)
   300      4000     186929.9     46.7      0.1          cls_tokens = self.cls_token.expand(B, -1, -1)
   301      4000      91174.2     22.8      0.1          dist_token = self.dist_token.expand(B, -1, -1)
   302      4000     708915.8    177.2      0.5          x = torch.cat((cls_tokens, dist_token, x), dim=1)
   303      4000     330904.9     82.7      0.2          x = x + self.pos_embed
   304      4000     343955.0     86.0      0.2          x = self.pos_drop(x)
   305     45138     302128.1      6.7      0.2          for blk_idx, blk in enumerate(self.blocks):
   306     42932   89970421.4   2095.6     58.6              x = blk.forward(x)
   307                                                       # if blk_idx in [8, 9, 10]: 
   308     42932     131236.1      3.1      0.1              if blk_idx in [7,  9]: ##UCM
   309                                                       # if blk_idx in [ 9, 10]:##NWPU
   310      7260    1317065.1    181.4      0.9                  inter_z = self.norm(x)
   311      7260   42799234.7   5895.2     27.9                  inter_z = inter_z.to('cpu')
   312      7260    2607606.7    359.2      1.7                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:,0])
   313      7260    6800963.5    936.8      4.4                  g = self.gates[blk_idx](inter_logit)
   314      7260     173555.0     23.9      0.1                  g = torch.sigmoid(g)
   315      7260     528112.0     72.7      0.3                  if g >= self.threshold: 
   316      1794       6092.7      3.4      0.0                      return inter_logit, blk_idx
   317                                           
   318      2206     293334.1    133.0      0.2          x = self.norm(x)
   319      2206    1642060.3    744.4      1.1          x = (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2
   320      2206      60029.4     27.2      0.0          return x, len(self.blocks) - 1
```
```javascript
311      7260   42799234.7   5895.2     27.9                  inter_z = inter_z.to('cpu')
311     11088    1402677.9    126.5      1.3                  inter_z = inter_z.to('cpu')
311      6920     714044.3    103.2      1.1                  inter_z = inter_z.to('cpu')
```
| shape | image_size | dataset | time(to CPU)  |
| --- | --- | --- | --- |
| [1,66,192] | 128*128*3 | NaSC | 103.2 |
| [1,258,192] | 256*256*3 | NWPU | 126.5 |
| [1,1602,192] | 600*600*3 | AID | 5895.2 |


| name | Params | dataset | acc@1 |
| --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 17.42M | AID | 96.25 |
| Distillation | 5.40M | AID | 92.85 |
| **Distillation+dynn** | **5.46M** | AID | **90.45** |
