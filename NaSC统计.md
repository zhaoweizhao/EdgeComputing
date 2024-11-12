```javascript
NaSC
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.35185185185185, 0.0, 98.80287310454908, 99.48761742100768, 88.50806451612904]
[0, 0, 0, 0, 0, 0, 0, 1080, 0, 1253, 1171, 496]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.0, 0.0, 31.324999999999996, 29.275000000000002, 12.4]
* Acc@1 97.875 Acc@5 100.000 loss 0.115
```
|  | Layer7 | Layer9 | Layer10  | Last Layer  |
| --- | --- | --- | --- | --- |
| Exit Rate | 27.0% | 31.32% | 29.28% | 12.4% |
| Exit Accuracy | 99.3 | 98.8 | 99.4 | 88.5 |

```javascript
PatternNet
[0.0, 0.0, 0.0, 0.0, 0.0, 99.7789240972734, 0.0, 99.34667211106574, 0.0, 97.50120134550697, 0.0, 83.93782383419689]
[0, 0, 0, 0, 0, 1357, 0, 2449, 0, 2081, 0, 193]
[0.0, 0.0, 0.0, 0.0, 0.0, 22.31907894736842, 0.0, 40.2796052631579, 0.0, 34.22697368421053, 0.0, 3.174342105263158]
* Acc@1 98.322 Acc@5 99.885 loss 0.112
```
### NVIDIA Jetson AGX Orin
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 19.47ms | 61.34mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 16.44ms | 37.28mJ | 39.22% |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 14.02ms | **35.06mJ** | **42.84%** |
| GFNet-12-384(Baseline) | 15.93M | PatternNet | 99.74 | 19.47ms | 61.34mJ | - |
| Distillation | 4.56M | PatternNet | 98.93 | 16.44ms | 37.28mJ | 39.22% |
| **Distillation+dynn** | **4.72M** | PatternNet | **98.32** | 14.02ms | **35.06mJ** | **42.84%** |

### NVIDIA Jetson Orin Nano
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 28.71ms | 58.88mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 16.53ms | 32.63mJ | 44.58% |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 16.34ms | **27.77mJ** | **52.83%** |

### Raspberry Pi 4 Model B
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 381.6ms | 412.12mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 156.5ms | 151.80mJ | 63.16% |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 137.9ms | **133.76mJ** | **67.54%** |

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
