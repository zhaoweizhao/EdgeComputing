## CIFAR10 GPU Jetson AGX Orin
### distil-12-192-dynn 蒸馏+早退 new
```javascript
power:11.0w-8.0w
Total number of parameters: 4278137
[0, 0, 0, 0, 0, 0, 0, 0, 2706, 3307, 962, 3025]
* Acc@1 96.060 Acc@5 99.880 loss 0.173
100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [02:59<00:00, 55.85it/s]
FPS: 55.845209279825376
elapsed_time_ms: 17.906638956069948
Avg Forward Time per Image: 15.251304745674133 ms

100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [03:10<00:00, 52.44it/s]
FPS: 52.44264112071769
elapsed_time_ms: 19.068452286720277
Avg Forward Time per Image: 16.388077521324156 ms

100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [03:06<00:00, 53.71it/s]
FPS: 53.71360134598117
elapsed_time_ms: 18.617258477211
Avg Forward Time per Image: 15.933584666252136 ms

100%|███████████████████████████████████████████████████████████████████████████| 10000/10000 [03:10<00:00, 52.60it/s]
FPS: 52.60282711651317
elapsed_time_ms: 19.010385084152222
Avg Forward Time per Image: 16.34760754108429 ms

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   327                                               @profile #FLOWERS
   328                                               def forward(self, x):
   329     10000      51926.0      5.2      0.0          B = x.shape[0]
   330     10000    5699237.8    569.9      3.5          x = self.patch_embed(x)
   331     10000     319162.9     31.9      0.2          cls_tokens = self.cls_token.expand(B, -1, -1)
   332     10000     216426.6     21.6      0.1          dist_token = self.dist_token.expand(B, -1, -1)
   333     10000     823641.1     82.4      0.5          x = torch.cat((cls_tokens, dist_token, x), dim=1)
   334     10000     645398.4     64.5      0.4          x = x + self.pos_embed
   335     10000     580972.7     58.1      0.4          x = self.pos_drop(x)
   336                                                   # threshold_tensor = torch.tensor(self.threshold, device=x.device)
   337    107330     335088.2      3.1      0.2          for blk_idx, blk in enumerate(self.blocks):
   338    104305  129516143.8   1241.7     79.9              x = blk.forward(x)
   339    104305     139663.0      1.3      0.1              if blk_idx == 8:
   340                                                           # x=x.to('cpu')
   341     10000    1060740.2    106.1      0.7                  inter_z = self.norm(x)
   342     10000    1171539.6    117.2      0.7                  inter_z = inter_z.to('cpu')
   343     10000    1506421.3    150.6      0.9                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:, 0])
   344                                                           # inter_logit = inter_logit.to('cpu')
   345     10000    6386508.0    638.7      3.9                  g = self.gates[blk_idx](inter_logit)
   346     10000     199666.3     20.0      0.1                  g = torch.sigmoid(g)
   347     10000     482395.4     48.2      0.3                  if g >= 0.55:
   348      2706       6065.4      2.2      0.0                      return inter_logit, blk_idx
   349     94305      61821.5      0.7      0.0              elif blk_idx == 9:
   350                                                           # x=x.to('cpu')
   351      7294     818107.4    112.2      0.5                  inter_z = self.norm(x)
   352      7294     786400.0    107.8      0.5                  inter_z = inter_z.to('cpu')
   353      7294     980239.3    134.4      0.6                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:, 0])
   354                                                           # inter_logit = inter_logit.to('cpu')
   355      7294    4161371.7    570.5      2.6                  g = self.gates[blk_idx](inter_logit)
   356      7294     125493.7     17.2      0.1                  g = torch.sigmoid(g)
   357      7294     317354.7     43.5      0.2                  if g >= 0.55:
   358      3308       7393.3      2.2      0.0                      return inter_logit, blk_idx
   359     87011      73377.3      0.8      0.0              elif blk_idx == 10:
   360      3986     445469.6    111.8      0.3                  inter_z = self.norm(x)
   361      3986     419412.8    105.2      0.3                  inter_z = inter_z.to('cpu')
   362      3986     529963.5    133.0      0.3                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:, 0])
   363      3986    2244101.4    563.0      1.4                  g = self.gates[blk_idx](inter_logit)
   364      3986      67276.9     16.9      0.0                  g = torch.sigmoid(g)
   365      3986     172363.8     43.2      0.1                  if g >= 0.65:
   366       961       2157.0      2.2      0.0                      return inter_logit, blk_idx
   367
   368      3025     327190.4    108.2      0.2          x = self.norm(x)
   369                                                   # x = x.to('cpu')
   370      3025    1312756.1    434.0      0.8          x = (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2
   371      3025      56833.2     18.8      0.0          return x, len(self.blocks) - 1
```
### distil-12-192-dynn 蒸馏+早退
```javascript
[0, 0, 0, 0, 0, 0, 0, 2396, 0, 4217, 1666, 1721]
* Acc@1 96.460 Acc@5 99.890 loss 0.153
power:11.0w-8.0w
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:05<00:00, 53.77it/s]
FPS: 53.7719258510785
elapsed_time_ms: 18.597064995765685
Avg Forward Time per Image: 15.954345440864563 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:10<00:00, 52.54it/s]
FPS: 52.54080747852286
elapsed_time_ms: 19.032825112342834
Avg Forward Time per Image: 16.342974972724914 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:07<00:00, 53.23it/s]
FPS: 53.22958555236487
elapsed_time_ms: 18.786544919013977
Avg Forward Time per Image: 16.130971121788026 ms

Total time: 8.60564 s
File: /home/nvidia/heShaoWei/GFNet/dynn/gate/learnable_uncertainty_gate.py
Function: forward at line 43

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    43                                               @profile
    44                                               def forward(self, logits: Tensor) -> Tensor:
    45                                                   # 使用log_softmax来进一步优化计算熵的效率
    46                                                   # print("device:", logits.device)
    47
    48
    49     21301     599999.2     28.2      7.0          log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    50     21301     361486.3     17.0      4.2          probs = torch.exp(log_probs)  # 等效于原来的 softmax
    51
    52                                                   # 优化top2概率的提取，避免冗余计算
    53     21301     632561.6     29.7      7.4          top2probs, top2indices = torch.topk(probs, 2, dim=1)
    54     21301     944161.1     44.3     11.0          p_max, next_p_max = top2probs[:, 0], top2probs[:, 1]
    55
    56                                                   # 计算margins
    57     21301     340610.4     16.0      4.0          margins = p_max - next_p_max
    58
    59                                               # 直接利用log_probs计算熵，避免重新计算log(probs)
    60     21301    1145815.7     53.8     13.3          entropy = -torch.sum(probs * log_probs, dim=1)
    61
    62                                               # 优化幂概率计算
    63     21301     630322.0     29.6      7.3          logits_scaled = logits * 2
    64     21301     489838.6     23.0      5.7          pow_probs = torch.nn.functional.softmax(logits_scaled, dim=1)  # 优化计算方式，直接调整logits
    65     21301     365280.0     17.1      4.2          log_pow_probs = torch.nn.functional.log_softmax(logits_scaled, dim=1)  # 直接使用log_softmax
    66     21301     898668.4     42.2     10.4          entropy_pow = -torch.sum(pow_probs * log_pow_probs, dim=1)  # 利用幂概率的log版本
    67
    68                                                   # 合并概率矩阵，无需单独添加维度
    69     21301     726305.2     34.1      8.4          uncertainty_metrics = torch.stack([p_max, entropy, margins, entropy_pow], dim=1)
    70                                                   # print("uncertainty_metrics:",uncertainty_metrics.device)
    71                                                   # print("self.linear:",self.linear.device)
    72                                               # 将结果移至适当设备并调整类型，以便与logits兼容
    73     21301    1470589.1     69.0     17.1          return self.linear(uncertainty_metrics)
    74                                                   # return self.linear(uncertainty_metrics.to(logits.device, dtype=logits.dtype))

Total time: 160.611 s
File: /home/nvidia/heShaoWei/GFNet/dynn/gfnet_dynn.py
Function: forward at line 296

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   296                                               @profile
   297                                               def forward(self, x):
   298     10100      48311.4      4.8      0.0          B = x.shape[0]
   299     10100    5665986.3    561.0      3.5          x = self.patch_embed(x)
   300     10100     302742.0     30.0      0.2          cls_tokens = self.cls_token.expand(B, -1, -1)
   301     10100     200435.6     19.8      0.1          dist_token = self.dist_token.expand(B, -1, -1)
   302     10100     811436.8     80.3      0.5          x = torch.cat((cls_tokens, dist_token, x), dim=1)
   303     10100     641410.2     63.5      0.4          x = x + self.pos_embed
   304     10100     545615.1     54.0      0.3          x = self.pos_drop(x)
   305                                                   # threshold_tensor = torch.tensor(self.threshold, device=x.device)
   306    103337     310229.8      3.0      0.2          for blk_idx, blk in enumerate(self.blocks):
   307    101515  125621523.1   1237.5     78.2              x = blk.forward(x)
   308    101515     135068.3      1.3      0.1              if blk_idx in [7, 9, 10]: ##cifar10
   309                                                       # if blk_idx in [7, 8, 9]: ##UCM
   310     21290    2280373.8    107.1      1.4                  inter_z = self.norm(x)
   311     21290    2350571.7    110.4      1.5                  inter_z = inter_z.to('cpu')
   312     21290    6509723.2    305.8      4.1                  inter_logit = (self.intermediate_heads[blk_idx](inter_z[:, 0]) + self.intermediate_heads_dist[blk_idx](inter_z[:, 1])) / 2
   313     21290   12983963.1    609.9      8.1                  g = self.gates[blk_idx](inter_logit)
   314     21290     344267.9     16.2      0.2                  g = torch.sigmoid(g)
   315     21290     925595.5     43.5      0.6                  if g >= self.threshold:  # 用torch.jit.script时可以使用.item()
   316      8278      17640.4      2.1      0.0                      return inter_logit, blk_idx
   317
   318      1822     195394.3    107.2      0.1          x = self.norm(x)
   319      1822     195335.8    107.2      0.1          x = x.to('cpu')
   320      1822     492980.1    270.6      0.3          x = (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2
   321      1822      32603.0     17.9      0.0          return x, len(self.blocks) - 1

Total time: 42.9335 s
File: /home/nvidia/heShaoWei/GFNet/gfnet.py
Function: forward at line 62

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    62                                               @profile
    63                                               def forward(self, x, spatial_size=None):
    64    101527     384733.2      3.8      0.9          B, N, C = x.shape
    65                                                   # print("**********",x.shape)
    66    101527      59991.2      0.6      0.1          if spatial_size is None:
    67                                                       # a = b = int(math.sqrt(N))
    68
    69    101527      77881.1      0.8      0.2              if self.distil:
    70    101527      35344.7      0.3      0.1                  a = 11
    71    101527      53459.5      0.5      0.1                  b = 18
    72                                                           # a = 43
    73                                                           # b = 6
    74                                                       else:
    75                                                           self.nnn_const = torch.tensor(N, dtype=torch.float32)
    76                                                           nnn = self.nnn_const.to(device=x.device)
    77                                                           a = b = torch.sqrt(nnn).to(torch.int32)
    78                                                   else:
    79                                                       a, b = spatial_size
    80
    81    101527    1454769.9     14.3      3.4          x = x.view(B, a, b, C)
    82
    83    101527     556091.7      5.5      1.3          x = x.to(torch.float32)
    84
    85    101527   13780574.2    135.7     32.1          x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
    86    101527    2913435.6     28.7      6.8          weight = torch.view_as_complex(self.complex_weight)
    87    101527    4766492.8     46.9     11.1          x = x * weight
    88    101527   16650719.8    164.0     38.8          x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')
    89
    90    101527    2118339.4     20.9      4.9          x = x.reshape(B, N, C)
    91
    92    101527      81655.2      0.8      0.2          return x

Total time: 122.475 s
File: /home/nvidia/heShaoWei/GFNet/gfnet.py
Function: forward at line 104

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   104                                               @profile
   105                                               def forward(self, x):
   106    101527   10647005.7    104.9      8.7          x1 = self.norm1(x)
   107    101527   48612982.5    478.8     39.7          x1 = self.filter(x1)
   108    101527   14762190.9    145.4     12.1          x1 = self.norm2(x1)
   109    101527   42377621.8    417.4     34.6          x1 = self.mlp(x1)
   110    101527    5967661.7     58.8      4.9          x = x + self.drop_path(x1)
   111    101527     107534.2      1.1      0.1          return x
```
### distil-12-192 蒸馏
```javascript
power: 11.6w-8.0w
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:15<00:00, 51.22it/s]
FPS: 51.21638336579713
elapsed_time_ms: 19.525002241134644
Avg Forward Time per Image: 16.90316274166107 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:06<00:00, 53.50it/s]
FPS: 53.499201456652415
elapsed_time_ms: 18.691867780685424
Avg Forward Time per Image: 16.052430415153502 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:10<00:00, 52.47it/s]
FPS: 52.46558473635233
elapsed_time_ms: 19.060113501548766
Avg Forward Time per Image: 16.384952354431153 ms
```

### GFNet-xs
```javascript
power: 13.2w-8.0w = 5.2w
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [04:51<00:00, 34.33it/s]
FPS: 34.334449354268656
elapsed_time_ms: 29.125266861915588
Avg Forward Time per Image: 26.382894468307494 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [04:50<00:00, 34.44it/s]
FPS: 34.43811734195278
elapsed_time_ms: 29.037591981887818
Avg Forward Time per Image: 26.30713641643524 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [04:49<00:00, 34.50it/s]
FPS: 34.500144467726315
elapsed_time_ms: 28.985385870933534
Avg Forward Time per Image: 26.264074826240538 ms
```


| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.6M | CIFAR10 | 98.47 | 26.31ms | 136.81mJ/0% |
| GFNet-XS-distil/12 | 4.25M | CIFAR10 | 97.10 | 16.44ms | 59.18mJ/56.7% |
| GFNet-XS-distil-dynn/12 | 4.29M | CIFAR10 | 96.46 | 16.14ms | 48.42mJ/64.6% |
| GFNet-XS-distil-dynn-new/12 | 4.27M | CIFAR10 | 96.06 | 15.97ms | 47.92mJ/65.0% |

## CIFAR10 GPU Orin Nano
### distil-12-192-dynn 蒸馏+早退
```javascript
power:7.6w-5.0w
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:43<00:00, 44.72it/s]
FPS: 44.68973554447218
elapsed_time_ms: 22.376502966880796
Avg Forward Time per Image: 18.853440928459168 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:39<00:00, 45.60it/s]
FPS: 45.567270448086695
elapsed_time_ms: 21.945576071739197
Avg Forward Time per Image: 18.434002017974855 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:39<00:00, 45.48it/s]
FPS: 45.48307343175289
elapsed_time_ms: 21.986201119422912
Avg Forward Time per Image: 18.480722761154176 ms

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   329                                               @profile #FLOWERS CIFAR10
   330                                               def forward(self, x):
   331     10000      53861.5      5.4      0.0          B = x.shape[0]
   332     10000   12867270.6   1286.7      7.0          x = self.patch_embed(x)
   333     10000     336347.7     33.6      0.2          cls_tokens = self.cls_token.expand(B, -1, -1)
   334     10000     199280.7     19.9      0.1          dist_token = self.dist_token.expand(B, -1, -1)
   335     10000     955725.6     95.6      0.5          x = torch.cat((cls_tokens, dist_token, x), dim=1)
   336     10000     693747.3     69.4      0.4          x = x + self.pos_embed
   337     10000     584916.1     58.5      0.3          x = self.pos_drop(x)
   338                                                   # threshold_tensor = torch.tensor(self.threshold, device=x.device)
   339    107330     432845.7      4.0      0.2          for blk_idx, blk in enumerate(self.blocks):
   340    104305  138751147.9   1330.2     75.8              x = blk.forward(x)
   341    104305     180940.8      1.7      0.1              if blk_idx == 8:
   342                                                           # x=x.to('cpu')
   343     10000    1065921.8    106.6      0.6                  inter_z = self.norm(x)
   344     10000    2034787.1    203.5      1.1                  inter_z = inter_z.to('cpu')
   345     10000    1757611.9    175.8      1.0                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:, 0])
   346                                                           # inter_logit = inter_logit.to('cpu')
   347     10000    7458319.9    745.8      4.1                  g = self.gates[blk_idx](inter_logit)
   348     10000     218122.1     21.8      0.1                  g = torch.sigmoid(g)
   349     10000     554739.6     55.5      0.3                  if g >= 0.55:
   350      2706       6838.8      2.5      0.0                      return inter_logit, blk_idx
   351     94305      97402.2      1.0      0.1              elif blk_idx == 9:
   352                                                           # x=x.to('cpu')
   353      7294     834007.6    114.3      0.5                  inter_z = self.norm(x)
   354      7294    1073430.4    147.2      0.6                  inter_z = inter_z.to('cpu')
   355      7294    1105379.1    151.5      0.6                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:, 0])
   356                                                           # inter_logit = inter_logit.to('cpu')
   357      7294    4818480.7    660.6      2.6                  g = self.gates[blk_idx](inter_logit)
   358      7294     138911.0     19.0      0.1                  g = torch.sigmoid(g)
   359      7294     365224.2     50.1      0.2                  if g >= 0.55:
   360      3308       8525.3      2.6      0.0                      return inter_logit, blk_idx
   361     87011      72487.8      0.8      0.0              elif blk_idx == 10:
   362      3986     452793.0    113.6      0.2                  inter_z = self.norm(x)
   363      3986     580559.7    145.6      0.3                  inter_z = inter_z.to('cpu')
   364      3986     596799.1    149.7      0.3                  inter_logit = self.intermediate_heads[blk_idx](inter_z[:, 0])
   365      3986    2600031.3    652.3      1.4                  g = self.gates[blk_idx](inter_logit)
   366      3986      73264.6     18.4      0.0                  g = torch.sigmoid(g)
   367      3986     195658.8     49.1      0.1                  if g >= 0.65:
   368       961       2420.1      2.5      0.0                      return inter_logit, blk_idx
   369
   370      3025     328288.3    108.5      0.2          x = self.norm(x)
   371                                                   # x = x.to('cpu')
   372      3025    1392229.9    460.2      0.8          x = (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2
   373      3025      52387.1     17.3      0.0          return x, len(self.blocks) - 1
```
### distil-12-192 蒸馏
```javascript
power:8.0w-5.0w
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:38<00:00, 45.86it/s]
FPS: 45.85430145073555
elapsed_time_ms: 21.80820486545563
Avg Forward Time per Image: 18.384045600891113 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:34<00:00, 46.63it/s]
FPS: 46.63007971575901
elapsed_time_ms: 21.445384740829468
Avg Forward Time per Image: 18.026765823364258 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:32<00:00, 46.99it/s]
FPS: 46.98784032747878
elapsed_time_ms: 21.2821017742157
Avg Forward Time per Image: 17.90388193130493 ms

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    22                                               @profile
    23                                               def forward_features(self, x):
    24                                                   # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    25                                                   # with slight modifications to add the dist_token
    26     10100      57887.0      5.7      0.0          B = x.shape[0]
    27     10100   14645044.7   1450.0      8.2          x = self.patch_embed(x)
    28
    29     10100     332165.1     32.9      0.2          cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    30     10100     172988.5     17.1      0.1          dist_token = self.dist_token.expand(B, -1, -1)
    31     10100     928306.2     91.9      0.5          x = torch.cat((cls_tokens, dist_token, x), dim=1)
    32
    33     10100     696027.9     68.9      0.4          x = x + self.pos_embed
    34     10100     531933.3     52.7      0.3          x = self.pos_drop(x)
    35
    36    131300     401601.0      3.1      0.2          for blk in self.blocks:
    37    121200  159270387.2   1314.1     89.0              x = blk(x)
    38
    39     10100    1093610.4    108.3      0.6          x = self.norm(x)
    40     10100     773735.5     76.6      0.4          return x[:, 0], x[:, 1]
```
### GFNet-xs
```javascript
power:9.0w - 5.0w
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [06:11<00:00, 26.94it/s]
FPS: 26.925980009084785
elapsed_time_ms: 37.138852500915526
Avg Forward Time per Image: 33.59483232498169 ms
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [06:10<00:00, 26.96it/s]
FPS: 26.956667439107093
elapsed_time_ms: 37.096573686599726
Avg Forward Time per Image: 33.602933859825136 ms
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [06:16<00:00, 26.55it/s]
FPS: 26.538106482074415
elapsed_time_ms: 37.68166356086731
Avg Forward Time per Image: 34.118512058258055 ms
```
| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.6M | CIFAR10 | 98.47 | 33.76ms | 135.04mJ/0% |
| GFNet-XS-distil/12 | 4.25M | CIFAR10 | 97.10 | 18.10ms | 54.3mJ/59.78% |
| GFNet-XS-distil-dynn-new/12 | 4.27M | CIFAR10 | 96.06 | 18.58ms | 48.32mJ/64.21% |
