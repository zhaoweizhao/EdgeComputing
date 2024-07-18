## CIFAR10 GPU
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
