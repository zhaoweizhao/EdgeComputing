## Jetson AGX Orin
### distil_12_192+Dynn GPU(new)
```javascript
threshold = 0.45
Total number of parameters: 4758703
power:11.5w - 8.0w
[0, 0, 0, 0, 0, 0, 0, 0, 0, 4207, 1338, 755]
* Acc@1 93.889 Acc@5 99.667 loss 0.243
100%|████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [01:57<00:00, 53.57it/s]
FPS: 53.56856496461078
elapsed_time_ms: 18.667664527893066
Avg Forward Time per Image: 13.99701856431507 ms
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [01:57<00:00, 53.79it/s]
FPS: 53.790628001866914
elapsed_time_ms: 18.590599090333967
Avg Forward Time per Image: 13.946038162897503 ms
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [01:58<00:00, 53.03it/s]
FPS: 53.02931703297689
elapsed_time_ms: 18.857493476262167
Avg Forward Time per Image: 14.16229490249876 ms
100%|████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [01:59<00:00, 52.84it/s]
FPS: 52.84175770741491
elapsed_time_ms: 18.924427259536017
Avg Forward Time per Image: 14.274525226108612 ms
```
### distil_12_192 GPU
```javascript
power:11.5w-8.0w
avg_time:16.75ms
Total number of parameters: 4567578
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:12<00:00, 47.57it/s]
FPS: 47.573175931284744
elapsed_time_ms: 21.020248920198473
Avg Forward Time per Image: 16.349544108860076 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:11<00:00, 47.78it/s]
FPS: 47.77747631104366
elapsed_time_ms: 20.930364623902335
Avg Forward Time per Image: 16.247585917276048 ms
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:15<00:00, 46.55it/s]
FPS: 46.5541094898093
elapsed_time_ms: 21.48038080760411
Avg Forward Time per Image: 16.77488126452007 ms
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:16<00:00, 46.08it/s]
FPS: 46.07530834380318
elapsed_time_ms: 21.70359865067497
Avg Forward Time per Image: 17.06746986934117 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:19<00:00, 45.29it/s]
FPS: 45.291446487946196
elapsed_time_ms: 22.07922417020041
Avg Forward Time per Image: 17.347342097570024 ms
```
### distil_12_192+Dynn GPU
```javascript
Total number of parameters: 4758703
power:11.0w - 8.0w
avg_time:16.35ms
[0, 0, 0, 0, 0, 0, 0, 0, 0, 4199, 1535, 566]
* Acc@1 94.429 Acc@5 99.587 loss 0.275

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:10<00:00, 48.23it/s]
FPS: 48.224903412230475
elapsed_time_ms: 20.736174242837087
Avg Forward Time per Image: 16.03307387185475 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:13<00:00, 47.36it/s]
FPS: 47.36061215510977
elapsed_time_ms: 21.114591946677557
Avg Forward Time per Image: 16.47583492218502 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:13<00:00, 47.12it/s]
FPS: 47.121905228332345
elapsed_time_ms: 21.221552803402854
Avg Forward Time per Image: 16.542862672654408 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:11<00:00, 48.07it/s]
FPS: 48.07245128661769
elapsed_time_ms: 20.80193485532488
Avg Forward Time per Image: 16.118425603896853 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:14<00:00, 46.89it/s]
FPS: 46.89318210923016
elapsed_time_ms: 21.325061661856513
Avg Forward Time per Image: 16.62572501197694 ms
```
### GFNet-xs GPU
```javascript
Total number of parameters: 15936045
power:13.2w-8.0w
avg_time:26.75ms
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:18<00:00, 31.81it/s]
FPS: 31.8145493599257
elapsed_time_ms: 31.432159817408003
Avg Forward Time per Image: 26.670436291467574 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:17<00:00, 31.92it/s]
FPS: 31.91522622388788
elapsed_time_ms: 31.333006790706087
Avg Forward Time per Image: 26.605402969178698 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:21<00:00, 31.34it/s]
FPS: 31.341349430479436
elapsed_time_ms: 31.906730825938872
Avg Forward Time per Image: 27.188821747189476 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:18<00:00, 31.82it/s]
FPS: 31.817196390715072
elapsed_time_ms: 31.42954481972588
Avg Forward Time per Image: 26.67081390108381 ms

100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:18<00:00, 31.80it/s]
FPS: 31.797994948046984
elapsed_time_ms: 31.448523771195184
Avg Forward Time per Image: 26.674143700372603 ms
```
| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.93M | RESISC45 | 96.54 | 26.75ms | 139.10mJ/0% |
| GFNet-XS-distil/12 | 4.56M | RESISC45 | 94.97 | 16.75ms | 58.62mJ/57.85% |
| GFNet-XS-distil-dynn/12 | 4.75M | RESISC45 | 94.43 | 16.35ms | 49.05mJ/64.74% |
| GFNet-XS-distil-dynn-new/12 | 4.56M | RESISC45 | 93.89 | 14.09ms | 49.31mJ/64.74% |

## Orin Nano
### distil_12_192+Dynn GPU
```javascript
[0, 0, 0, 0, 0, 0, 0, 0, 0, 4208, 1338, 754]
* Acc@1 93.889 Acc@5 99.667 loss 0.243
power:7.7w-5.0w
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:27<00:00, 42.68it/s]
FPS: 42.642662872540555
elapsed_time_ms: 23.45069310021779
Avg Forward Time per Image: 17.633168167538113 ms
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:29<00:00, 42.03it/s]
FPS: 42.030787809140215
elapsed_time_ms: 23.792083187708776
Avg Forward Time per Image: 17.928702944800968 ms
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:25<00:00, 43.21it/s]
FPS: 43.20983776575422
elapsed_time_ms: 23.14287791176448
Avg Forward Time per Image: 17.354166924007355 ms
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:31<00:00, 41.71it/s]
FPS: 41.71036198977928
elapsed_time_ms: 23.97485786014133
Avg Forward Time per Image: 18.146728220440092 ms
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:27<00:00, 42.78it/s]
FPS: 42.7762113479727
elapsed_time_ms: 23.377479409414622
Avg Forward Time per Image: 17.495989837343732 ms
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   297                                               @profile
   298                                               def forward(self, x):
   299     12600      84731.0      6.7      0.0          B = x.shape[0]
   300     12600   15279101.3   1212.6      6.4          x = self.patch_embed(x)
   301     12600     426605.0     33.9      0.2          cls_tokens = self.cls_token.expand(B, -1, -1)
   302     12600     233178.3     18.5      0.1          dist_token = self.dist_token.expand(B, -1, -1)
   303     12600    1886273.7    149.7      0.8          x = torch.cat((cls_tokens, dist_token, x), dim=1)
   304     12600     964960.7     76.6      0.4          x = x + self.pos_embed
   305     12600     755912.5     60.0      0.3          x = self.pos_drop(x)
   306                                                   # threshold_tensor = torch.tensor(self.threshold, device=x.device)
   307    133199     521147.2      3.9      0.2          for blk_idx, blk in enumerate(self.blocks):
   308    131692  195581438.8   1485.1     82.1              x = blk.forward(x)
   309                                                       # if blk_idx in [7, 9, 10]: ##cifar10
   310                                                       # if blk_idx in [7, 8, 9]: ##UCM
   311    131692     272436.9      2.1      0.1              if blk_idx in [ 9, 10]:
   312
   313     16785    1838569.7    109.5      0.8                  inter_z = self.norm(x)
   314     16785     878882.4     52.4      0.4                  inter = inter_z[:, 0]
   315     16785    2324835.3    138.5      1.0                  inter = inter.to('cpu')
   316     16785    2186398.0    130.3      0.9                  inter_logit = self.intermediate_heads[blk_idx](inter)
   317                                                           # inter_logit = (self.intermediate_heads[blk_idx](inter_z[:, 0]) + self.intermediate_heads_dist[blk_idx](inter_z[:, 1])) / 2
   318                                                           # inter_logit = inter_logit.to('cpu')
   319     16785   12401614.5    738.9      5.2                  g = self.gates[blk_idx](inter_logit)
   320     16785     381074.7     22.7      0.2                  g = torch.sigmoid(g)
   321     16785     924169.2     55.1      0.4                  if g >= self.threshold:  # 用torch.jit.script时可以使用.item()
   322     11093      26060.0      2.3      0.0                      return inter_logit, blk_idx
   323
   324      1507     167133.2    110.9      0.1          x = self.norm(x)
   325                                                   # x = x.to('cpu')
   326      1507     942146.0    625.2      0.4          x = (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2
   327      1507      26190.0     17.4      0.0          return x, len(self.blocks) - 1
```
### distil_12_192 GPU
```javascript
power:8.1w-5.0w
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:23<00:00, 44.00it/s]
FPS: 44.00220650853542
elapsed_time_ms: 22.726133058941553
Avg Forward Time per Image: 17.064007312532457 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:22<00:00, 44.11it/s]
FPS: 44.06709554939418
elapsed_time_ms: 22.692668702867294
Avg Forward Time per Image: 16.932225265200177 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:22<00:00, 44.13it/s]
FPS: 44.086762390149396
elapsed_time_ms: 22.68254563921974
Avg Forward Time per Image: 16.95811381415715 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:26<00:00, 43.02it/s]
FPS: 43.01433320246717
elapsed_time_ms: 23.24806466935173
Avg Forward Time per Image: 17.593441955626957 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [02:22<00:00, 44.31it/s]
FPS: 44.308458596436154
elapsed_time_ms: 22.56905411917066
Avg Forward Time per Image: 16.908745917062912 ms
```
### GFNet-xs GPU
```javascript
power:9.2w-5.0w
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [04:05<00:00, 25.68it/s]
FPS: 25.669037189261154
elapsed_time_ms: 38.95744092880733
Avg Forward Time per Image: 33.189417407626195 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:53<00:00, 27.03it/s]
FPS: 27.00945925831516
elapsed_time_ms: 37.02406591839261
Avg Forward Time per Image: 31.20404777072725 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:59<00:00, 26.31it/s]
FPS: 26.313379761919762
elapsed_time_ms: 38.00347994244288
Avg Forward Time per Image: 32.32547877326844 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [04:00<00:00, 26.24it/s]
FPS: 26.24375926840441
elapsed_time_ms: 38.1042970929827
Avg Forward Time per Image: 32.290588674091154 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6300/6300 [03:51<00:00, 27.27it/s]
FPS: 27.252015656734528
elapsed_time_ms: 36.69453344647847
Avg Forward Time per Image: 31.10201074963524 ms
```
| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.93M | RESISC45 | 96.54 | 32.01ms | 134.47mJ/0% |
| GFNet-XS-distil/12 | 4.56M | RESISC45 | 94.97 | 17.08ms | 52.96mJ/60.61% |
| GFNet-XS-distil-dynn/12 | 4.75M | RESISC45 | 93.89 | 17.70ms | 38.95mJ/71.03% |
