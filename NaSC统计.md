
## Nano
```javascript
<<<<<<+++++++++++++GFNet-xs: Time = 28.71ms Energy = 58.88mJ +++++++++++++>>>>>>
<<<<<<+++++++++++++Distillation: Time = 16.53ms Energy = 32.63mJ +++++++++++++>>>>>>
<<<<<<+++++++++++++Distillation + DYNN: Time = 16.34ms Energy = 27.77mJ +++++++++++++>>>>>>
```
## AGX
```javascript
<<<<<<+++++++++++++GFNet-xs: Time = 19.47ms Energy = 61.34mJ +++++++++++++>>>>>>
<<<<<<+++++++++++++Distillation: Time = 14.91ms Energy = 37.28mJ +++++++++++++>>>>>>
<<<<<<+++++++++++++Distillation + DYNN: Time = 14.02ms Energy = 35.06mJ +++++++++++++>>>>>>
```

```javascript
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 99.35185185185185, 0.0, 98.80287310454908, 99.48761742100768, 88.50806451612904]
[0, 0, 0, 0, 0, 0, 0, 1080, 0, 1253, 1171, 496]
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.0, 0.0, 31.324999999999996, 29.275000000000002, 12.4]
* Acc@1 97.875 Acc@5 100.000 loss 0.115
```

```javascript
3.26w - 2.18w
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [26:20<00:00,  2.53it/s]
FPS: 2.5309006282531223
elapsed_time_ms: 395.1162636876106
Avg Forward Time per Image: 381.65567296743393 ms

3.15w - 2.18w
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [11:11<00:00,  5.96it/s]
FPS: 5.956094535437651
elapsed_time_ms: 167.89525318145752
Avg Forward Time per Image: 156.52790904045105 ms

100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4000/4000 [09:58<00:00,  6.69it/s]
FPS: 6.688610805748307
elapsed_time_ms: 149.50787675380707
Avg Forward Time per Image: 137.98511749505997 ms
```
### NVIDIA Jetson AGX Orin
| name | Params | dataset | acc@1 | latency | energy consumption | improve |
| --- | --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 14.89M | NaSC | 99.10 | 19.47ms | 61.34mJ | - |
| Distillation | 3.92M | NaSC | 98.28 | 16.44ms | 37.28mJ | 39.22% |
| **Distillation+dynn** | **3.94M** | NaSC | **97.88** | 14.02ms | **35.06mJ** | **42.84%** |

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

