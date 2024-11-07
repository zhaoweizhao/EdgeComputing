
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

