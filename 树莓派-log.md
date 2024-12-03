

| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-12-384(Baseline) | 15.6M | CIFAR10 | 98.47 | 919.5ms | 1002.2mJ/0% |
| GFNet-distil-12-192 | 4.25M | CIFAR10 | 97.10 | 347.0ms | 360.8mJ/63.9% |
| GFNet-dynn-12-192 | 4.27M | CIFAR10 | 96.06 | 313.2ms | **325.7mJ/67.5%** |
| GFNet-12-384(Baseline) | 15.93M | RESISC45 | 96.54 | 1135.6ms | 1237.8mJ/0% |
| GFNet-distil-12-192 | 4.56M | RESISC45 | 94.97 | 518.0ms | 538.7mJ/56.4% |
| GFNet-dynn-12-192 | 4.75M | RESISC45 | 94.43 | 741.78ms | **856.90mJ/61.2%** |
| GFNet-dynn-12-192 | 4.75M | RESISC45 | 94.43 | 460.8ms | **479.2mJ/61.2%** |
| GFNet-12-384(Baseline) | 15.92M | UMC | 99.52 | 1131.7ms | 1233.5mJ/0% |
| GFNet-distil-12-192 | 4.55M | UMC | 98.57 | 526.2ms | 547.2mJ/55.6% |
| GFNet-dynn-12-192 | 4.64M | UMC | 96.66 | 412.6ms | **429.1mJ/65.2%** |
