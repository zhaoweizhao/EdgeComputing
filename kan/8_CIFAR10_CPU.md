![image](https://github.com/user-attachments/assets/df404fd0-95df-47fc-9166-e02fda69ab6f)![image](https://github.com/user-attachments/assets/0bd8f02e-dabc-4d11-98f4-e9ab2ef7155c)
![image](https://github.com/user-attachments/assets/f225ad8f-b321-46bb-8a2d-a4beb2e7653a)
![image](https://github.com/user-attachments/assets/4fadf51b-b881-4742-9a20-6dee8077a922)
### distil-12-192-dynn 蒸馏+早退 new
```javascript
power:14.0w-8.0w
avg_time:51.68ms
100%|██████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [09:08<00:00, 18.23it/s]
FPS: 18.22936137046163
elapsed_time_ms: 54.85655694007874
Avg Forward Time per Image: 51.29515073299408 ms
100%|██████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [09:14<00:00, 18.03it/s]
FPS: 18.029388077430863
elapsed_time_ms: 55.46499946117401
Avg Forward Time per Image: 51.867917799949645 ms
100%|██████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [09:15<00:00, 18.01it/s]
FPS: 18.01308134118065
elapsed_time_ms: 55.51521036624908
Avg Forward Time per Image: 51.89518051147461 ms
```
### distil-12-192-dynn 蒸馏+早退
```javascript
100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [09:09<00:00, 18.20it/s]
FPS: 18.1961751989434
elapsed_time_ms: 54.95660429000854
Avg Forward Time per Image: 51.371874523162845 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

Total time: 518.247 s

100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [09:08<00:00, 18.25it/s]
FPS: 18.245875633000733
elapsed_time_ms: 54.80690650939941
Avg Forward Time per Image: 51.206229019165036 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

Total time: 516.253 s

100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [09:05<00:00, 18.33it/s]
FPS: 18.32792903335813
elapsed_time_ms: 54.56153819561005
Avg Forward Time per Image: 51.01166038513183 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

Total time: 514.56 s
```
#########################################################################################
### distil-12-192-dynn 蒸馏
```javascript

100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [10:25<00:00, 16.00it/s]
FPS: 15.99879882675511
elapsed_time_ms: 62.50469243526459
Avg Forward Time per Image: 58.85108413696289 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [10:15<00:00, 16.26it/s]
FPS: 16.2586659327782
elapsed_time_ms: 61.50566129684448
Avg Forward Time per Image: 57.87736392021179 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

100%|██████████████████████████████████████████████████████████████████████████| 10000/10000 [10:14<00:00, 16.26it/s]
FPS: 16.260120726801713
elapsed_time_ms: 61.5001583814621
Avg Forward Time per Image: 57.86465232372284 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s
```
### GFNet-xs cpu
```javascript

100%|████████████████████████████████████████████████████████████████████████| 10000/10000 [21:36<00:00,  7.71it/s]
FPS: 7.710776179945299
elapsed_time_ms: 129.6886301279068
Avg Forward Time per Image: 125.70509548187256 ms
Wrote profile results to CPUtttt.py.lprof
Timer unit: 1e-06 s

```
| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.6M | CIFAR10 | 98.47 | 125.7ms | 842.19mJ/0% |
| GFNet-XS-distil/12 | 4.25M | CIFAR10 | 97.10 | 58.1ms | 360.22mJ/57.23% |
| GFNet-XS-distil-dynn/12 | 4.29M | CIFAR10 | 96.46 | 51.2ms | 317.44mJ/62.31% |
| GFNet-XS-distil-dynn-new/12 | 4.29M | CIFAR10 | 96.46 | 51.68ms | 310.08mJ/62.31% |

| name | Params | dataset | acc@1 |
| --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.6M | CIFAR10 | 98.47 |
| GFNet-distil-192/12 | 4.25M | CIFAR10 | 97.10 |
| GFNet-XS/12(Baseline) | 15.6M | CIFAR100 | 89.20 |
| GFNet-distil-256/11 | 6.71M | CIFAR100 | 83.24 |
| GFNet-distil-192/14 | 4.29M | CIFAR100 | 82.78 |

| name | Params | dataset | acc@1 |
| --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.6M | CIFAR10 | 98.47 |
| GFNet-XS-KAN/12 | 5.8M | CIFAR10 | 93.38 |
| GFNet-tiny-distil/12 | 4.2M | CIFAR10 | 97.10 |
