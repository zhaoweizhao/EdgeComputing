## UMC AGX Orin
### distil_12_192 GPU
```javascript
name_list = ["gates", "intermediate_heads_dist", "intermediate_heads", "head"]
power:11.5w - 8.0w
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 47.32it/s]
FPS: 47.30812364166981
elapsed_time_ms: 21.13801865350632
Avg Forward Time per Image: 15.590688728150868 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 46.72it/s]
FPS: 46.71511340829287
elapsed_time_ms: 21.40634854634603
Avg Forward Time per Image: 15.84657487415132 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 49.41it/s]
FPS: 49.3953284997942
elapsed_time_ms: 20.24482942762829
Avg Forward Time per Image: 14.703788643791562 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 49.25it/s]
FPS: 49.23629792727157
elapsed_time_ms: 20.310219128926597
Avg Forward Time per Image: 14.75718134925479 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 49.67it/s]
FPS: 49.66247733433417
elapsed_time_ms: 20.135926632654098
Avg Forward Time per Image: 14.628841195787702 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 48.06it/s]
FPS: 48.04930596658711
elapsed_time_ms: 20.811955134073894
Avg Forward Time per Image: 15.256288505735851 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 49.01it/s]
FPS: 48.99531206242581
elapsed_time_ms: 20.410115945906867
Avg Forward Time per Image: 14.839245591844831 ms
```


### distil_12_192_dynn GPU(new)
```javascript
[0, 0, 0, 0, 0, 0, 0, 175, 0, 223, 22, 0]
* Acc@1 96.667 Acc@5 100.000 loss 0.224
power: 11.0W - 8.0w
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 50.49it/s]
FPS: 50.47723733232746
elapsed_time_ms: 19.81090988431658
Avg Forward Time per Image: 14.473512626829601 ms
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 50.43it/s]
FPS: 50.41833296012023
elapsed_time_ms: 19.83405521937779
Avg Forward Time per Image: 14.255831355140323 ms
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 50.89it/s]
FPS: 50.88001976964356
elapsed_time_ms: 19.654080413636706
Avg Forward Time per Image: 14.11615666889009 ms
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 51.49it/s]
FPS: 51.472819527398244
elapsed_time_ms: 19.427729220617387
Avg Forward Time per Image: 13.938233398255848 ms
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:08<00:00, 51.59it/s]
FPS: 51.583543761708256
elapsed_time_ms: 19.386027540479386
Avg Forward Time per Image: 13.861374060312906 ms
```


### GFNet-xs GPU
```javascript
power:13.3w - 8.0w
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 31.85it/s]
FPS: 31.846309397442287
elapsed_time_ms: 31.40081280753726
Avg Forward Time per Image: 25.773693266369047 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 31.62it/s]
FPS: 31.61741259665897
elapsed_time_ms: 31.6281415167309
Avg Forward Time per Image: 25.950872898101807 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 31.57it/s]
FPS: 31.56235559066928
elapsed_time_ms: 31.683313278924853
Avg Forward Time per Image: 26.06432835261027 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 31.34it/s]
FPS: 31.335457254743083
elapsed_time_ms: 31.91273042133876
Avg Forward Time per Image: 26.32360855738322 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 31.88it/s]
FPS: 31.878496840602473
elapsed_time_ms: 31.36910767782302
Avg Forward Time per Image: 25.738737696693057 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 32.09it/s]
FPS: 32.08510880421674
elapsed_time_ms: 31.167106401352655
Avg Forward Time per Image: 25.53037064416068 ms

100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:13<00:00, 31.76it/s]
FPS: 31.76036290455546
elapsed_time_ms: 31.485786324455624
Avg Forward Time per Image: 25.818615300314768 ms
```


| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.92M | UMC | 99.52 | 25.88ms | 137.17mJ/0% |
| GFNet-XS-distil-192/12 | 4.55M | UMC | 98.57 | 15.08ms | 52.79mJ/61.51% |
| GFNet-XS-distil-dynn/12 | 4.64M | UMC | 96.66 | 14.12ms | 42.37mJ/69.11% |

## UMC Orin Nano
### distil_12_192_dynn GPU
```javascript
power:7.6w-5.0w

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 41.09it/s]
FPS: 40.51942008258924
elapsed_time_ms: 24.6795239902678
Avg Forward Time per Image: 16.725584438868932 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 41.58it/s]
FPS: 40.99179227639443
elapsed_time_ms: 24.395127523513068
Avg Forward Time per Image: 17.05195222582136 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:11<00:00, 37.79it/s]
FPS: 37.31083972808531
elapsed_time_ms: 26.801862603142148
Avg Forward Time per Image: 16.810124828701927 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 38.91it/s]
FPS: 38.90053814384736
elapsed_time_ms: 25.706585248311363
Avg Forward Time per Image: 16.71414261772519 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:11<00:00, 38.15it/s]
FPS: 38.132269825154694
elapsed_time_ms: 26.224507604326522
Avg Forward Time per Image: 16.956983293805802 ms
```
### distil_12_192 GPU
```javascript
power:8.0w-5.0w
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 40.28it/s]
FPS: 39.740475253946755
elapsed_time_ms: 25.163262231009345
Avg Forward Time per Image: 17.816160406385148 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 41.81it/s]
FPS: 41.79880030183462
elapsed_time_ms: 23.92413162049793
Avg Forward Time per Image: 17.24575020018078 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 40.77it/s]
FPS: 40.75870650161931
elapsed_time_ms: 24.534635316757928
Avg Forward Time per Image: 17.84526336760748 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 41.55it/s]
FPS: 41.528152987577606
elapsed_time_ms: 24.080049991607666
Avg Forward Time per Image: 17.055674961635045 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:10<00:00, 40.69it/s]
FPS: 40.67243318723091
elapsed_time_ms: 24.58667755126953
Avg Forward Time per Image: 17.811259769258044 ms
```
### GFNet-xs GPU
```javascript
power:9.2w - 5.0w
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:16<00:00, 25.42it/s]
FPS: 25.41101011743554
elapsed_time_ms: 39.35302041825794
Avg Forward Time per Image: 32.04410814103626 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:16<00:00, 25.56it/s]
FPS: 25.34090613052049
elapsed_time_ms: 39.46188801810855
Avg Forward Time per Image: 31.839438279469807 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:17<00:00, 24.64it/s]
FPS: 24.421421735611066
elapsed_time_ms: 40.9476569720677
Avg Forward Time per Image: 32.36601579756964 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:16<00:00, 25.01it/s]
FPS: 24.802947663164485
elapsed_time_ms: 40.3177885782151
Avg Forward Time per Image: 32.83405871618362 ms

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 420/420 [00:16<00:00, 25.92it/s]
FPS: 25.91471940818665
elapsed_time_ms: 38.58810833522252
Avg Forward Time per Image: 31.148235003153484 ms
```
| name | Params | dataset | acc@1 | latency | energy consumption |
| --- | --- | --- | --- | --- | --- |
| GFNet-XS/12(Baseline) | 15.92M | UMC | 99.52 | 32.04ms | 134.56mJ/0% |
| GFNet-XS-distil-192/12 | 4.55M | UMC | 98.57 | 17.55ms | 52.65mJ/60.87% |
| GFNet-XS-distil-dynn/12 | 4.64M | UMC | 96.66 | 16.84ms | 43.80mJ/67.44% |
