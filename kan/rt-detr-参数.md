```python
------------------------------------------------------------------------
      Layer (type)          Input Shape         Param #     Tr. Param #
========================================================================
          RTDETR-1     [1, 3, 640, 640]      42,747,522      42,719,010
========================================================================
Total params: 42,747,522
Trainable params: 42,719,010
Non-trainable params: 28,512
------------------------------------------------------------------------


================================================== Hierarchical Summary ==================================================

DistributedDataParallel(
  (module): RTDETR(
    (backbone): PResNet(
      (conv1): Sequential(
        (conv1_1): ConvNormLayer(
          (conv): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 864 params
          (norm): FrozenBatchNorm2d(32, eps=1e-05), 0 params
          (act): ReLU(inplace=True), 0 params
        ), 864 params
        (conv1_2): ConvNormLayer(
          (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 9,216 params
          (norm): FrozenBatchNorm2d(32, eps=1e-05), 0 params
          (act): ReLU(inplace=True), 0 params
        ), 9,216 params
        (conv1_3): ConvNormLayer(
          (conv): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 18,432 params
          (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
          (act): ReLU(inplace=True), 0 params
        ), 18,432 params
      ), 28,512 params
      (res_layers): ModuleList(
        (0): Blocks(
          (blocks): ModuleList(
            (0): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 4,096 params
                (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 4,096 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 36,864 params
                (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 36,864 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 16,384 params
              (short): ConvNormLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 16,384 params
              (act): ReLU(inplace=True), 0 params
            ), 73,728 params
            (1): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
                (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 16,384 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 36,864 params
                (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 36,864 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 16,384 params
              (act): ReLU(inplace=True), 0 params
            ), 69,632 params
            (2): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
                (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 16,384 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 36,864 params
                (norm): FrozenBatchNorm2d(64, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 36,864 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 16,384 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 16,384 params
              (act): ReLU(inplace=True), 0 params
            ), 69,632 params
          ), 212,992 params
        ), 212,992 params
        (1): Blocks(
          (blocks): ModuleList(
            (0): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 32,768 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 32,768 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 147,456 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 147,456 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 65,536 params
              (short): Sequential(
                (pool): AvgPool2d(kernel_size=2, stride=2, padding=0), 0 params
                (conv): ConvNormLayer(
                  (conv): Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
                  (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                  (act): Identity(), 0 params
                ), 131,072 params
              ), 131,072 params
              (act): ReLU(inplace=True), 0 params
            ), 376,832 params
            (1): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 65,536 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 147,456 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 147,456 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 65,536 params
              (act): ReLU(inplace=True), 0 params
            ), 278,528 params
            (2): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 65,536 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 147,456 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 147,456 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 65,536 params
              (act): ReLU(inplace=True), 0 params
            ), 278,528 params
            (3): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 65,536 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 147,456 params
                (norm): FrozenBatchNorm2d(128, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 147,456 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 65,536 params
              (act): ReLU(inplace=True), 0 params
            ), 278,528 params
          ), 1,212,416 params
        ), 1,212,416 params
        (2): Blocks(
          (blocks): ModuleList(
            (0): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 131,072 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 589,824 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 589,824 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 262,144 params
              (short): Sequential(
                (pool): AvgPool2d(kernel_size=2, stride=2, padding=0), 0 params
                (conv): ConvNormLayer(
                  (conv): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 524,288 params
                  (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                  (act): Identity(), 0 params
                ), 524,288 params
              ), 524,288 params
              (act): ReLU(inplace=True), 0 params
            ), 1,507,328 params
            (1): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 262,144 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 589,824 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 262,144 params
              (act): ReLU(inplace=True), 0 params
            ), 1,114,112 params
            (2): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 262,144 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 589,824 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 262,144 params
              (act): ReLU(inplace=True), 0 params
            ), 1,114,112 params
            (3): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 262,144 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 589,824 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 262,144 params
              (act): ReLU(inplace=True), 0 params
            ), 1,114,112 params
            (4): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 262,144 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 589,824 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 262,144 params
              (act): ReLU(inplace=True), 0 params
            ), 1,114,112 params
            (5): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 262,144 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): FrozenBatchNorm2d(256, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 589,824 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
                (norm): FrozenBatchNorm2d(1024, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 262,144 params
              (act): ReLU(inplace=True), 0 params
            ), 1,114,112 params
          ), 7,077,888 params
        ), 7,077,888 params
        (3): Blocks(
          (blocks): ModuleList(
            (0): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 524,288 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 524,288 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 2,359,296 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 2,359,296 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,048,576 params
                (norm): FrozenBatchNorm2d(2048, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 1,048,576 params
              (short): Sequential(
                (pool): AvgPool2d(kernel_size=2, stride=2, padding=0), 0 params
                (conv): ConvNormLayer(
                  (conv): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False), 2,097,152 params
                  (norm): FrozenBatchNorm2d(2048, eps=1e-05), 0 params
                  (act): Identity(), 0 params
                ), 2,097,152 params
              ), 2,097,152 params
              (act): ReLU(inplace=True), 0 params
            ), 6,029,312 params
            (1): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,048,576 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 1,048,576 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 2,359,296 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 2,359,296 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,048,576 params
                (norm): FrozenBatchNorm2d(2048, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 1,048,576 params
              (act): ReLU(inplace=True), 0 params
            ), 4,456,448 params
            (2): BottleNeck(
              (branch2a): ConvNormLayer(
                (conv): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,048,576 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 1,048,576 params
              (branch2b): ConvNormLayer(
                (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 2,359,296 params
                (norm): FrozenBatchNorm2d(512, eps=1e-05), 0 params
                (act): ReLU(inplace=True), 0 params
              ), 2,359,296 params
              (branch2c): ConvNormLayer(
                (conv): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False), 1,048,576 params
                (norm): FrozenBatchNorm2d(2048, eps=1e-05), 0 params
                (act): Identity(), 0 params
              ), 1,048,576 params
              (act): ReLU(inplace=True), 0 params
            ), 4,456,448 params
          ), 14,942,208 params
        ), 14,942,208 params
      ), 23,445,504 params
    ), 23,474,016 params
    (decoder): RTDETRTransformer(
      (input_proj): ModuleList(
        (0): Sequential(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
        ), 66,048 params
        (1): Sequential(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
        ), 66,048 params
        (2): Sequential(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
        ), 66,048 params
      ), 198,144 params
      (decoder): TransformerDecoder(
        (layers): ModuleList(
          (0): TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 263,168 params
            (dropout1): Dropout(p=0.0, inplace=False), 0 params
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (cross_attn): MSDeformableAttention(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True), 49,344 params
              (attention_weights): Linear(in_features=256, out_features=96, bias=True), 24,672 params
              (value_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
              (output_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 205,600 params
            (dropout2): Dropout(p=0.0, inplace=False), 0 params
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
            (dropout3): Dropout(p=0.0, inplace=False), 0 params
            (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
            (dropout4): Dropout(p=0.0, inplace=False), 0 params
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
          ), 995,872 params
          (1): TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 263,168 params
            (dropout1): Dropout(p=0.0, inplace=False), 0 params
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (cross_attn): MSDeformableAttention(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True), 49,344 params
              (attention_weights): Linear(in_features=256, out_features=96, bias=True), 24,672 params
              (value_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
              (output_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 205,600 params
            (dropout2): Dropout(p=0.0, inplace=False), 0 params
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
            (dropout3): Dropout(p=0.0, inplace=False), 0 params
            (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
            (dropout4): Dropout(p=0.0, inplace=False), 0 params
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
          ), 995,872 params
          (2): TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 263,168 params
            (dropout1): Dropout(p=0.0, inplace=False), 0 params
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (cross_attn): MSDeformableAttention(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True), 49,344 params
              (attention_weights): Linear(in_features=256, out_features=96, bias=True), 24,672 params
              (value_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
              (output_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 205,600 params
            (dropout2): Dropout(p=0.0, inplace=False), 0 params
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
            (dropout3): Dropout(p=0.0, inplace=False), 0 params
            (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
            (dropout4): Dropout(p=0.0, inplace=False), 0 params
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
          ), 995,872 params
          (3): TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 263,168 params
            (dropout1): Dropout(p=0.0, inplace=False), 0 params
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (cross_attn): MSDeformableAttention(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True), 49,344 params
              (attention_weights): Linear(in_features=256, out_features=96, bias=True), 24,672 params
              (value_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
              (output_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 205,600 params
            (dropout2): Dropout(p=0.0, inplace=False), 0 params
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
            (dropout3): Dropout(p=0.0, inplace=False), 0 params
            (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
            (dropout4): Dropout(p=0.0, inplace=False), 0 params
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
          ), 995,872 params
          (4): TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 263,168 params
            (dropout1): Dropout(p=0.0, inplace=False), 0 params
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (cross_attn): MSDeformableAttention(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True), 49,344 params
              (attention_weights): Linear(in_features=256, out_features=96, bias=True), 24,672 params
              (value_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
              (output_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 205,600 params
            (dropout2): Dropout(p=0.0, inplace=False), 0 params
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
            (dropout3): Dropout(p=0.0, inplace=False), 0 params
            (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
            (dropout4): Dropout(p=0.0, inplace=False), 0 params
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
          ), 995,872 params
          (5): TransformerDecoderLayer(
            (self_attn): MultiheadAttention(
              (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 263,168 params
            (dropout1): Dropout(p=0.0, inplace=False), 0 params
            (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (cross_attn): MSDeformableAttention(
              (sampling_offsets): Linear(in_features=256, out_features=192, bias=True), 49,344 params
              (attention_weights): Linear(in_features=256, out_features=96, bias=True), 24,672 params
              (value_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
              (output_proj): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            ), 205,600 params
            (dropout2): Dropout(p=0.0, inplace=False), 0 params
            (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
            (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
            (dropout3): Dropout(p=0.0, inplace=False), 0 params
            (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
            (dropout4): Dropout(p=0.0, inplace=False), 0 params
            (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
          ), 995,872 params
        ), 5,975,232 params
      ), 5,975,232 params
      (denoising_class_embed): Embedding(11, 256, padding_idx=10), 2,816 params
      (query_pos_head): MLP(
        (layers): ModuleList(
          (0): Linear(in_features=4, out_features=512, bias=True), 2,560 params
          (1): Linear(in_features=512, out_features=256, bias=True), 131,328 params
        ), 133,888 params
        (act): ReLU(inplace=True), 0 params
      ), 133,888 params
      (enc_output): Sequential(
        (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
        (1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
      ), 66,304 params
      (enc_score_head): Linear(in_features=256, out_features=10, bias=True), 2,570 params
      (enc_bbox_head): MLP(
        (layers): ModuleList(
          (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
          (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
          (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
        ), 132,612 params
        (act): ReLU(inplace=True), 0 params
      ), 132,612 params
      (dec_score_head): ModuleList(
        (0): Linear(in_features=256, out_features=10, bias=True), 2,570 params
        (1): Linear(in_features=256, out_features=10, bias=True), 2,570 params
        (2): Linear(in_features=256, out_features=10, bias=True), 2,570 params
        (3): Linear(in_features=256, out_features=10, bias=True), 2,570 params
        (4): Linear(in_features=256, out_features=10, bias=True), 2,570 params
        (5): Linear(in_features=256, out_features=10, bias=True), 2,570 params
      ), 15,420 params
      (dec_bbox_head): ModuleList(
        (0): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
          ), 132,612 params
          (act): ReLU(inplace=True), 0 params
        ), 132,612 params
        (1): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
          ), 132,612 params
          (act): ReLU(inplace=True), 0 params
        ), 132,612 params
        (2): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
          ), 132,612 params
          (act): ReLU(inplace=True), 0 params
        ), 132,612 params
        (3): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
          ), 132,612 params
          (act): ReLU(inplace=True), 0 params
        ), 132,612 params
        (4): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
          ), 132,612 params
          (act): ReLU(inplace=True), 0 params
        ), 132,612 params
        (5): MLP(
          (layers): ModuleList(
            (0): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (1): Linear(in_features=256, out_features=256, bias=True), 65,792 params
            (2): Linear(in_features=256, out_features=4, bias=True), 1,028 params
          ), 132,612 params
          (act): ReLU(inplace=True), 0 params
        ), 132,612 params
      ), 795,672 params
    ), 7,322,658 params
    (encoder): HybridEncoder(
      (input_proj): ModuleList(
        (0): Sequential(
          (0): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
          (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
        ), 131,584 params
        (1): Sequential(
          (0): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 262,144 params
          (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
        ), 262,656 params
        (2): Sequential(
          (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 524,288 params
          (1): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
        ), 524,800 params
      ), 919,040 params
      (encoder): ModuleList(
        (0): TransformerEncoder(
          (layers): ModuleList(
            (0): TransformerEncoderLayer(
              (self_attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True), 65,792 params
              ), 263,168 params
              (linear1): Linear(in_features=256, out_features=1024, bias=True), 263,168 params
              (dropout): Dropout(p=0.0, inplace=False), 0 params
              (linear2): Linear(in_features=1024, out_features=256, bias=True), 262,400 params
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True), 512 params
              (dropout1): Dropout(p=0.0, inplace=False), 0 params
              (dropout2): Dropout(p=0.0, inplace=False), 0 params
              (activation): GELU(approximate='none'), 0 params
            ), 789,760 params
          ), 789,760 params
        ), 789,760 params
      ), 789,760 params
      (lateral_convs): ModuleList(
        (0): ConvNormLayer(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
          (act): SiLU(inplace=True), 0 params
        ), 66,048 params
        (1): ConvNormLayer(
          (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
          (act): SiLU(inplace=True), 0 params
        ), 66,048 params
      ), 132,096 params
      (fpn_blocks): ModuleList(
        (0): CSPRepLayer(
          (conv1): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (conv2): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (bottlenecks): Sequential(
            (0): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (1): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (2): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
          ), 1,969,152 params
          (conv3): Identity(), 0 params
        ), 2,232,320 params
        (1): CSPRepLayer(
          (conv1): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (conv2): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (bottlenecks): Sequential(
            (0): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (1): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (2): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
          ), 1,969,152 params
          (conv3): Identity(), 0 params
        ), 2,232,320 params
      ), 4,464,640 params
      (downsample_convs): ModuleList(
        (0): ConvNormLayer(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 589,824 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
          (act): SiLU(inplace=True), 0 params
        ), 590,336 params
        (1): ConvNormLayer(
          (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 589,824 params
          (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
          (act): SiLU(inplace=True), 0 params
        ), 590,336 params
      ), 1,180,672 params
      (pan_blocks): ModuleList(
        (0): CSPRepLayer(
          (conv1): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (conv2): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (bottlenecks): Sequential(
            (0): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (1): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (2): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
          ), 1,969,152 params
          (conv3): Identity(), 0 params
        ), 2,232,320 params
        (1): CSPRepLayer(
          (conv1): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (conv2): ConvNormLayer(
            (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 131,072 params
            (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
            (act): SiLU(inplace=True), 0 params
          ), 131,584 params
          (bottlenecks): Sequential(
            (0): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (1): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
            (2): RepVggBlock(
              (conv1): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 589,824 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 590,336 params
              (conv2): ConvNormLayer(
                (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False), 65,536 params
                (norm): SyncBatchNorm(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 512 params
                (act): Identity(), 0 params
              ), 66,048 params
              (act): SiLU(inplace=True), 0 params
            ), 656,384 params
          ), 1,969,152 params
          (conv3): Identity(), 0 params
        ), 2,232,320 params
      ), 4,464,640 params
    ), 11,950,848 params
  ), 42,747,522 params
), 42,747,522 params


==========================================================================================================================
```
