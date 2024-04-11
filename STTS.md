# STTS
### Detectron2安装

`git clone https://github.com/facebookresearch/detectron2.git `
`python -m pip install -e detectron2`

### import cv2时，动态链接报错报错
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/976dd7d5-6a9e-4035-8840-0aaa6d65f2c9)

1、首先卸载卸载干净所有的opencv包

`pip list | grep opencv`

针对每个列出的 OpenCV 包，运行以下命令来卸载它们  

`pip uninstall <package_name>`

例如，`pip uninstall opencv-python`

2、安装完整包*opencv-contrib-python-headless<4.3*

`pip install "opencv-contrib-python-headless<4.3" -i https://pypi.tuna.tsinghua.edu.cn/simple
`

### python 导入其他目录下的模块
**通用：从任意文件夹路径下导入模块**
<img width="770" alt="image" src="https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/950ac3e4-431e-4cbd-bccc-0d30d8b591f5">

参考博客：https://blog.csdn.net/Strive_For_Future/article/details/106716745

### NOTICE: Containers losing access to GPUs with error: "Failed to initialize NVML: Unknown Error"
Containerized GPU workloads may suddenly lose access to their GPUs. This situation occurs when systemd is used to manage the cgroups of the container and it is triggered to reload any Unit files that have references to NVIDIA GPUs (e.g. with something as simple as a `systemctl daemon-reload`).

When the container loses access to the GPU, you will see the following error message from the console output:

`Failed to initialize NVML: Unknown Error`

**The container needs to be deleted once the issue occurs.**

When it is restarted (manually or automatically depending on the use of a container orchestration platform), it will regain access to the GPU.

### xx object has no attribute ‘module‘,do you mean 'modules'

原因：**多卡训练转换到单卡训练**

解决办法：将类似于`torch.module.statedict()`语句中的module删掉，变成`torch.statedict()`

### Kinetics-400数据集
百度网盘下载（现已存入我个人网盘中）
1.下载label和raw-part中的文件，label里面存储的是标记信息，raw-part里面是视频分块的文件;

2.拼接压缩包：

```javascript
 cat compress.tar.gz.* > compress.tar.gz
 tar zxvf compress.tar.gz
```
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/96ff1042-1638-4910-a026-84e25f45beb8)

3.下载整理过的record文件，再将该文件稍作处理，转换成.csv文件

|类别 | 数据条数  | list文件 |
| :------: | :----------: | :----: |
|训练集 | 234619  |  [train.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/train.list)|
|验证集 | 19761 |  [val.list](https://videotag.bj.bcebos.com/PaddleVideo/Data/Kinetic400/val.list)|

4. .csv文件的格式应该是如下格式：
   
```
path_to_video_1 label_1
path_to_video_2 label_2
path_to_video_3 label_3
...
path_to_video_N label_N
```

