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


