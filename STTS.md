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
