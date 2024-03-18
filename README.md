# EdgeComputing  
## 使用docker镜像快速构建TVM
### 下载dockerimage  
TVM官方建议docker 镜像位置在https://hub.docker.com/r/tlcpack/

在这里我们选择ci-gpu
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/621685bb-efa4-4b56-8c33-585349531961)

我们利用以下命令拉取我们所需要的镜像
```
docker pull tlcpack/ci-gpu:20240105-165030-51bdaec6
```
下载时间比较长,请耐心等待

创建并运行container
```
docker run --gpus all -tid --name=vit_tvm --ipc=host --network=host -v /datadisk/HeShaoWei/vit_tvm:/workspace/vit tlcpack/ci-gpu:20240105-165030-51bdaec6
docker exec -it vit_tvm bash
```

### 下载TVM源码
```
git clone --recursive https://github.com/apache/tvm tvm
```

现在tvm目录里建一个build目录，把cmake目录下的config文件拷到build目录下

```
mkdir build
cp cmake/config.cmake build
```
**make**
```
cd build
cmake ..
make -j4
```
**设置python环境**
```
# add TVM path
export TVM_HOME=/root/workspace/tvm/
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
```
**测试python环境**
```
root@KokomiDePC:/workspace/vit/tvm/build# python
Python 3.8.18 (default, Aug 25 2023, 13:20:30)
[GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tvm
>>> print(tvm.__version__)
0.16.dev0
>>> exit()
```
###参考博客链接
https://blog.csdn.net/sexyluna/article/details/135438181
https://www.cnblogs.com/dawningblue/p/16874661.html
