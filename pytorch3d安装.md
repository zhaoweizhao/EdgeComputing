# pytorch3d安装注意事项：ImportError: libcudart.so.10.1: cannot open shared object file: No such file or dire
pytorch3d表面安装十分简单，直接pip install pytorch3d就会显示安装成功。但实际运行就会出现种种问题,比如这个报错：

ImportError: libcudart.so.10.1: cannot open shared object file: No such file or directory

造成这种报错的原因，是因为在使用pip安装的时候，会下载编译好的pytorch3d进行安装，而这些事先编译好的pytorch3d安装包，在编译的时候，它的环境和你现在的环境并不一样，从而引发此种bug。
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/e0b18752-c692-4d1a-82b3-e24a16e12b0b)

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/776a9684-781e-4252-a306-227cf551197a)


这里使用第二种方法，git没法用，就复制网址到浏览器直接去下，然后解压重命名为pytorch3d，再运行

`cd pytorch3d && pip install -e .`
## 参考文档
https://blog.csdn.net/YnullW/article/details/126961883
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
