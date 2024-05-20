## Nvidia Jetson Agx Orin 刷机step3 internet connection 报错
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/6933305e-83b6-4fbc-a4fd-f05b142c67bf)

### 错误信息：
Internet connection: Internet connection check failure ping -c 3 nvidia.com
command < ssh -F /dev/null -o PreferredAuthentications-password -o PubkeyAuthentication=no -oUserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no -o ConnectTimeout-5 nvidia@192.168.55.1 "ping -c 3 nvidia.com > terminated with error.

### 解决方法：
`ping -c 3 www.nvidia.com` 得到nvidia.com的ip地址

更改Xavier的hosts文件（需要root权限）

`sudo su`

`vim /etc/hosts`

将下列内容写入并保存

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/ae1d7584-29bd-464a-a777-1ae08e750ba0)

## Nvidia Jetson Agx Orin 刷机step2 下载出错（cuda Runtime组件下载失败）

### 解决办法：把网络换成手机热点（个人PC和待刷机的Jetson Agx Orin要在同一局域网下）

##  Nvidia Jetson Agx Orin 刷机完成后，安装jtop

```
sudo apt-get update
sudo apt-get install python-pip
sudo apt-get install python3-pip
sudo pip3 install jetson-stats
sudo jtop   # 启动jtop
```

##  Nvidia Jetson Agx Orin 刷机完成后，初次安装Anaconda的注意事项
1、寻找aarch64的shell安装包，下载地址[anaconda清华镜像源](https://repo.anaconda.com/archive/)，我选择的是`Anaconda3-2023.09-0-Linux-aarch64.sh`，由于Jetson下载过慢，可以先在个人PC下载下来，再通过`scp`指令传回Jetson中。

**注：从此处开始，用Jetson代指Nvidia Jetson Agx Orin边缘设备**

2、进入到下载文件夹，将下载改成可执行文件,再运行文件：

```
chmod +x Anaconda3-2021.11-Linux-aarch64.sh
./Anaconda3-2021.11-Linux-aarch64.sh
```

3、如何找不到conda命令，可尝试下面的指令：

```
# 将anaconda的bin目录加入PATH，根据版本不同，也可能是~/anaconda3/bin
echo 'export PATH="~/anaconda3/bin:$PATH"' >> ~/.bashrc
# 更新bashrc以立即生效
source ~/.bashrc
```

4、初始化conda：

`conda init`

## 个人PC通过ssh连接Jetson，ssh连接出现问题
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

@    WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!     @

@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/ec122718-74c4-4289-b555-a3753eb7726a)

### 原因：
此报错是由于远程的主机的公钥发生了变化导致的。 
ssh服务是通过公钥和私钥来进行连接的，它会把每个曾经访问过计算机或服务器的公钥（public key），记录在~/.ssh/known_hosts 中，当下次访问曾经访问过的计算机或服务器时，ssh就会核对公钥，如果和上次记录的不同，OpenSSH会发出警告。

### 解决：
使用命令清除所连接的IP 

`ssh-keygen -R XX.XX.XX.XX `

这里的'XX.XX.XX.XX'就是所要连接设备的ip地址

## Jetson重装Anaconda的注意事项：
1、要删除anaconda3文件夹

![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/b5590c5a-2015-46a4-a4c1-989e7c2e330c)

2、删除Anaconda的环境变量

`sudo vim ~/.bashrc`

删掉下面圈住的部分
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/638c190d-56c4-4e4c-9a01-538fa973de6c)

3、终端输入`source ~/.bashrc`使环境变量更改生效。

### Jetson重装Anaconda时，报错
```
>>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "conda/exceptions.py", line 1118, in __call__
      File "conda/cli/main.py", line 61, in main_subshell
      File "conda/base/context.py", line 399, in __init__
      File "conda/common/configuration.py", line 1291, in __init__
      File "conda/common/configuration.py", line 1297, in _set_search_path
      File "conda/common/configuration.py", line 488, in load_file_configs
      File "conda/common/configuration.py", line 488, in <genexpr>
      File "conda/common/configuration.py", line 461, in _file_loader
      File "conda/common/configuration.py", line 387, in make_raw_parameters_from_file
      File "conda/common/serialize.py", line 57, in yaml_round_trip_load
      File "ruamel/yaml/main.py", line 434, in load
      File "ruamel/yaml/constructor.py", line 121, in get_single_data
      File "ruamel/yaml/constructor.py", line 131, in construct_document
      File "ruamel/yaml/constructor.py", line 1569, in construct_yaml_map
      File "ruamel/yaml/constructor.py", line 1453, in construct_mapping
      File "ruamel/yaml/constructor.py", line 294, in check_mapping_key
    ruamel.yaml.constructor.DuplicateKeyError: while constructing a mapping
      in "/root/.condarc", line 1, column 1
    found duplicate key "show_channel_urls" with value "True" (original value: "True")
      in "/root/.condarc", line 4, column 1
    
    To suppress this check see:
        http://yaml.readthedocs.io/en/latest/api.html#duplicate-keys
    

`$ /root/miniconda3/conda.exe install --offline --file /root/miniconda3/pkgs/env.txt -yp /root/miniconda3`

  environment variables:
                 CIO_TEST=<not set>
           CONDA_CHANNELS=https://repo.anaconda.com/pkgs/main,https://repo.anaconda.com/pkgs/r
                CONDA_EXE=/root/miniconda3/bin/conda
CONDA_EXTRA_SAFETY_CHECKS=no
          CONDA_PKGS_DIRS=/root/miniconda3/pkgs
         CONDA_PYTHON_EXE=/root/miniconda3/bin/python
               CONDA_ROOT=/root/miniconda3/install_tmp/_MEIbiVABH
      CONDA_SAFETY_CHECKS=disabled
              CONDA_SHLVL=0
           CURL_CA_BUNDLE=<not set>
          LD_LIBRARY_PATH=/root/miniconda3/install_tmp/_MEIbiVABH
               LD_PRELOAD=<not set>
      OLD_LD_LIBRARY_PATH=
                     PATH=/root/miniconda3/condabin:/usr/anaconda3/bin:/usr/local/sbin:/usr/loca
                          l/bin:/usr/sbin:/usr/bin:/root/bin
       REQUESTS_CA_BUNDLE=<not set>
            SSL_CERT_FILE=<not set>

     active environment : None
            shell level : 0
       user config file : /root/.condarc
 populated config files : 
          conda version : 22.11.1
    conda-build version : not installed
         python version : 3.9.15.final.0
       virtual packages : __archspec=1=x86_64
                          __glibc=2.17=0
                          __linux=3.10.0=0
                          __unix=0=0
       base environment : /root/miniconda3/install_tmp/_MEIbiVABH  (read only)
      conda av data dir : /root/miniconda3/install_tmp/_MEIbiVABH/etc/conda
  conda av metadata url : None
           channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /root/miniconda3/install_tmp/_MEIbiVABH/pkgs
                          /root/.conda/pkgs
       envs directories : /root/.conda/envs
                          /root/miniconda3/install_tmp/_MEIbiVABH/envs
               platform : linux-64
             user-agent : conda/22.11.1 requests/2.28.1 CPython/3.9.15 Linux/3.10.0-1160.45.1.el7.x86_64 centos/7.6.1810 glibc/2.17
                UID:GID : 0:0
             netrc file : None
           offline mode : False


An unexpected error has occurred. Conda has prepared the above report.

If submitted, this report will be used by core maintainers to improve
future releases of conda.
Would you like conda to send this report to the core maintainers? [y/N]:  
Timeout reached. No report sent.
```

**报错原因**：
`.condarc`的内容在换源的时候，多加了一行的`show_channel_urls: true`,删掉即可。

1、报错的`.condarc`文件内容：

```
show_channel_urls: true
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

2、正确的`.condarc`文件内容：

```
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch-lts: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```
