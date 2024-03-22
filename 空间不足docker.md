# You don‘t have enough free space in /var/cache/apt/archives

给var/cache/apt/archives 创建一个软链接。

首先可以使用df -h 命令查看一下机器中存储使用情况

##举例
于是就可以在/dev 目录下创建软链接。命令如下:
`sudo mkdir -p "/dev/debs/partial"
sudo rm -rf /var/cache/apt/archives
sudo ln -s "/dev/debs" /var/cache/apt/archives
`
上述命令第一个时在 /dev 下创建了debs 以及其子文件夹 partial，命令加了引号，因为需要创建目录以及子目录。

第二条命令删除了 /var/cache/apt/archives 文件。

第三条命令建立了软连接，表示我们的 /dev/debs 目录指向 /var/cache/apt/archives，如果系统寻找 /var/cache/apt/archives 目录的话，最终会指向 /dev/debs。

##参考文档：
https://blog.csdn.net/t46414704152abc/article/details/116234182
