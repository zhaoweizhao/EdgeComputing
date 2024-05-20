## Nvidia Jetson Agx Xavier 刷机step3 internet connection 报错
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
