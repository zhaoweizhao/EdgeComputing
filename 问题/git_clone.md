### error: RPC failed; curl 16 Error in the HTTP2 framing layer  
### fatal: expected flush after ref listing
### 问题:
```python
(GFNet) nvidia@ubuntu:~/heShaoWei$ git clone https://github.com/raoyongming/GFNet.git
Cloning into 'GFNet'...
error: RPC failed; curl 16 Error in the HTTP2 framing layer
fatal: expected flush after ref listing
```
### 解决办法：
通过指令(1)设置 git config 来强制 git 使用 HTTP 1.1

**(1)**` git config --global http.version HTTP/1.1`

同样，也可以通过指令(2)将 git 设置成原来的 HTTP 2

**(2)**` git config --global http.version HTTP/2`
### 实践过程：
![image](https://github.com/zhaoweizhao/EdgeComputing/assets/151530559/fd6e4a8c-3656-4bb3-88ff-6c5a09dc70cc)
