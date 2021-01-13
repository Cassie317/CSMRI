# Github建立仓库并连接

## 1. 注册github账号，安装git

本步骤跳过

## 2. Github上新建一个仓库

## 3.本地项目文件

> 打开本地项目的文件夹，除了代码等必要文件外，一个良好的习惯是添加下面几个文件。**README.md：**项目的说明文档。**LICENSE：**许可。从随便一个别人的库里下载，将 Copyright 行修改为自己的时间和名字即可。**.gitignore：**指明无需上传的文件和子文件夹。
>
> ![图片](https://img-blog.csdn.net/201808060002050)
>
> 关于 .gitignore：首先新建一个 .txt 文档，在其中写上需要忽略的文件和文件夹，如下图所示。
>
> ![gitignore](https://img-blog.csdn.net/2018080523511390)
>
> [参考](https://blog.csdn.net/Xingyb14/article/details/81437651?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.control)

## SSH配置

---

因为我是想用服务器和github连接，服务器上本身又有公钥了，所以第一次尝试直接用那已有的公钥放到github中

> ssh -T git@github.com 

卡住报错没反应

后来尝试

> ssh -v git@github.com:Cassie317/CSMRI.git

![image-20210113164144468](C:\Users\70902\AppData\Roaming\Typora\typora-user-images\image-20210113164144468.png)

这个应该是因为没有在ssh的config文件中配置的原因，ok，接下来去配置上

> Host github.com
> 	HostName  github.com
> 	User email@qq.com     (换成自己的github邮箱)
> 	PreferredAuthentications publickey  （公钥连接）
> 	IdentityFile  /.ssh/id_github_rsa （公钥位置）

配完之后居然还是不行！心态崩了，什么个鬼

> git clone https://github.com/Cassie317/CSMRI.git

报错

fatal: unable to access 'https://github.com/Cassie317/CSMRI.git/': Failed to connect to github.com port 443: Connection timed out

然后上网一顿狂搜，说是让弄代理的最多，一顿操作，也没啥用

后来觉得还是SSH公钥有问题吧可能，然后看了另外一个老哥的博客，在电脑上配置了个新的SSH公钥。

1. 

> 同时配置两个公钥的时候，主要是注意不要覆盖掉原来的公钥，新取一个名字。

``` 
ssh-keygen -t rsa -C "email自己的邮箱"
Enter file in which to save the key：这一行注意改名字不要用默认的
```

2. 生成好后，将config里的IdentityFile路径更换

3. 将新生成的id_rsa.pub内容复制到github里面
4. 因为我的服务器需要执行一个文件才能联网，恨死自己了，现在才想起来

