## ollama
### ollama是一个快速部署大模型的工具，利用openai的function call接口去部署自己的大模型,可以用来发邮件或者搜索等等。 
## onnx2engine
### ①--onnx 指定ONNX文件路径
### ②--tacticSources指定使用的方法库
### ③--workspace指定工作空间大小，单位是MB
### ④--fp16 开启FP16模式
### ⑤--saveEngine指定生成的engine的保存路径
### ⑥--verbose打开verbose模式，更多打印信息。
### `trtexec --onnx=path to onnx --saveEngine=end2end.engine --best --workspace=1024 --minShapes=input1:10x20x384,input2:10x20 --optShapes=input1:10x20x384,input2:10x20  --maxShapes=input1:10x20x384,input2:10x20 `
### 更多信息查阅搜素trtexec


