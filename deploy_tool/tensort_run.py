# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        tensort_run.py
# Author:           wzw
# Version:          0.1
# Created:          2024/12/12
# Description:      转换模型到onnx
# ------------------------------------------------------------------
import tensorrt as trt
import numpy as np
import torch
import collections
# 为什么不直接使用pycuda.autoinit？自动初始化很多时候不好使，比如多线程
import pycuda.autoinit
import pycuda.driver as cuda  # GPU CPU之间的数据传输

# cuda.init()
# cfx=cuda.Device(0).make_context()

# 初始化和加载模型
engine = "/home/light/tools/TensorRT/bin/end2end.engine"
# 创建logger：日志记录器
logger = trt.Logger(trt.Logger.INFO)
# 读取engine文件并记录log
with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
    # 将engine进行反序列化，得到tensort可以推理的对象
    model = runtime.deserialize_cuda_engine(
        f.read())


# 构建可执行的context(上下文：记录执行任务所需要的相关信息)
context = model.create_execution_context()

# output_names = []
# fp16 = False
# dynamic = False
# bindings = collections.OrderedDict()
# Binding = collections.namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
#
#
# for i in range(model.num_bindings):
#     # 获得输入输出的名字"images","output0"
#     name = model.get_binding_name(i)
#     dtype = trt.nptype(model.get_binding_dtype(i))
#     # 判断是否为输入
#     if model.binding_is_input(i):
#         if -1 in tuple(model.get_binding_shape(
#                 i)):
#             dynamic = True
#             # Set the dynamic shape of a binding
#             context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
#         if dtype == np.float16:
#             fp16 = True
#     else:
#         output_names.append(name)
#     shape = tuple(context.get_binding_shape(i))  # 记录输入输出shape
#     input_tensor = torch.from_numpy(np.empty(shape, dtype=dtype))  # 创建一个全0的与输入或输出shape相同的tensor
#     bindings[name] = Binding(name, dtype, shape, input_tensor, int(input_tensor.data_ptr()))  # 放入之前创建的对象中

# cuda.pagelocked_empty：分配锁页内存（Page-Locked Memory），这类内存对于 GPU 的数据传输非常高效。每个输入和输出的内存大小是通过 trt.volume(context.get_binding_shape(i)) 获取的。
h_input1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
h_input2 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
h_input3 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(2)), dtype=np.float32)
h_input4 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(3)), dtype=np.float32)
h_input5 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(4)), dtype=np.float32)
h_output1 = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(5)), dtype=np.float32)

# 为 GPU 分配显存，用于存储输入和输出数据。
d_input1 = cuda.mem_alloc(h_input1.nbytes)
d_input2 = cuda.mem_alloc(h_input2.nbytes)
d_input3 = cuda.mem_alloc(h_input3.nbytes)
d_input4 = cuda.mem_alloc(h_input4.nbytes)
d_input5 = cuda.mem_alloc(h_input5.nbytes)
d_output1 = cuda.mem_alloc(h_output1.nbytes)
# 创建cuda流
stream = cuda.Stream()

# 异步地将数据从 CPU（主机）内存拷贝到 GPU（设备）内存。数据传输操作通过流来管理，以便实现更高效的并行执行。
cuda.memcpy_htod_async(d_input1, h_input1, stream)
cuda.memcpy_htod_async(d_input2, h_input2, stream)
cuda.memcpy_htod_async(d_input3, h_input3, stream)
cuda.memcpy_htod_async(d_input4, h_input4, stream)
cuda.memcpy_htod_async(d_input5, h_input5, stream)
# cfx.push()
# 异步执行推理操作。通过指定绑定的 GPU 内存地址（输入和输出），以及流（stream_handle），TensorRT 会在 GPU 上执行推理并将结果写入输出缓冲区。
context.execute_async_v2(
    bindings=[int(d_input1), int(d_input2), int(d_input3), int(d_input4), int(d_input5), int(d_output1)],
    stream_handle=stream.handle)
# 异步地将推理结果从 GPU 内存复制回 CPU 内存。
cuda.memcpy_dtoh_async(h_output1, d_output1, stream)
# 等待流中的所有操作完成。确保所有异步操作都已完成，获取最新的输出结果。
stream.synchronize()
print(h_output1)
# cfx.pop()
