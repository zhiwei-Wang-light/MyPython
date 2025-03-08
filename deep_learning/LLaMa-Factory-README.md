### 1.[数据格式选择](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html#id14)

#### 如果您希望使用自定义数据集，请务必在dataset_info.json件中添加对数据集及其内容的定义。目前我们支持 Alpaca 格式和 ShareGPT 格式的数据集。

### 2.[sft微调](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/sft.html)

#### 配置yaml文件,主要是数据集名称需要与上一步的数据集名称相同,模板选择正确

### 3.[LoRA合并](https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/merge_lora.html)

#### 训练完成之后,使用yaml配置文件对模型进行量化合并,注意：不要对量化过的模型进行量化

### 4.[colab示例](https://colab.research.google.com/drive/1d5KQtbemerlSDSxZIfAaWXhKr30QypiK?usp=sharing#scrollTo=yQDp0sXX3qE4)

#### 使用这个示例可以实现模型的微调与推理