# -*- coding: utf-8 -*- #
# ------------------------------------------------------------------
# File Name:        peft_fine_tuning_data.py
# Author:           wzw
# Version:          0.1
# Created:          2024/11/18
# Description:      lora微调技术
# ------------------------------------------------------------------
import torch, os
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from dataloader import Dataload
import json

# 需要GPU,训练时把这个注释去掉,把下面的注释
# use bf16 and FlashAttention if supported
# if torch.cuda.is_bf16_supported():
#     os.system('pip install flash_attn')
#     compute_dtype = torch.bfloat16
#     attn_implementation = 'flash_attention_2'
# else:
#     compute_dtype = torch.float16
#     attn_implementation = 'sdpa'

# 电脑没有GPU,仅为了在笔记本调试
compute_dtype = torch.float16
attn_implementation = 'sdpa'


# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['content'])):
#         text = f"text: {example['content'][i]}"
#         output_texts.append(text)
#     return output_texts
def merge_columns(example):
    example['chat'] = [
        {"role": "user", "content": example['ask']},
        {"role": "assistant", "content": example['answer']}
    ]
    return example
# def merge_columns(example):
#     example = json.loads(str(example).replace("\"", "\\'").replace(' ', '').replace("'", "\""))
#     dictory = json.loads(example["text"])
#     example['chat'] = []
#     for i in range(len(dictory['history']) // 2):
#         example['chat'].append({"role": "user", "content": dictory['history'][i * 2]})
#         example['chat'].append({"role": "assistant", "content": dictory['history'][i * 2 + 1]})
#     return example


dataload = Dataload()
# dataload.load_text("D:\\DATA\\douban_new\\douban_new\\douban_train.txt", merge_columns)
ds = dataload.load_csv(
    "D:\\DATA\\Chinese-medical-dialogue-data-master\\Chinese-medical-dialogue-data-master\\Data_数据\\Andriatria_男科\\男科5-13000.csv",
    merge_columns, remove_columns=['title'])
# 保存整个数据集到 Arrow 格式
# ds.save_to_disk("squad_dataset")
model_name = "D:\DATA\MyProjects\Qwen2.5-0.5B"
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, add_eos_token=True, use_fast=True)
ds = ds.map(
    lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'
bnb_config = BitsAndBytesConfig(
    # 以 4 位精度加载，并使用 NF4 数据类型和嵌套量化技术
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=None,
    bnb_4bit_use_double_quant=True,
)

# 需要GPU,训练时把这个注释去掉,把下面的注释
# model = AutoModelForCausalLM.from_pretrained(
#     model_name, quantization_config=bnb_config, device_map={"": 0}, attn_implementation=attn_implementation
# )

# 电脑没有GPU,仅为了在笔记本上调试
model = AutoModelForCausalLM.from_pretrained(
    model_name
)
# 配置lora
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    # 低秩矩阵维度
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

training_arguments = TrainingArguments(
    output_dir="./Qwen2.5-0.5B",
    evaluation_strategy="steps",
    do_eval=True,
    optim="paged_adamw_8bit",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=8,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    eval_steps=100,
    num_train_epochs=3,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
)
# 加载lora
trainer = SFTTrainer(
    model=model,
    train_dataset=ds['train'],
    eval_dataset=ds['test'],
    # peft配置
    peft_config=peft_config,
    dataset_text_field='formatted_chat',
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
    # formatting_func=formatting_prompts_func
)

trainer.train()
