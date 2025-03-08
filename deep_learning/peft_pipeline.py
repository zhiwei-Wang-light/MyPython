# # 第一种方式,适用于单个模型
# from transformers import pipeline
#
# pipe = pipeline("text-generation", "D:\DATA\MyProjects\Qwen2.5-0.5B", use_fast=True)
# messages = [
#     {
#         "role": "system",
#         "content": "You are a friendly chatbot who always responds in the style of a pirate",
#     },
#     {"role": "user",
#      "content": "现在18岁了，最近半年，发觉，性生活总是提不起劲，同时，每次才开始就已经射了，请问：男孩早泄究竟是什么因素引发的。"},
# ]
# output = pipe(messages, max_new_tokens=256)
# print(output[-1]['generated_text'][-1]['content'])

# 第二种方式
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "D:\DATA\MyProjects\Qwen2.5-0.5B"
peft_model_id = "D:\DATA\MyProjects\Qwen2.5-0.5B-lora"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)

tokenizer = AutoTokenizer.from_pretrained("D:\DATA\MyProjects\Qwen2.5-0.5B", add_eos_token=True, use_fast=True)

messages = [
    {"role": "user",
     "content": "得了龟头瘙痒怎么治较好"},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2,
                         repetition_penalty=1.2,  eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0],skip_special_tokens=True))
