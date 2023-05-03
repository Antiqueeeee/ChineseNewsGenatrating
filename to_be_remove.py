from model import GPT2LMHeadModel
from transformers import BertTokenizer
from utils import GPT2NewsTitleDataSet
from transformers.modeling_gpt2 import GPT2Config
import torch
config_path = './config/config.json'
model_config = GPT2Config.from_json_file(config_path)
vocab_path = './vocab/vocab.txt'
model = GPT2LMHeadModel(config=model_config)
tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=True)
original_vocab_size = len(tokenizer)
tokenizer.add_tokens("[Spa]", special_tokens=True)
model.resize_token_embeddings(len(tokenizer))


# 看看cls是啥样
print(model.transformer.wte.weight[tokenizer.cls_token_id][:100])
print(f'cls：\n{model(torch.tensor(tokenizer.convert_tokens_to_ids(["[CLS]"])))[0][:100]}')


#全0初始化
with torch.no_grad():
    model.transformer.wte.weight[original_vocab_size:,:] = torch.zeros([len(tokenizer) - original_vocab_size,model.config.hidden_size],requires_grad=True)
print(model.transformer.wte.weight[original_vocab_size:,:100])
print(f'看看全0效果：\n{model(torch.tensor(tokenizer.convert_tokens_to_ids(["[Spa]"])))[0][:100]}')


#已有token初始化
token_embedding = model.transformer.wte.weight[tokenizer.cls_token_id]
with torch.no_grad():
    for i in range(len(tokenizer) - original_vocab_size,0,-1):
        model.transformer.wte.weight[-i,:] = token_embedding.clone().detach().requires_grad_(True)
print(model.transformer.wte.weight[original_vocab_size:,:100])
print(f'看看cls效果：\n{model(torch.tensor(tokenizer.convert_tokens_to_ids(["[Spa]"])))[0][:100]}')




