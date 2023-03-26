import torch
from transformers import BertModel,BertTokenizer
def data_reading(text_path,label_path):
    text,label = list(), list()
    with open(text_path,encoding="utf-8") as f:
        for l in f:
            text.append(l)
    with open(label_path,encoding="utf-8") as f:
        for l in f:
            label.append(l)
    data = [{"text":k,"label":v}for k,v in zip(text,label)]
    return data

train_corpus = data_reading("./LCSTS/train.src.txt","./LCSTS/train.tgt.txt")
dev_corpus = data_reading("./LCSTS/valid.src.txt","./LCSTS/valid.tgt.txt")
test_corpus = data_reading("./LCSTS/test.src.txt","./LCSTS/test.tgt.txt")


class model(torch.nn.Module):
    def __init__(self,config):
        super(model).__init__(config)
        self.config = config
        self.bert = BertModel.from_pretrain(config.bert_name,cache_dir="./cache",output_hidden_states=True)
