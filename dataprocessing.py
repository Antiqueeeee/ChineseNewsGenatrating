import json

def data_reading(text_path,label_path):
    text,label = list(), list()
    with open(text_path,encoding="utf-8") as f:
        for l in f:
            text.append(l)
    with open(label_path,encoding="utf-8") as f:
        for l in f:
            label.append(l)
    data = [{"content":k,"title":v} for k,v in zip(text,label)]
    return data


train_corpus = data_reading("./data/train.src.txt","./data/train.tgt.txt")
dev_corpus = data_reading("./data/valid.src.txt","./data/valid.tgt.txt")
test_corpus = data_reading("./data/test.src.txt","./data/test.tgt.txt")

with open("./data/train_data.json","w",encoding="utf-8") as f:
    json.dump(train_corpus,f,indent=2,ensure_ascii=False)
with open("./data/dev_data.json","w",encoding="utf-8") as f:
    json.dump(dev_corpus,f,indent=2,ensure_ascii=False)
with open("./data/test_data.json","w",encoding="utf-8") as f:
    json.dump(test_corpus,f,indent=2,ensure_ascii=False)