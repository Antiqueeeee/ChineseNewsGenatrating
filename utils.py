import json
import logging
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader,Dataset
import os


logger = logging.getLogger(__name__)

class GPT2NewsTitleDataSet(Dataset):
    def __init__(self,tokenizer,max_len,title_max_len,data_dir,data_set_name,path_file=None,is_overwrite=False):
        self.tokenizer = tokenizer
        self.content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        self.title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        self.space_id = self.tokenizer.convert_tokens_to_ids("[Space]")
        self.max_len = max_len
        self.title_max_len = title_max_len
        cached_feature_file = os.path.join(data_dir,f"cached_{data_set_name}_{max_len}")
        if os.path.exists(cached_feature_file) and not is_overwrite:
            logger.info(f"已经存在缓存文件{cached_feature_file}，直接加载")
            self.data_set = torch.load(cached_feature_file["data_set"])
        else:
            logger.info(f"不存在缓存文件{cached_feature_file}，进行数据预处理操作")
            self.data_set = self.load_data(path_file)
            logger.info(f"数据预处理操作完成，将处理后的数据存到{cached_feature_file}中，作为缓存文件")
            torch.save({"data_set":self.data_set},cached_feature_file)
    def load_data(self,path_file):
        self.data_set = list()
        with open(path_file,"r",encoding="utf-8") as f:
            data = json.load(f)
            for idx,sample in enumerate(tqdm(data,desc="iter",disable=False)):
                input_ids,token_type_ids = self.convert_feature(sample)
                self.data_set.append({"input_ids":input_ids,"token_type_ids":token_type_ids})
        return self.data_set
    def convert_feature(self,sample):
        title_tokens = self.tokenizer.tokenize(sample["title"].replace(" ", "[Space]"))[:self.title_max_len]
        content_tokens = self.tokenizer.tokenize(sample["content"])[:self.max_len - len(title_tokens) - 3]
        input_ids = [self.tokenizer.cls_token_id] + \
                    self.tokenizer.convert_tokens_to_ids(content_tokens) + \
                    [self.tokenizer.sep_token_id] + \
                    self.tokenizer.convert_tokens_to_ids(title_tokens) + \
                    [self.tokenizer.sep_token_id]
        token_type_ids = [self.content_id] + \
                         [self.content_id] * len(content_tokens) + \
                         [self.content_id] + \
                         [self.title_id] * len(title_tokens) +\
                         [self.title_id]
        # 判断input_ids与token_type_ids长度是否一致
        assert len(input_ids) == len(token_type_ids)
        # 判断input_ids长度是否小于等于最大长度
        assert len(input_ids) <= self.max_len
        return input_ids,token_type_ids
    def __len__(self):
        return len(self.data_set)
    def __getitem__(self, item):
        return self.data_set[item]


