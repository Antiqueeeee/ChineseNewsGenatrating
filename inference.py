import argparse
import os
import torch
from transformers import BertTokenizer
from model import GPT2LMHeadModel
from torch.nn import functional as F
import copy

def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    assert logits.dim() == 2
    top_k = min(top_k,logits[0].size(-1))
    if top_k > 0 :
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit,top_k)[0][...,-1,None]
            logit[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # 对排序后的结果使用softmax归一化，再获取累积概率序列
        # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 删除累积概率高于top_p的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，使第一个标记也保持在top_p之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的累积概率达到top_p的标记
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits


def predict_one_sample(model,tokenizer,args,content):

    unk_id,sep_id = tokenizer.convert_tokens_to_ids("[UNK]"),tokenizer.convert_tokens_to_ids("[SEP]")
    content_tokens = ["[Content]"] + tokenizer.tokenize(content) + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    input_ids = [copy.deepcopy(input_ids) for i in range(args.batch_size)]



    token_type = ["[Content]"] * len(content_tokens)
    token_type_ids = tokenizer.convert_tokens_to_ids(token_type)
    token_type_ids = [copy.deepcopy(token_type_ids) for i in range(args.batch_size)]

    input_tensors = torch.tensor(input_ids).long().to(args.device)
    token_type_tensors = torch.tensor(token_type_ids).long().to(args.device)

    next_token_type = torch.tensor(
        [[tokenizer.convert_tokens_to_ids("[Title]")] for i in range(args.batch_size)]
    ) .long().to(args.device)
    generated = list()
    finish_set = [0] * args.batch_size
    with torch.no_grad():
        for i in range(args.generate_max_len):
            outputs = model(input_ids = input_tensors,token_type_ids = token_type_tensors)
            next_token_logits = outputs[0][:,-1,:]
            # 对batch_size进行遍历，将词表中出现在序列中的词的概率进行惩罚
            #
            # 为啥要惩罚
            #
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty

            # 对batch_size进行遍历，将词表中的UNK的值设为无穷小
            for next_token_logit in next_token_logits:
                next_token_logit[unk_id] = -float("Inf")
            # 使用top_k_top_p_filtering函数，按照top_k和top_p的值，对预测结果进行筛选
            filter_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            # 判断如果哪个序列的预测标记为sep_id时，则加入到finish_set
            for index, token_id in enumerate(next_tokens[:, 0]):
                if token_id == sep_id:
                    # finish_set.add(index)
                    finish_set[index] = 1
            # # 判断，如果finish_set包含全部的序列序号，则停止预测；否则继续预测
            # finish_flag = True
            # for index in range(args.batch_size):
            #     if index not in finish_set:
            #         finish_flag = False
            #         break
            if sum(finish_set) == args.batch_size:
                break
            # 将预测标记添加到generated中
            generated.append([token.item() for token in next_tokens[:, 0]])
            # 将预测结果拼接到input_tensors和token_type_tensors上，继续下一次预测
            input_tensors = torch.cat((input_tensors, next_tokens), dim=-1)
            token_type_tensors = torch.cat((token_type_tensors, next_token_type), dim=-1)
        # 用于存储预测结果
        candidate_responses = []
        # 对batch_size进行遍历，并将token_id变成对应汉字
        for index in range(args.batch_size):
            responses = []
            for token_index in range(len(generated)):
                # 判断，当出现sep_id时，停止在该序列中添加token
                if generated[token_index][index] != sep_id:
                    responses.append(generated[token_index][index])
                else:
                    break
            # 将token_id序列变成汉字序列，去除"##"，并将[Space]替换成空格
            candidate_responses.append(
                "".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[space]", " ")
            )
    return candidate_responses



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--model_path', default='output_dir/checkpoint-139805', type=str, help='模型文件路径')
    # parser.add_argument('--model_path', default='output_dir/checkpoint-2', type=str, help='模型文件路径')
    parser.add_argument('--vocab_path', default='vocab/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--batch_size', default=3, type=int, help='生成标题的个数')
    parser.add_argument('--generate_max_len', default=32, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=5, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device


    print('开始对新闻生成标题，输入CTRL + Z，则退出')
    # for root,dir,files in os.walk(os.path.join(os.path.abspath("."),"output_dir")):
    #     print(root)
    #     for file in files:
    #         print(os.path.join(root,file))

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(args.device)
    # try:
    #     while True:
    print("输入的新闻正文为:")
    content = "新华社受权于18日全文播发修改后的《中华人民共和国立法法》，修改后的立法法分为“总则”“法律”“行政法规”“地方性法规、自治条例和单行条例、规章”“适用与备案审查”“附则”等6章，共计105条。\n"
    titles = predict_one_sample(model, tokenizer, args, content)
    for i, title in enumerate(titles):
        print("生成的第{}个标题为：{}".format(i + 1, title))
    # except:
    #     pass