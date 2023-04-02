
from transformers.modeling_gpt2 import GPT2PreTrainedModel,GPT2Model
from torch import nn
from torch.nn import CrossEntropyLoss

class GPT2LMHeadModel(GPT2PreTrainedModel):
    def __init__(self,config):
        super(GPT2LMHeadModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.init_weights()
    def forward(self,input_ids=None,past=None,token_type_ids=None,labels=None,title_id=None):
        transformer_outputs = self.transformer(input_ids,past=past,token_type_ids=token_type_ids)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            if title_id is None or token_type_ids is None:
                raise Exception("当labels不为None时， title_id和token_type_ids均不可以为None。")
            mask = (token_type_ids == title_id).long()
            labels = labels * mask
            # 对预测结果和标签进行偏移操作
            # GPT2的生成机制为通过前面的token，预测下一个token；并且labels与input_ids相同，
            # 因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss
            shift_logits = lm_logits[...,:-1,:].contiguous()
            shift_labels = labels[...,1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=0,reduction="sum")
            loss = loss_fct(shift_logits.view(-1,shift_logits.size(-1)),shift_labels.view(-1))
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs
