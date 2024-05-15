# coding: UTF-8
import torch
import numpy as np
from importlib import import_module
from transformers import BertModel, BertTokenizer
import tensorflow as tf

from utils import build_iterator

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号

# 加载数据集，这里用示例数据代替
text = "村广场最近有一些文体活动，会聚集很多人，没人维持秩序很容易造成一些危险，希望村里可以安排一些人员来维持秩序"


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'bert'
        self.class_list = ['other', 'consult', 'recommendation', 'feedback', 'report']  # 类别名单
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 3  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = '../bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


config = Config()
x = import_module('models.bert')
# train
model = x.Model(config).to(config.device)
# 加载已训练的ckpt模型文件
model.load_state_dict(torch.load('../THUCNews/saved_dict/bert.ckpt'))
model.eval()

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True  # 保证每次结果一样

token = config.tokenizer.tokenize(text)
# 'input_ids', 'token_type_ids' 和 'attention_mask'是模型的输入张量
token = [CLS] + token
seq_len = len(token)
mask = []
token_ids = config.tokenizer.convert_tokens_to_ids(token)
if config.pad_size:
    if len(token) < config.pad_size:
        mask = [1] * len(token_ids) + [0] * (config.pad_size - len(token))
        token_ids += ([0] * (config.pad_size - len(token)))
    else:
        mask = [1] * config.pad_size
        token_ids = token_ids[:config.pad_size]
        seq_len = config.pad_size
token_ids = torch.tensor(token_ids).unsqueeze(0)
mask = torch.tensor(mask).unsqueeze(0)
datas = (token_ids, seq_len, mask)


# 使用模型对张量进行推断
with torch.no_grad():
    outputs = model(datas)
    print(outputs)
# 获取预测结果
logits = outputs[0]
print(logits)
predicted_class = torch.argmax(logits)

# 输出分类结果
print("预测结果:", predicted_class.item())


