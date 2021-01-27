"""
__project_ = 'PycharmProjects'
__file_name__ = 'joint_bert_model'
__author__ = 'duty'
__time__ = '2021/1/14 7:39 PM'
__product_name = PyCharm
"""
from torchcrf import CRF
from torch.nn import Dropout, Linear, CrossEntropyLoss, Module
from transformers import BertModel
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class IntentClassifier(Module):
    def __init__(self, input_dim, intent_class_cnt):
        super(IntentClassifier, self).__init__()
        self.dropout = Dropout(0.3)
        self.dense_layer = Linear(input_dim, intent_class_cnt)
    def forward(self, input):
        dropout_out = self.dropout(input)
        dense_out = self.dense_layer(dropout_out)
        return dense_out

class SlotClassifier(Module):
    def __init__(self, input_dim, slot_class_cnt):
        super(SlotClassifier, self).__init__()
        self.dropout = Dropout(0.3)
        self.dense_layer = Linear(input_dim, slot_class_cnt)
    def forward(self, input):
        dropout_out = self.dropout(input)
        dense_out = self.dense_layer(dropout_out)
        return dense_out

class JointBertModel(Module):
    def __init__(self, config, intent_class_cnt, slot_class_cnt, use_crf):
        super(JointBertModel, self).__init__()
        self.bert = BertModel(config).to(device)
        self.input_dim = config.hidden_size ## bert的隐层神经元数
        self.intent_class_cnt = intent_class_cnt ##意图类别数
        self.slot_class_cnt = slot_class_cnt ## 实体类别数
        self.intent_net = IntentClassifier(self.input_dim, self.intent_class_cnt).to(device)
        self.slot_net = SlotClassifier(self.input_dim, self.slot_class_cnt).to(device)
        self.crf = CRF(self.slot_class_cnt).to(device)
        self.intent_loss_func = CrossEntropyLoss()
        self.slot_loss_func = CrossEntropyLoss()
        self.use_crf = use_crf
    ## 模型的输入和bert的输入一致
    def forward(self, input_ids, attention_masks, token_type_ids, intent_label_ids, slot_label_ids):
        bert_out = self.bert(input_ids, attention_masks, token_type_ids)
        sequence_out = bert_out[0]
        pooler_out = bert_out[1]  ## bert的输出为 last_hidden_state:最后一层的输出， pooler_output:cls的输出 hidden_states:tuple of list，所有层的输出
        ## cls的输出用来做意图分类
        intent_out_logits = self.intent_net(pooler_out)
        ## sequence_out用来做实体识别
        slot_out_logits = self.slot_net(sequence_out)
        intent_loss = self.intent_loss_func(intent_out_logits.view(-1, self.intent_class_cnt), intent_label_ids.view(-1))
        if self.use_crf:
            slot_loss = self.crf(slot_out_logits, slot_label_ids, attention_masks.byte(), reduction = 'mean')
            slot_loss = -1*slot_loss
        else:
            slot_loss = self.slot_loss_func(slot_out_logits.view(-1, self.slot_class_cnt), slot_label_ids.view(-1))
        total_loss = intent_loss+slot_loss
        return total_loss, intent_out_logits, slot_out_logits



