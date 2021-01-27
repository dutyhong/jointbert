"""
__project_ = 'PycharmProjects'
__file_name__ = 'joint_bert_train'
__author__ = 'duty'
__time__ = '2021/1/18 11:08 AM'
__product_name = PyCharm
"""
from torch.utils.data import DataLoader
from transformers import BertConfig

from self_joint_bert.data_process import DataReader, JointBertDataset
from self_joint_bert.joint_bert_model import JointBertModel
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def joint_bert_train(model, data_iter, iter_epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    for i in range(iter_epochs):
        total_cnt = 0
        acc_cnt = 0
        print("第%s个epoch结果如下：："%(i+1))
        for batch_tokens, batch_attention_masks, batch_token_type_ids, batch_sent_intent, batch_sent_slots in data_iter:
            total_loss, intent_out_logits, slot_out_logits = model(batch_tokens.to(device), batch_attention_masks.to(device), batch_token_type_ids.to(device),
                                                                   batch_sent_intent.to(device), batch_sent_slots.to(device))
            print(total_loss.item())
            total_loss.backward()
            optimizer.step()
            total_cnt = total_cnt+len(batch_sent_intent)
            for i in range(len(batch_sent_intent)):
                if(torch.argmax(intent_out_logits[i])==batch_sent_intent[i]):
                    acc_cnt = acc_cnt +1
        print("epoch's  accuracy is %s"%(float(acc_cnt)/total_cnt))
    torch.save(model, "joint_bert_model")
    torch.save(model.state_dict(), "joint_bert_model.params")

def joint_bert_evaluate(model, test_data):
    model.predict()





if __name__=="__main__":
    data_reader = DataReader()
    intent_class_cnt = data_reader.intent_class_cnt
    slot_class_cnt = data_reader.slot_class_cnt
    dataset = JointBertDataset(data_reader=data_reader, max_seq_len=20)
    data_loader_iter = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    config = BertConfig()
    use_crf = True
    joint_bert_model = JointBertModel(config, intent_class_cnt, slot_class_cnt, use_crf)
    tokens_list = []
    intent_list = []
    slot_list = []
    epochs = 10
    joint_bert_train(joint_bert_model, data_loader_iter, epochs)
    # for batch_sent_tokens, batch_intents, batch_slots in data_loader_iter:
    #     tokens_list.append(batch_sent_tokens)
    #     intent_list.append(batch_intents)
    #     slot_list.append(batch_slots)