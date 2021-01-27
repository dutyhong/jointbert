"""
__project_ = 'PycharmProjects'
__file_name__ = 'data_process'
__author__ = 'duty'
__time__ = '2021/1/15 10:10 AM'
__product_name = PyCharm
"""
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class DataReader(object):
    def __init__(self):
        self.in_text_file = open("/home/tz/pythoncode/myjointbert/data/atis/train/seq.in")
        self.slot_text_file = open("/home/tz/pythoncode/myjointbert/data/atis/train/seq.out")
        self.intent_text_file = open("/home/tz/pythoncode/myjointbert/data/atis/train/label")
        self.sentences = []
        for line in self.in_text_file.readlines():
            words = line.rstrip().split(" ")
            self.sentences.append(words)
        self.intents = []
        self.intent_set= set()
        self.slots_set = set()
        for line in self.intent_text_file.readlines():
            intent = line.rstrip()
            self.intents.append(intent)
            self.intent_set.add(intent)
        self.sent_slots = []
        for line in self.slot_text_file.readlines():
            slots = line.rstrip().split(" ")
            self.sent_slots.append(slots)
            for slot in slots:
                self.slots_set.add(slot)
        self.intent2idx = {}
        self.slot2idx = {}
        self.idx2intent = {}
        self.idx2slot = {}
        for i, intent in enumerate(self.intent_set):
            self.intent2idx[intent] = i
            self.idx2intent[i] = intent
        for i, slot in enumerate(self.slots_set):
            self.slot2idx[slot] = i
            self.idx2slot[i] = slot
        self.slot2idx["UNK"] = len(self.slot2idx)
        self.idx2slot[len(self.slot2idx)] = "UNK"
        # self.slot2idx["PAD"] = 0
        # self.idx2slot[0] = "PAD"
        self.intent_class_cnt = len(self.intent_set)
        self.slot_class_cnt = len(self.slots_set)+1


class JointBertDataset(Dataset):
    def __init__(self, data_reader:DataReader, max_seq_len:int):
        self.tokenizer = BertTokenizer.from_pretrained("/home/tz/publicmodel/bert-base-uncased")
        self.data_reader = data_reader
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        sentence = self.data_reader.sentences[index]
        # bert_tokens = self.tokenizer.encode(sentence)
        sent_slots = self.data_reader.sent_slots[index]
        # sent_slot_ids = [self.data_reader.slot2idx[ss] for ss in sent_slots]

        intent = self.data_reader.intents[index]
        sent_intent_id = self.data_reader.intent2idx[intent]
        ## 做padding操作， 加入attention_masks, token_type_ids
        attention_masks = self.max_seq_len*[0]
        token_type_ids = self.max_seq_len*[0]
        if len(sentence)>self.max_seq_len-2:
            sentence = [sentence[i] for i in range(self.max_seq_len-2)]
            sent_slots = [sent_slots[i] for i in range(self.max_seq_len-2)]
            # sentence = ["CLS"]+sentence
            # sentence = sentence+["SEP"]
            sent_slots = ["UNK"]+sent_slots
            sent_slots = sent_slots+["UNK"]
            for i in range(self.max_seq_len):
                attention_masks[i] = 1
        else:
            diff_len = self.max_seq_len-2 - len(sentence)
            sentence = sentence+diff_len*["[PAD]"] ## 加PAD
            sent_slots = sent_slots + diff_len*["UNK"]
            # sentence = ["CLS"] + sentence
            # sentence = sentence + ["SEP"]
            sent_slots = ["UNK"] + sent_slots
            sent_slots = sent_slots + ["UNK"]
            for i in range(self.max_seq_len):
                attention_masks[i] = 1
        bert_tokens = self.tokenizer.encode(sentence)
        sent_slot_ids = []
        for sent_slot in sent_slots:
            sent_slot_ids.append(self.data_reader.slot2idx[sent_slot])
        return torch.tensor(bert_tokens), torch.tensor(attention_masks), torch.tensor(token_type_ids), torch.tensor(sent_intent_id), torch.tensor(sent_slot_ids)

    def __len__(self):
        return len(self.data_reader.sentences)


    # def test_tokenizer(self, input_sent):
    #     return self.tokenizer(input_sent)

if __name__=="__main__":
    ## 100，102，0， 101，103 UNK, SEP, PAD, CLS, MASK
    tokenizer = BertTokenizer.from_pretrained("/Users/duty/downloads/bert-base-uncased")
    res = tokenizer.encode("show me the flights from dallas to baltimore in first class")
    sents = ["show", "me", "the", "fights", "from"]
    res1 = tokenizer.encode(sents)
    date_reader = DataReader()
    dataset = JointBertDataset(data_reader=date_reader, max_seq_len=20)
    loader_iter = DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    tokens_list = []
    intent_list = []
    slot_list = []
    for batch_sent_tokens, batch_intents, batch_slots in loader_iter:
        tokens_list.append(batch_sent_tokens)
        intent_list.append(batch_intents)
        slot_list.append(batch_slots)

    print("ddd")
