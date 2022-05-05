import sys
import os
from pathlib import Path

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT(nn.Module):

    def __init__(self,bert_model,max_length,device,n_classes,e_tags = []):
        super(BERT, self).__init__()
        self.device = device
        self.max_length = max_length
    
        
        self.tokenizer = bert_model[0].tokenizer
        self.sentbert = bert_model[0].auto_model
        self.additional_index = len(self.tokenizer)
        # add special tokens
        if len(e_tags)!=0:
            print("Add {num} special tokens".format(num=len(e_tags)))
            special_tokens_dict = {'additional_special_tokens': e_tags}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            self.sentbert.resize_token_embeddings(len(self.tokenizer))  # enlarge vocab
        
        self.embed_dim = self.sentbert.config.hidden_size
        #如果不放开bert的话就冻住
        self.out = nn.Linear(3*self.embed_dim,self.embed_dim) # 这个是md+word的拼接
        self.out1 = nn.Linear(self.embed_dim,n_classes)
        self.out2 = nn.Linear(self.embed_dim*2,n_classes)

    @staticmethod
    def cls_pooling(model_output):
        return model_output[0][:,0]  # model_output[0] 表示 last hidden state, bert_output[0].shape => [bs,max_len,d_model]
    @staticmethod   
    def word_pooling(model_output):
        return model_output[0][:,1:-1,:].mean(1)  # model_output[0] 表示 last hidden state, bert_output[0].shape => [bs,max_len,d_model]
    def find_tags_pos(self,input_ids_arr,tokenizer,special_token):
        """
        return: special_token的位置
        args:
            input_ids_arr like:
            tensor([[  101,  9499,  1071,  2149, 30522,  8696, 30522, 30534,  6874,  9033,
                4877,  3762, 30534, 10650,  1999, 12867,  1024,  5160,   102,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                    0,     0,     0,     0,     0,     0,     0,     0],
                [  101,  2019, 21931, 17680,  2013, 11587, 30532,  2149, 30532, 14344,
                5016, 30537,  2406, 22517,  3361, 30537,  2006,  5958,  1010, 11211,
                2007, 10908,  2005,  1037,  2149,  3446,  3013,  2006,  9317,  1010,
                2992,  8069,  2008,  1996, 23902,  2013,  1996,  2149,  3847, 24185,
                2229,  2003, 24070,  1010, 16743,  2056,  1012,   102]])
            tokenizer:
                as named
            special_token:
                like: "<VERB>"
        """
        target_pos = tokenizer.convert_tokens_to_ids(special_token) # position of special token
        input_ids_arr_np = input_ids_arr.cpu().numpy()
        target_index = []
        for item in input_ids_arr_np:
            res = np.argwhere(item==target_pos)
            if len(res)==0:
                target_index.append(0)
            else:
                target_index.append(res[0][0])
        target_index = np.array(target_index)
        position = np.argwhere(input_ids_arr_np==target_pos)[:,1]
        assert len(input_ids_arr_np)==len(target_index)
        # try:
        #     assert len(input_ids_arr_np)==len(position)
        # except:
        #     print("有词没有被tag !")
        #     sys.exit() # 有词没有被tag
        return target_index


    def get_embeddings(self, text_arr):
        """
        用CLS model直接得到句子的representation
        """
        #这里的x都是文本
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        #feature的value都放到device中
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        bert_output = self.sentbert.forward(**feat_text)

        #计算embedding (CLS output)
        embedding = BERT.cls_pooling(bert_output)

        return embedding

    def get_word_embeddings(self, text_arr):
        """
        用CLS model直接得到句子的representation
        """
        #这里的x都是文本
        feat_text= self.tokenizer.batch_encode_plus(text_arr, 
                                                    max_length=self.max_length+2,  # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        #feature的value都放到device中
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)
        bert_output = self.sentbert.forward(**feat_text)

        #计算embedding (CLS output)
        embedding = BERT.word_pooling(bert_output)

        return embedding
    
    def get_embeddings_PURE(self,text_arr):
        """
        from paper:
            A Frustratingly Easy Approach for Entity and Relation Extraction
            ent1_spos 是每个句子的entity1的开始位置
            ent2_spos 是每个句子的entity2的开始位置
        """
        feat_text= self.tokenizer.batch_encode_plus(text_arr,   # +2是因为CLS 和SEQ也算进去max_length的
                                                    return_tensors='pt', 
                                                    padding='longest',
                                                    truncation=True)
        verb_start = self.find_tags_pos(feat_text['input_ids'],self.tokenizer,"<VERB>")+1 # 它的位置+1就是 word的位置
        # 放到device中
        for k,_ in feat_text.items():
            feat_text[k] = feat_text[k].to(self.device)

        bert_output = self.sentbert.forward(**feat_text)
        bert_output = bert_output[0]
        bs = bert_output.shape[0]
        ent1_spos = torch.tensor(verb_start).long()
        embeddings_word = bert_output[[i for i in range(bs)],ent1_spos,:]  # 得到这个词的embedding
        embeddings_cls = bert_output[:,0]
        embeddings = torch.cat([embeddings_cls,embeddings_word],dim = 1)
        return embeddings,embeddings_word  # [bs, d_model]
    def Out(self,out):
        out = self.out(out)
        out = F.dropout(out,0.01)
        out = self.out1(out)
        return out
    def forward_word(self,texts,words):
        sen_embed = self.get_embeddings_PURE(texts)[0]  # 得到这个句子的embed
        word_embed = self.get_word_embeddings(words)
        embed = torch.cat([sen_embed,word_embed],dim = 1)
        out = self.out(embed)
        return out
    def forward1(self,texts):
        sen_embed = self.get_embeddings_PURE(texts)[0]
        out = self.out2(sen_embed)
        return out

