from random import random
import sys
import os
from pathlib import Path
from turtle import forward

CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(CURR_DIR)
P = PATH.parent
for i in range(3): # add parent path, height = 3
    P = P.parent
    sys.path.append(str(P.absolute()))

from utils import MDFeat, save, load, set_global_random_seed
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import BERT
from sentence_transformers import SentenceTransformer
import argparse
import torch.utils.data as util_data
from typing import List
import numpy as np
import random
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

class mydataset(Dataset):
    def __init__(self, data:List[MDFeat]) -> object:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "sentence":self.data[idx].sentence,
            "verb":self.data[idx].verb,
            "verb_idx":self.data[idx].verb_idx,
            "label":self.data[idx].label,
            "aug_verb":random.choice(self.data[idx].addinfo1) if self.data[idx].addinfo1 is not None else None,
            "aug_others":random.choice(self.data[idx].addinfo2) if self.data[idx].addinfo2 is not None else None,
        }


class mydataset_test(Dataset):
    def __init__(self, data:List[MDFeat]) -> object:
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "sentence":self.data[idx].sentence,
            "verb":self.data[idx].verb,
            "verb_idx":self.data[idx].verb_idx,
            "label":self.data[idx].label
        }


class cosine_loss(nn.Module):
    def __init__(self):
        super(cosine_loss, self).__init__()
    def forward(self,x,y):
        similarity = torch.cosine_similarity(x, y, dim=1)
        loss = 1 - similarity
        return loss

def data_preprocess(data:List[MDFeat]):
    # 给目标单词加上tags
    datas:List[MDFeat] = []
    for d in data:
        sentence_split = d.sentence.split()
        sentence_split[d.verb_idx] = " <VERB> {} </VERB> ".format(sentence_split[d.verb_idx])
        d.sentence = " ".join(sentence_split)
        datas.append(d)
    return datas

def md_main(args):
    set_global_random_seed(args.seed)
    args.device = device = torch.device('cuda:{}'.format(args.cuda_index))
    train_dataset = mydataset(data_preprocess(load(args.train_path)))
    if args.mode!="10fold":
        dev_dataset = mydataset_test(data_preprocess(load(args.dev_path)))
        dev_loader = util_data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dataset = mydataset_test(data_preprocess(load(args.test_path)))
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = util_data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # 加载模型
    bert_model  = SentenceTransformer(args.model_dir)
    model:BERT = BERT(bert_model,args.max_len,device,args.n_classes,e_tags = ['<VERB>','</VERB>']).to(device)
    optimizer = torch.optim.AdamW([
        {'params':model.sentbert.parameters()},
        {'params':model.out.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr,weight_decay=0.001)
    CEloss = nn.CrossEntropyLoss()
    Cosloss = cosine_loss()
    best_test = np.array([0.,0.,0.,0.])
    base_f1 = -1
    for i_batch, batch in enumerate(train_loader):
        sentence = batch['sentence']
        sentence_neg = batch['aug_verb']
        sentence_pos = batch['aug_others']
    for i_epoch in range(args.epochs):
        model.train()
        num_iter = len(train_loader)
        batch_losses = []
        batch_pos_loss = []
        batch_neg_loss = []
        for i_batch, batch in enumerate(train_loader):
            label = batch['label'].to(device)
            pos_index = label.bool()
            neg_index = pos_index==False
            sentence = batch['sentence']
            sentence_neg = batch['aug_verb']
            sentence_pos = batch['aug_others']
            embed_sen = model.get_embeddings_PURE(sentence)[0]
            embed_neg = model.get_embeddings_PURE(sentence_neg)[0]
            embed_pos = model.get_embeddings_PURE(sentence_pos)[0]
            feat_sen = F.normalize(embed_sen,dim=1)
            feat_neg = F.normalize(embed_neg,dim=1)
            feat_pos = F.normalize(embed_pos,dim=1)

            ## 首先计算pos的
            

            pos_loss = Cosloss(feat_sen,feat_pos).mean() #  所有样本, 拉进 ori和mask_ot
            pos_loss += Cosloss(feat_sen[neg_index],feat_neg[neg_index]).mean() # 负样本: 拉进ori和mask_verb
            neg_loss = (1-Cosloss(feat_sen[pos_index],feat_neg[pos_index])).mean()  # 拉远正样本的 ori 和mask_verb
            logits = model.out2(embed_sen)
            loss = CEloss(logits,label)+pos_loss*2+neg_loss*2
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_losses.append(loss.detach().cpu().item())
            batch_pos_loss.append(pos_loss.detach().cpu().item())
            batch_neg_loss.append(neg_loss.detach().cpu().item())
            sys.stdout.write('\r')
            sys.stdout.write('Trian : | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f \t pos_loss: %.4f \t neg_losss: %.4f'
                    %( i_epoch+1, args.epochs, i_batch+1, num_iter, np.mean(batch_losses), np.mean(batch_pos_loss), np.mean(batch_neg_loss)))
            sys.stdout.flush()
            
        model.eval()
        dev_preds = []
        dev_gt = []
        if args.mode!="10fold":
            with torch.no_grad():
                for i_batch, batch in enumerate(dev_loader):
                    logits = model.forward1(batch['sentence'])
                    _, pred = torch.max(logits.data, -1) 
                    dev_preds.extend(pred.detach().cpu().numpy().tolist())
                    dev_gt.extend(batch['label'].numpy().tolist())
                dev_acc = accuracy_score(dev_gt, dev_preds)
                dev_pre = precision_score(dev_gt, dev_preds)
                dev_rec = recall_score(dev_gt, dev_preds)
                dev_f1 = f1_score(dev_gt, dev_preds)
                # dev_pre, dev_rec, dev_f1, _ = precision_recall_fscore_support(
                #     dev_gt, dev_preds, average="macro")
                # dev_acc = sum(np.array(dev_preds)==np.array(dev_gt))/len(dev_gt)
                sys.stdout.write("\n")
                sys.stdout.write('Dev   : | Epoch [%3d/%3d] f1: %.4f acc: %.4f pre: %.4f recall: %.4f  '
                        %( i_epoch+1, args.epochs, dev_f1, dev_acc,dev_pre,dev_rec))
                sys.stdout.write("\n")
                if dev_f1>base_f1:
                    base_f1 = dev_f1
                    print("This is the best!")
                dev_preds = []
                dev_gt = []
        with torch.no_grad():
            for i_batch, batch in enumerate(test_loader):
                logits = model.forward1(batch['sentence'])
                _, pred = torch.max(logits.data, -1) 
                dev_preds.extend(pred.detach().cpu().numpy().tolist())
                dev_gt.extend(batch['label'].numpy().tolist())
            # dev_pre, dev_rec, dev_f1, _ = precision_recall_fscore_support(
            #     dev_gt, dev_preds, average="macro")
            # dev_acc = sum(np.array(dev_preds)==np.array(dev_gt))/len(dev_gt)
            dev_acc = accuracy_score(dev_gt, dev_preds)
            dev_pre = precision_score(dev_gt, dev_preds)
            dev_rec = recall_score(dev_gt, dev_preds)
            dev_f1 = f1_score(dev_gt, dev_preds)
            sys.stdout.write('Test  : | Epoch [%3d/%3d] f1: %.4f acc: %.4f pre: %.4f recall: %.4f '
                    %( i_epoch+1, args.epochs, dev_f1, dev_acc,dev_pre,dev_rec))
            sys.stdout.write("\n")
            sys.stdout.write("\n")
            if dev_f1>base_f1:
                base_f1 = dev_f1
                best_test = np.array([dev_f1,dev_acc,dev_pre,dev_rec])
    return best_test   




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_index", type=int,default=1, help="as named")
    parser.add_argument("--batch_size", type=int,default=32, help="as named")
    parser.add_argument('--seed', type=int, default=16, help='as named')
    ## VUA_verb
    parser.add_argument("--train_path", type=str,default="/home/tywang/MD/data/VUA/vua_train_mlm.pkl", help="as named")
    parser.add_argument("--dev_path", type=str,default="/home/tywang/MD/data/VUA/vua_dev.pkl", help="as named")
    parser.add_argument("--test_path", type=str,default="/home/tywang/MD/data/VUA/vua_test.pkl", help="as named")

    ## VUA_pos
    # parser.add_argument("--train_path", type=str,default="/home/tywang/MD/data/VUA_ALL_POS/vua_pos_train.pkl", help="as named")
    # parser.add_argument("--dev_path", type=str,default="/home/tywang/MD/data/VUA_ALL_POS/vua_pos_dev.pkl", help="as named")
    # parser.add_argument("--test_path", type=str,default="/home/tywang/MD/data/VUA_ALL_POS/vua_pos_test.pkl", help="as named")

    parser.add_argument("--model_dir", type=str,default='/data/transformers/bert-base-uncased', help="as named")
    parser.add_argument('--max_len', type=int, default=64, help='length of input sentence')
    parser.add_argument('--n_classes', type=int, default=2, help=' ')
    parser.add_argument('--epochs', type=int, default=100, help='Emax epochs of algorithm2 in noisy CV paper')
    parser.add_argument('--lr', type=float, default=1e-5,help='learning rate')
    parser.add_argument('--lr_scale', type=int, default=100, help='as named')
    parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
    parser.add_argument('--mode', default="normal", type=str, help='if 10fold')
    args = parser.parse_args()

    # train_p = '/home/tywang/MD/data/MOH-X/mohx{}_train.pkl'
    # val_p = '/home/tywang/MD/data/MOH-X/mohx{}_val.pkl'
    # train_p = '/home/tywang/MD/data/TroFi/trofi{}_train.pkl'
    # val_p = '/home/tywang/MD/data/TroFi/trofi{}_val.pkl'
    # best_test = np.array([0.,0.,0.,0.])
    # if args.mode=="10fold":
    #     for i in range(10):
    #         print(i)
    #         args.train_path = train_p.format(i)
    #         args.test_path = val_p.format(i)
    #         best = md_main(args)
    #         print(best)
    #         best_test+=best
    #         print("curr: ",best_test/(i+1))

    #     print(best_test/10) # 打印1- fold的最终结果
    # else:
    #     md_main(args)

    md_main(args)


