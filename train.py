from transformers import BertModel,BertTokenizer
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from model import BertClassifier
from sklearn.metrics import roc_curve, auc
import os

TRAIN_PATH= 'dataset/Toxic_Comment/train.csv'
BEST_MODEL_WEIGHTS_PATH='best_weights/Toxic_Comment/pred.csv'
BERT_PATH= 'bert-base-cased'

tokenizer=BertTokenizer.from_pretrained(BERT_PATH)
print(tokenizer.tokenize('I have a good time, thank you.'))
bert=BertModel.from_pretrained(BERT_PATH)
print('load bert best_weights over')

train_df=pd.read_csv(TRAIN_PATH)
train_df.head()

train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
print(train_df.shape)
print(val_df.shape)

class Dataset(torch.utils.data.Dataset):
  def __init__(self,df):
    self.labels=torch.tensor(df.iloc[:,2:8].values,dtype=torch.float32)
    self.texts=[
        tokenizer(
            text,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        for text in df['comment_text']
    ]

  def classes(self):
    return self.labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    batch_texts=self.texts[idx]
    batch_y=self.labels[idx]
    return batch_texts, batch_y

def load_data(train_data,val_data):
  train,val=Dataset(train_data),Dataset(val_data)
  train_dataloader=torch.utils.data.DataLoader(train,batch_size=16,shuffle=True)
  val_dataloader=torch.utils.data.DataLoader(val,batch_size=16)
  return train_dataloader,val_dataloader

train_dataloader,val_dataloader=load_data(train_df,val_df)

def get_metrics(y_pred,y_true):
  y_pred_cpu=y_pred.to("cpu").detach().numpy()
  for i in range(y_pred_cpu.shape[1]):
    fpr, tpr, thresholds = roc_curve(y_true[:,i],y_pred_cpu[:,i])
    roc_auc = auc(fpr,tpr)
  return roc_auc

def train(model,train_dataloader,train_data_len,val_dataloader,val_data_len,learning_rate,epoches,save_dir=BEST_MODEL_WEIGHTS_PATH):
  use_cuda=torch.cuda.is_available()
  print(use_cuda)
  device=torch.device("cuda" if use_cuda else "cpu")

  criterion=nn.BCEWithLogitsLoss()
  optimizer=Adam(model.parameters(),lr=learning_rate)

  if use_cuda:
    model=model.cuda()
    criterion=criterion.cuda()

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  best_auc_val=-1.0
  best_model_weights=None

  print("strat training!")
  for epoch_num in range(epoches):
    outputs_train = []
    total_loss_train=0
    for train_input,train_label in tqdm(train_dataloader):
      train_label=train_label.to(device)
      mask=train_input['attention_mask'].to(device)
      input_id=train_input['input_ids'].squeeze(1).to(device)

      output=model(input_id,mask)

      batch_loss=criterion(output,train_label)
      total_loss_train+=batch_loss.item()
      model.zero_grad()
      batch_loss.backward()
      optimizer.step()

      outputs_train.append(output)

    total_outputs_train=torch.cat(outputs_train,0)
    auc_train=get_metrics(total_outputs_train,train_df.iloc[:,2:8].values)

    outputs_val=[]
    total_loss_val=0
    with torch.no_grad():
      for val_input,val_label in val_dataloader:
        val_label=val_label.to(device)
        mask=val_input['attention_mask'].to(device)
        input_id=val_input['input_ids'].squeeze(1).to(device)

        output=model(input_id,mask)
        outputs_val.append(output)

        batch_loss=criterion(output,val_label)
        total_loss_val+=batch_loss.item()
      total_outputs_val=torch.cat(outputs_val,0)
      auc_val=get_metrics(total_outputs_val,val_df.iloc[:,2:8].values)

    if auc_val>best_auc_val:
      best_auc_val=auc_val
      best_model_weights=model.state_dict()

    print(
        f'''Epoches:{epoch_num+1}
        Train Loss:{total_loss_train/train_data_len: .3f}
        Train Auc:{auc_train: .3f}
        Val Loss:{total_loss_val/val_data_len: .3f}
        Val Auc:{auc_val: .3f}
        '''
    )
  torch.save(best_model_weights,os.path.join(save_dir,'best_weights.pth'))

EPOCHES=4
LR=5e-5
model=BertClassifier()
train(model,train_dataloader,len(train_df),val_dataloader,len(val_df),LR,EPOCHES,BEST_MODEL_WEIGHTS_PATH)