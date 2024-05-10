import torch
from transformers import BertModel,BertTokenizer
import pandas as pd
from tqdm import tqdm
from model import BertClassifier

TEST_PATH= 'dataset/Toxic_Comment/test.csv'
PREDICTION_PATH='predictions/pred.csv'
BERT_PATH= 'bert-base-cased'
BEST_MODEL_WEIGHTS_PATH= 'best_weights/Toxic_Comment/best_weights.pth'

tokenizer=BertTokenizer.from_pretrained(BERT_PATH)
print(tokenizer.tokenize('I have a good time, thank you.'))
bert=BertModel.from_pretrained(BERT_PATH)
print('load bert best_weights over')

test_df=pd.read_csv(TEST_PATH)
test_df.head()
print(test_df.shape)

class Dataset(torch.utils.data.Dataset):
  def __init__(self,df):
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

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    batch_texts=self.texts[idx]
    return batch_texts

test_dataset=Dataset(test_df)
test_dataloader=torch.utils.data.DataLoader(test_dataset,batch_size=16)

model=BertClassifier()

model.load_state_dict(torch.load(BEST_MODEL_WEIGHTS_PATH))

def evaluate(model, test_dataloader):
  use_cuda=torch.cuda.is_available()
  print(use_cuda)
  device=torch.device("cuda" if use_cuda else "cpu")
  if use_cuda:
    model=model.cuda()
  model.eval()

  outputs = []

  with torch.no_grad():
    for test_input in tqdm(test_dataloader):
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      output = model(input_id, mask)
      outputs.append(output)
    total_outputs=torch.cat(outputs,0)
  outputs_df=pd.DataFrame(total_outputs.detach().numpy(),columns=['toxic','severe_toxic','obscene','threat','insult','identity_hate'])
  final_outputs_df=pd.concat([test_df['id'],outputs_df],axis=1)
  final_outputs_df.to_csv(PREDICTION_PATH,index=False)

evaluate(model, test_dataloader)