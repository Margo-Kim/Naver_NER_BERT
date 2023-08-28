from data import *
from data_loader import *
from model import *
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
import torch.nn as nn
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.utils import shuffle
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


plm = "klue/roberta-base"
batch_size = 32

all_input_ids, all_attention_masks, all_token_type_ids, all_label = preprocess()

all_input_ids = all_input_ids[:200]
all_attention_masks = all_attention_masks[:200]
all_token_type_ids = all_token_type_ids[:200]
all_label = all_label[:200]


train_input_ids, val_input_ids, train_attention_masks, val_attention_masks, train_token_type_ids, val_token_type_ids, train_labels, val_labels = train_test_split(
        all_input_ids, all_attention_masks, all_token_type_ids, all_label, test_size=0.2, random_state=42
    )


train_dataset = NERDataset(train_input_ids, train_attention_masks, train_token_type_ids, train_labels)
val_dataset = NERDataset(val_input_ids, val_attention_masks, val_token_type_ids, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


model = NERClassification (plm, 30)

trainer = pl.Trainer(
        max_epochs=1,
        logger=TensorBoardLogger(save_dir='logs/', name='ner_model')
    )

trainer.fit(model, train_dataloader, val_dataloader)

os.makedirs('saved_models', exist_ok=True)
trainer.save_checkpoint('saved_models/ner_model.ckpt')

model = NERClassification.load_from_checkpoint('saved_models/ner_model.ckpt', plm=plm, num_labels=30)

tokenizer = AutoTokenizer.from_pretrained(plm)

max_seq_length = 512

input_text = '나는 대한민국 국민으로 학교에 갔었다'
tokenized_input = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
input_ids = tokenized_input['input_ids']
attention_mask = tokenized_input['attention_mask']
token_type_ids = tokenized_input['token_type_ids'] 

model.eval()

with torch.no_grad():  
    logits = model(input_ids, attention_mask)

predicted_labels = torch.argmax(logits, dim=-1)

print(predicted_labels)