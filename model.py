import pytorch_lightning as pl
from transformers import AutoModelForTokenClassification
import torch.nn as nn
import torchmetrics 
import torch.optim as optim
import torchmetrics 
import logging


class NERClassification(pl.LightningModule):
    
    def __init__(self,plm, num_labels):
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(plm, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=30)
        
    
    def forward(self, input_ids, attention_mask):
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        
        )
        logits = outputs.logits
        return logits

        
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        
        
        loss_fn = nn.CrossEntropyLoss(ignore_index= 29)
        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        
        
        # 정확도를 계산할때, PAD 인덱스 찾아서 없애야한다
        acc = self.accuracy(logits.argmax(dim=-1), labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        labels = batch['labels']
        logits = self(input_ids, attention_mask)
        
        loss_fn = nn.CrossEntropyLoss(ignore_index= 29)
        loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        acc = self.accuracy(logits.argmax(dim=-1), labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=2e-5)
    
    
    
# seqeval --> F1 값을 계산
# 정밀도 : co-reference