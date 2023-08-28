import torch
from torch.utils.data import Dataset, DataLoader


class NERDataset(Dataset):
    
    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = labels
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(self.token_type_ids[idx], dtype=torch.long),
            'labels': torch.tensor(self.label[idx], dtype=torch.long)
                  
        }

