# LegalBERT.py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Paths to your processed CSV files
train_csv = "CUAD_v1/processed/train_clauses.csv"
val_csv = "CUAD_v1/processed/validation_clauses.csv"
test_csv = "CUAD_v1/processed/test_clauses.csv"

# Load data
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

#print(f"Columns: {train_df.columns.tolist()}")  # Debug: should be ['paragraph', 'clause', 'label']

# Create label mapping from all unique labels
all_labels = sorted(list(set(train_df['label'].tolist() +
                             val_df['label'].tolist() +
                             test_df['label'].tolist())))

label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label = {idx: label for label, idx in label2id.items()}

# print("Label mapping:", label2id)

# Load LegalBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")

class ContractDataset(Dataset):
    def __init__(self, dataframe, tokenizer, label2id, max_length=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        paragraph = str(self.data.iloc[idx]['paragraph'])
        clause = str(self.data.iloc[idx]['clause'])
        label = self.data.iloc[idx]['label']

        # For sequence classification, we can concatenate clause + paragraph as input
        text = f"{clause} [SEP] {paragraph}"

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.label2id[label], dtype=torch.long)
        return item

# Create datasets
train_dataset = ContractDataset(train_df, tokenizer, label2id)
val_dataset = ContractDataset(val_df, tokenizer, label2id)
test_dataset = ContractDataset(test_df, tokenizer, label2id)

# Optional: create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
# print("Ready for training LegalBERT on CUAD!")
