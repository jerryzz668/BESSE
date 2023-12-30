from torch.utils.data import DataLoader, Dataset

# 参数解析器和TextDataset类...
class CoupletDataset(Dataset):
    def __init__(self, tokenizer, source_path, target_path, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_lines = []
        self.target_lines = []

        with open(source_path, 'r', encoding='utf-8') as f:
            self.source_lines = f.readlines()

        with open(target_path, 'r', encoding='utf-8') as f:
            self.target_lines = f.readlines()

        assert len(self.source_lines) == len(self.target_lines), "Source and target files must have the same number of lines."

    def __len__(self):
        return len(self.source_lines)

    def __getitem__(self, idx):
        source_line = self.source_lines[idx]
        target_line = self.target_lines[idx]

        source_encoded = self.tokenizer.encode_plus(
            source_line, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )
        target_encoded = self.tokenizer.encode_plus(
            target_line, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids': source_encoded['input_ids'].squeeze(0),
            'attention_mask': source_encoded['attention_mask'].squeeze(0),
            'labels': target_encoded['input_ids'].squeeze(0),
            'decoder_attention_mask': target_encoded['attention_mask'].squeeze(0)
        }