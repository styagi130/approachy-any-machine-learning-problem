import torch

class BertClassificationDataset:
    """
        Dataset class for bert
    """
    def __init__(self, feature_label_pair, tokenizer, max_len=512):
        """
            BertClassification dataset
            :param feature_label_pair: A list of tuples
            :param tokenizer: Tokenizer to use
            :param max_len: Maximum number words in 
        """
        self.feature_label_pair = list(feature_label_pair)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        """
            Function to return total number of feature_label_pair
        """
        return len(self.feature_label_pair)

    def __getitem__(self, idx):
        text = self.feature_label_pair[idx][0]
        label = self.feature_label_pair[idx][1]

        inputs = self.tokenizer.encode_plus(text, None, add_special_tokens = True, max_length=self.max_len, pad_to_max_length=True, truncation=True)
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.long), torch.tensor(token_type_ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)
