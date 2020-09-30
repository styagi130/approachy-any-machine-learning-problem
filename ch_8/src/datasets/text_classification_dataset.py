import torch
import numpy as np

class TextClassifier:
    def __init__(self, feature_label_list, tokenizer_encoder):
        """
        :param feature_label_list: a feature_label tuple
        :param tokenizer_encoder: a text tokenizer and text to seq converter
        """
        self.features_label = feature_label_list
        self.tokenizer_encoder = tokenizer_encoder
        self.sort_items()

    def __len__(self):
        return len(self.features_label)

    def sort_items(self):
        """
            Function to sort items
        """
        self.features_label = [(self.tokenizer_encoder.tokenize_encode(text), label) for text, label in self.features_label]
        self.features_label.sort(key=lambda x: len(x[0]), reverse = True)

    def __getitem__(self, idx):
        return {"text":torch.tensor(self.features_label[idx][0]), "label": torch.tensor(self.features_label[idx][1])}

    def __pad__text(self, text_list, max_len):
        batch_size = len(text_list)
        padded_batch = np.zeros((batch_size, max_len),np.int)
        for idx, text in enumerate(text_list):
            padded_batch[idx,:text.shape[0]] = text
        return padded_batch


    def collate_fn(self, batch):
        labels = [data_dict["label"] for data_dict in batch]
        inputs = [data_dict["text"] for data_dict in batch]
        input_lens = [data_dict["text"].size(0) for data_dict in batch]
        max_len = max(input_lens)

        padded_batch = self.__pad__text(inputs, max_len)
        padded_batch = torch.FloatTensor(self.tokenizer_encoder.return_wv(padded_batch))
        input_lens = torch.tensor(input_lens)    
        labels = torch.tensor(labels, dtype=torch.float)

        return padded_batch, input_lens, labels
