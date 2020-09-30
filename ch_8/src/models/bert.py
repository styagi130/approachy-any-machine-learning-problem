import torch
import transformers

class BertClassificationModel(torch.nn.Module):
    """
        Class for all bert classification
    """
    def __init__(self, config):
        """
            Function to instantiate bertclassification class
            :param config: A SimpleNamespace config object
        """
        super(BertClassificationModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.bert_model)
        self.bert_drop = torch.nn.Dropout(0.5)
        self.out = torch.nn.Linear(768, 1)

    def forward(self, inputs, input_masks, input_type):
        _, o2 =self.bert(inputs, attention_mask=input_masks, token_type_ids=input_type)
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output