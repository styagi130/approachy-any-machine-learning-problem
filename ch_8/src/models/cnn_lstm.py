import torch

class Model(torch.nn.Module):
    """
        Model architecture for CNN + LSTM
    """
    def __init__(self, embedding_dim, conv_out_channels=150, kernel_size=3):
        """
            Instantiate model for embedding -> conv -> Lstm -> output_layer
            :param embeddding_matrix: Word embedding matrix
            :kwparam conv_out_channels: Number of dimensions in convolution output
            :kwparam kernel_size: convolution kernel size
        """
        super(Model, self).__init__()
        
        #num_words = embedding_matrix.shape[0]
        self.embedding_dim = embedding_dim #embedding_matrix.shape[1]
        
        # Initialize embedding matrix
        #self.embedding = torch.nn.Embedding(num_embeddings=num_words, embedding_dim=self.embedding_dim)
        #self.embedding.weight = torch.nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float))
        #self.embedding.weight.requires_grad = False

        padding = (kernel_size - 1) // 2
        self.convolution1d = torch.nn.Conv1d(self.embedding_dim,
                                       conv_out_channels,
                                       kernel_size,
                                       padding=padding)
        self.normalization = torch.nn.BatchNorm1d(conv_out_channels, momentum=0.1, eps=1e-5)

        self.lstm = torch.nn.LSTM(conv_out_channels, 64, bidirectional=True, batch_first=True)
        self.dense = torch.nn.Linear(256,1)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, batch, input_lens = None, masking=False):
        """
            Function to run the callable
            :param batch: All inputs
            :param input_lens: Parameter to create masks
        """
        x = batch#self.embedding(batch)
        if input_lens is not None and masking:
            mask = self.masked_from_lens(input_lens)
            x = x.masked_fill_(~mask, 1e-5)

        x = x.permute(0,2,1)
        x = self.convolution1d(x)
        #x = self.normalization(x)
        x = x.permute(0,2,1)

        x, _ = self.lstm(x)
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        dense_input = torch.cat((avg_pool, max_pool), dim=1)
        logits = self.dense(dense_input)
        return self.sigmoid(logits)
