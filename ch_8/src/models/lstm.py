import torch

class Model(torch.nn.Module):
    def __init__(self, embedding_dim):
        """
            embedding -> Lstm -> output_layer
            :param embedding_matrix: Embedding matrix with all all word vectors. 
        """
        super(Model,self).__init__()

        #num_words = embedding_matrix.shape[0]
        #self.embedding_dim = embedding_matrix.shape[1]
        self.embedding_dim = embedding_dim
        # Initialize embedding matrix
        #self.embedding = torch.nn.Embedding(num_embeddings=num_words, embedding_dim=self.embedding_dim)
        #self.embedding.weight = torch.nn.Parameter(torch.tensor(embedding_matrix, dtype = torch.float))
        #self.embedding.weight.requires_grad = False

        self.lstm = torch.nn.LSTM(self.embedding_dim, 128, bidirectional=True, batch_first=True)
        self.drop_out = torch.nn.Dropout(0.3)

        self.dense = torch.nn.Linear(512,1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, batch, input_lens = None, masking=False):
        """
            Function to run the callable
            :param batch: All inputs
            :param input_lens: Parameter to create masks
        """
        #x = self.embedding(batch)
        x = batch
        if input_lens is not None and masking:
            mask = self.masked_from_lens(input_lens)
            x = x.masked_fill_(~mask, 1e-5)

        x, _ = self.lstm(x)
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)

        dense_input = torch.cat((avg_pool, max_pool), dim=1)
        #dense_input = self.drop_out(dense_input)
        logits = self.dense(dense_input)

        return self.sigmoid(logits)
        
    def masked_from_lens(self, input_lens):
        """
            Function to generate masks for images
        """
        batch_size, max_len = input_lens.size(0), input_lens.max().item()

        sequence_range = torch.arange(0, max_len).unsqueeze(-1).expand((batch_size, max_len, self.embedding_dim)).to(input_lens.device)
        sequence_expanded = input_lens.unsqueeze(1).unsqueeze(-1).expand((batch_size, max_len,self.embedding_dim))

        return sequence_expanded < sequence_range
