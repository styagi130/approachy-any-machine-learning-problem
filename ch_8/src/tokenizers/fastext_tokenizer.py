import json
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize

class Fasttext:
    def __init__(self, text_wv_file, return_wv=False):
        """
            Class for all fasttext related interactions
            :param text_wv_file: path to word vectors in text format
        """
        self.text_wv_file = text_wv_file
        base_dir = text_wv_file.parent

        self.word2idx_filename = base_dir / f"{text_wv_file.name}_word2idx.json"
        self.idx2word_filename = base_dir / f"{text_wv_file.name}_idx2word.json"
        self.wv_filename = base_dir / f"{text_wv_file.name}_wv.npy"

        self.num_words = self.cal_vocab_size()
        self.embedding_dim = 300

        if not self.word2idx_filename.exists():
            self.create_word2idx()
        if not self.idx2word_filename.exists():
            self.create_idx2word()
        if not self.wv_filename.exists():
            self.create_wv()
        
        self.load_meta_maps()
        self.gen_unk_token_idx()
        if return_wv:
        	self.wv_matrix = self.return_wv_matrix()

    def cal_vocab_size(self):
        """
            Function to calcualte total vocabulary size
            :return: integer denoting total words in vocabulary
        """
        num = 0
        with self.text_wv_file.open() as wv_file:
            for line in wv_file:
                num += 1
        return num

    def create_wv(self):
        """
            Function to create and persist numpy array of word vectors
        """
        embedding_matrix = np.zeros((self.num_words+1, self.embedding_dim))

        with self.text_wv_file.open() as wv_file:
            for idx, line in tqdm.tqdm(enumerate(wv_file), total=self.num_words, desc="embedding_matrix"):
                embedding_matrix[idx,:] = [float(item) for item in line.rstrip().split(' ')[1:]]
        embedding_matrix[-1,:] = np.random.normal(size=(300,))    
        np.save(self.wv_filename, embedding_matrix)

    def create_word2idx(self):
        """
            Function to create index to word map
        """
        word2idx = {}
        with self.text_wv_file.open() as wv_file:
            for idx, line in tqdm.tqdm(enumerate(wv_file), total=self.num_words, desc="word2index"):
                word2idx[line.rstrip().split(' ')[0]] = idx
        with self.word2idx_filename.open("w") as f:
            json.dump(word2idx, f, indent=4, ensure_ascii=False)

    def create_idx2word(self):
        """
            Function to create index to word map
        """
        idx2word = {}
        with self.text_wv_file.open() as wv_file:
            for idx, line in tqdm.tqdm(enumerate(wv_file), total=self.num_words, desc="idx2word"):
                idx2word[idx] = line.rstrip().split(' ')[0]
        with self.idx2word_filename.open("w") as f:
            json.dump(idx2word, f, indent=4, ensure_ascii=False)

    def load_meta_maps(self):
        """
            Function to load word2idx, idx2word in memory
        """
        with self.word2idx_filename.open() as word2idx_file:
            self.word2idx_dict = json.load(word2idx_file)
        with self.idx2word_filename.open() as idx2word_file:
            self.idx2word_dict = json.load(idx2word_file)

    def return_wv_matrix(self):
        """
            Function to return word2vec matrix
        """
        return np.load(self.wv_filename).astype(np.float32)

    def return_wv(self, indexes):
    	"""
    		Function to return word vectors given index
    		:param indexes: List of indexes
    	"""
    	assert self.wv_matrix is not None, f"pass return wv as true in tokenizer"
    	return self.wv_matrix[indexes]

    def gen_unk_token_idx(self):
        """
            Function to generate index of the unique token
        """
        self.unk_index = self.num_words
        self.num_words += 1

    def tokenize_encode(self, sentence):
        """
            Function to tokenize and generate an ineteger sequence from words
            :param sentence: An string with all the words
            :return: A sequence with inetger encodings
        """
        words = word_tokenize(sentence)
        seq = []
        for word in words:
            if word in self.word2idx_dict.keys():
                seq.append(self.word2idx_dict[word])
            else:
                seq.append(self.unk_index)
        return seq

    def decode(self, sequence):
        """
            Function to generate an words from ineteger sequence
            :param sequence: An string with all the inetegers encodings
            :return: english decoded sequence
        """
        sentence = []
        for wordID in sequence:
            if str(wordID) in self.idx2word_dict.keys():
                sentence.append(self.idx2word_dict[str(wordID)])
            else:
                sentence.append("<OOV>")
        return sentence

if __name__=="__main__":
    import argparse
    import pathlib
    import sys
    sys.path.append("./../../")
    from src.utils import io
    
    parser = argparse.ArgumentParser(description = "Use this to create a stratified k-folded dataframe")
    parser.add_argument("config_filepath", metavar="fc", type=pathlib.Path, help="config filepath")
    args = parser.parse_args()

    config = io.load_config(args.config_filepath)
    wv_filepath = config.wv_filepath
    tokenizer =  Fasttext(wv_filepath)

    sentence = "HI!!!! I am siddharth hello how are you?? asa sadaiufaf oia fyaofafafasf"
    print (f"Orignal sentence is: \"{sentence}\"")
    sequence = tokenizer.tokenize_encode(sentence)
    print  (f"Encoded sequence is: \"{sequence}\"")
    decoded_sentence = tokenizer.decode(sequence)
    print  (f"decoded sequence is: \"{' '.join(decoded_sentence)}\"")
    embedding_matrix = tokenizer.return_wv()
    print (f"word vectors shape is: {embedding_matrix.shape}")