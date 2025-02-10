import regex as re
import json

from bpe import get_pairs, merge

# create tokenizer
class MyTokenizer:
    def __init__(self):
        self.merges = {} # {(1, 2): 4, ..., (32, 123): 23}

        self.special_ids = [] # ['<|im_start|>', '<|im_end|>', '<|im_sep|>] possible special_tokens
        self.vocab = self.build_vocab() # {0: 0, ..., 65: 'A'}

        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def set_special_ids(self, special_ids: list[str]):
        self.special_ids = special_ids

        self.vocab = self.build_vocab()

    def build_vocab(self, loaded=(False, {})):
        '''
        create self.vocab with base 256 ids and if there are also the special ids (<|im_end|>, <|im_start|>, <|im_space|>, ...)
        '''

        vocab = {} 
        self.dict_special_ids = {} # {'<|im_start|>': id, ...}

        if not loaded[0]:
            for id in range(256):
                vocab[id] = chr(id)
            
            for special_id_str in self.special_ids:            
                special_id = max(vocab) + 1

                vocab[id] = special_id_str

                self.dict_special_ids[special_id_str] = special_id
        else:
            json_vocab = loaded[1]

            for id, id_str in json_vocab.items():
                id = int(id)

                vocab[id] =  id_str

                if id_str in self.special_ids:
                    self.dict_special_ids[id_str] = id

        return vocab

    def save(self, file_name: str) -> None:
        '''
        save all the data that tokenizer needs to be loded in a future time
        '''

        with open(file_name + '-merges' + '.json', 'w') as f1, open(file_name + '-vocab' + '.json', 'w') as f2:
            json_merges = {str(k): v for k, v in self.merges.items()}

            json.dump(json_merges, f1)
            json.dump(self.vocab, f2)

    @classmethod
    def load(cls, file_name: str) -> None:
        '''
        load a saved tokenizer
        '''

        with open(file_name + '-merges' + '.json') as f1, open(file_name + '-vocab' + '.json') as f2:
            json_merges = json.load(f1)
            merges = {(int(k.split(',')[0][1:]), int(k.split(',')[1][1:-1])) : v for k, v in json_merges.items()}
     
            json_vocab = json.load(f2)
        
        tn = cls()

        tn.merges = merges
        tn.vocab = tn.build_vocab(loaded=(True, json_vocab))

        return tn

    def train(self, corpus: str, vocab_size: int, test_seq: str = False) -> None:
        '''
        train the tokenizer to increase vocab doing more and more merges 
        '''

        assert vocab_size >= 256, f'vocab_size needs to be >= 256 not {vocab_size}'

        num_merges = vocab_size - 256

        corpus_chunks = re.findall(self.pattern, corpus) # I am tall? -> [I, am, tall, ?]

        ids_chunks = [list(corpus_chunk.encode('utf-8')) for corpus_chunk in corpus_chunks] # [I, am, tall, ?] -> [[1], [3, 4], [5, 3, 6, 6], [23]]

        for i in range(num_merges):
            pairs = {}

            for ids_chunk in ids_chunks:
                get_pairs(ids_chunk, pairs)
        
            pair = max(pairs, key=pairs.get) # pair with the highest freq
            id = max(self.vocab) + 1

            ids_chunks = [merge(ids_chunk, pair, id) for ids_chunk in ids_chunks] # [[1], [3, 4], [5, 3, 7], [23]]  
            
            self.merges[pair] = id
            self.vocab[id] =  self.vocab[pair[0]] + self.vocab[pair[1]] # pair=(23, 1), id=257-> vocab[23] (a) + vocab[1] (m) = am, vocab[257]=am

            # stats
            if i % 100 == 0 or i == num_merges - 1:    
                print(f'MERGES {i} / {num_merges}')
                # print(f'Actual lenght: {bpe.prev_len_ids}, Compressed lenght: {bpe.len_ids}, Improvement: {round(bpe.prev_len_ids / bpe.len_ids, 2)}X')

                if test_seq:
                    test_ids = self.encode(test_seq)
                    ids_str = [self.vocab[test_id] for test_id in test_ids]
        
                    print(ids_str, len(ids_str))

                print('-------------------------------------------------------------------------------------------------------------------------------------------------------------')

    def encode_chunk(self, ids_chunk: list[int]) -> list[int]:
        '''
        given a ids chunk of the ids chunks, ids_chunk=[3, 4, 5, 7] -> ids_chunk[3, 8, 7]
        '''

        while len(ids_chunk) >= 2:
            pairs = {}

            get_pairs(ids_chunk, pairs)

            pair = min(pairs, key=lambda x: self.merges.get(x, float('inf'))) # take the least freq pair

            if pair not in self.merges:
                break

            id = self.merges[pair]

            ids_chunk = merge(ids_chunk, pair, id)
        
        return ids_chunk
    
    def encode_ordinary(self, seq: str) -> list[int]:
        '''
        encode a seq without special ids 
        '''

        seq_chunks = re.findall(self.pattern, seq) # I am tall? -> [I, am, tall, ?]

        ids = []

        for seq_chunk in seq_chunks:
            ids_chunk = list(seq_chunk.encode('utf-8'))

            ids_chunk = self.encode_chunk(ids_chunk)
            
            ids.extend(ids_chunk)
        
        return ids

    def encode(self, seq: str) -> list[int]:
        '''
        given a seq=I am tall -> ids=[1, 34, 23]
        '''
        
        # if there are no special tokens
        if not self.special_ids:
            return self.encode_ordinary(seq)
        
        special_pattern = "(" + "|".join(re.escape(special_id_str) for special_id_str in self.special_ids) + ")" # detect special id

        seq_special_chunks = re.split(special_pattern, seq) # [<|im_start|>, I am tall ?, <|im_end|>]
        
        ids = []

        for seq_special_chunk in seq_special_chunks:
            if seq_special_chunk in self.special_ids:
                special_id = self.dict_special_ids[seq_special_chunk]

                ids.append(special_id)
            else:
                ids.extend(self.encode_ordinary(seq_special_chunk))

        return ids

    def decode(self, ids: list[int]) -> str:
        '''
        given a list of tokens ids=[1, 34, 23] -> I am tall
        '''

        seq = ''.join([self.vocab[id] for id in ids])
        
        return seq

tn = MyTokenizer()
