import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
from tn import MyTokenizer

# import ds
ds = load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

corpus = ' '.join(ds['text'])

# my tokenizer
my_tn = MyTokenizer()

'''
my_tn.train(corpus[:1000000], 7000)

my_tn.save('tokenizer save/my_tn')
'''

my_tn.load('tokenizer save/my_tn')

# comparator tokenizer
'''
tokenizer = Tokenizer(models.BPE())

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=7000, show_progress=True)

tokenizer.train_from_iterator(corpus.split('\n'), trainer=trainer)

tokenizer.save('tokenizer save/comp_tn.json')
'''

comp_tn = Tokenizer.from_file('tokenizer save/comp_tn.json')

# compare
seq = 'The dollar is one of the most famous currencies in the world.'

# my tn results
print('MY')

my_encode = my_tn.encode(seq)

print([my_tn.vocab[id] for id in my_encode])
print(my_encode)
print(len(my_encode))
print('\n')

# comparator tn results
print('COMPARATOR')

comp_encode = comp_tn.encode(seq).ids

print([comp_tn.id_to_token(id) for id in comp_encode])
print(comp_encode)
print(len(comp_encode))
