# Tokenizer
⚠️ **Important:** This is not a a "competitive" tokenizer (a tokenizer similar to the one used by the state-of-the-arte models) 
it's a basic tokenizer build it to show the basic structure of a tokenizer

Subworkd Tokenizer based on the BPE (Byte-Level paire encoding) algorithm used in LLM tokenization, check this site 
https://huggingface.co/learn/nlp-course/en/chapter6/5 if you want to know more about BPE tokenization and in general
about tokenization

## Files
### src
- tn.py: Code of the tokenizer
- tn_fn.py: External functions used by the tokenizer, there are two functions `get_pairs()` and `merge()` that are the two functions that make works the BPE algorithm

### test
- compare.py: Used during the developing process to make sure that my tokenizer was working as well as the hugging face tokenizer (that it's a famous tokenizer library)

### tokenizer save
folder where the tokenizer data are saved after calling the function `tn.save()`, there is also the function `tn.load()` in case you want load a saved tokenizer
  
## How to use it
⚠️ **Important:** Before running the code rember to install the needed packages, so run this command: `pip install datasets`
### train tokenizer
```python
from datasets import load_dataset
from tn import Tokenizer

# import ds
ds = load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

corpus = ' '.join(ds['text'])

tn = Tokenizer()

# train tokenizer
tn.train(corpus=corpus[:1000000], vocab_size=7000)
```

### save tokenizer
```python
from datasets import load_dataset
from tn import Tokenizer

# import ds
ds = load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

corpus = ' '.join(ds['text'])

tn = Tokenizer()

# train tokenizer
tn.train(corpus=corpus[:1000000], vocab_size=7000)

# save tokenizer
tn.save('my_tokenizer')
```

### load, encode and decode
```python
from tn import Tokenizer

tn = Tokenizer()

# load tokenizer
tn.load('my_tokenizer')

# encode
seq = 'The square root of 4 is 2, right?'

ids = tn.encode(seq)

# decode
seq_back = tn.decode(ids)
```

### special ids (tokens)
```python
from tn import Tokenizer

# import ds
ds = load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

corpus = ' '.join(ds['text'])

tn = Tokenizer()

# set special ids
tn.set_special_ids(['<|im_start|>'])

# train tokenizer
tn.train(corpus=corpus[:1000000], vocab_size=7000)

# encode
seq = '<|im_start|> The square root of 4 is 2, right?'

ids = tn.encode(seq) # ids_str: ['<|im_start|>', ' The', ' square, ' root',  ' of',  ' 4',  ' is', ' 2', 'right', ' ?']
```

