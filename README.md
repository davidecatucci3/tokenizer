# Tokenizer
⚠️ **Important:** This is not a a "competitive" tokenizer (a tokenizer similar to the one used by the state-of-the-arte models) 
it's a basic tokenizer build it to show the basic structure of a tokenizer

Subworkd Tokenizer based on the BPE (Byte-Level paire encoding) algorithm used in LLM tokenization, check this site 
https://huggingface.co/learn/nlp-course/en/chapter6/5 if you want to know more about BPE tokenization and in general
about tokenization

## Files
- src/tn.py: Code of the tokenizer
- src/tn_fn.py: Externak functions used by the tokenizer, there are two functions `get_pairs()` and `merge()` that are the two main functions of the BPE algorithm

## How to use it
