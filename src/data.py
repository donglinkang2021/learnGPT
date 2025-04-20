
def get_data(data_path:str = 'data/input.txt'):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def get_tokenizer(text:str):
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    return vocab_size, encode, decode

