import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.stats import pearsonr
from scipy.stats import spearmanr

# Why katie is wrong and im right (maybe havent read the paper):
# https://arxiv.org/pdf/2109.07173.pdf


# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
model.to(device)



# Create latent representation for text
def code_embedding(function):
    tokens = tokenizer.tokenize(function)
    tokens=[tokenizer.cls_token] + tokens + [tokenizer.eos_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]

    # return context_embeddings
    return context_embeddings[0][0]

def flatten_arrays(a, b):
    a = torch.flatten(a).detach().numpy()
    b = torch.flatten(b).detach().numpy()

    if len(a) > len(b):
        b = np.pad(b, (0, len(a)-len(b)))
    elif len(b) > len(a):
        a = np.pad(a, (0, len(b)-len(a)))

    return a, b

# Calculate cosine similarity
def cosine_similarity(a, b):
    # First ensure both embeddings have same dimensionality
    a, b = flatten_arrays(a, b)
    similarity = (np.dot(a, b)) / (np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2)))
    print(similarity)
    
    return similarity

# Calculate pearson correlation coefficient
def pearson_similarity(a, b):
    # First ensure both embeddings have same dimensionality
    a, b = flatten_arrays(a, b)
    corr, pval = pearsonr(a, b)
    print(corr)
    
    return corr, pval

# Calculate spearman rank correlation 
def spearman_similarity(a, b):
    # First ensure both embeddings have same dimensionality
    a, b = flatten_arrays(a, b)
    corr, pval = spearmanr(a, b)
    print(corr)

    return corr, pval

# Calculate similarity scores
def average_similarity(a, b):
    cosine_score = cosine_similarity(a, b)
    pearson_score, _ = pearson_similarity(a, b) 
    spearman_score, _ = spearman_similarity(a, b)
    print((cosine_score + pearson_score + spearman_score) / 3)
    
    return (cosine_score + pearson_score + spearman_score) / 3


# Helper function to replace token with ''
def empty_mask(function, degree=5):
    tokens = tokenizer.tokenize(function)
    num_tokens = len(tokens)
    mask_idx = np.random.randint(0, num_tokens, size=degree)

    for mask_id in mask_idx:
        tokens[mask_id] = ''
    
    return tokens

# Helper function to remove token from token_list
def remove_mask(function, degree=5):
    tokens = tokenizer.tokenize(function)
    num_tokens = len(tokens)
    mask_idx = np.random.randint(0, num_tokens, size=degree)

    for mask_id in mask_idx:
        tokens.pop(mask_id)
    
    return tokens

# Helper function to add '' to token_list
def add_mask(function, degree=5):
    tokens = tokenizer.tokenize(function)
    num_tokens = len(tokens)
    mask_idx = np.random.randint(0, num_tokens, size=degree)

    counter = 0
    for mask_id in mask_idx:
        mask_id += counter
        tokens = tokens[:mask_id] + [''] + tokens[mask_id:]
        counter += 1
    
    return tokens



def main(input_function, samples=100):
    similarity_score = 0
    for _ in range(samples):
        # Mask input function
        empty_masked_input = empty_mask(input_function)
        rand_masked_input = random_mask(input_function)

        # # TODO: Get output function from CodeGen
        # output1 = codegen(input_function)
        # output2 = codegen(empty_masked_input)
        # output3 = codegen(rand_masked_input)

        # Embed output function
        python1 = code_embedding(output1)
        python2 = code_embedding(output2)
        python3 = code_embedding(output3)
        python1, python2 = flatten_arrays(python1, python2)

        # Get similarity score
        similarity_score = average_similarity(python1, python2)



# Python
python_function1 = '''
def my_func():
    statement = "Hello World"
    print(statement)
'''
python_function2 = '''
public static void 

def life():
    a = 1 + 1
    b = 1 + 1
    c = a + b
    return c

def death():
    a = 1 + 1
    b = 1 + 1
    c = a + b
    return c
'''
python_function3 = '''
def mod_func(self):
    output = "Hello Beautiful World"
    print(output)
'''

# Test mask functions
masked_1 = empty_mask(python_function1)
masked_2 = random_mask(python_function1)
# print(masked_1, masked_2)
# Test embedding functions
python1 = code_embedding(python_function1)
python2 = code_embedding(python_function2)
# Test similarity function
similarity_score = average_similarity(python1, python2)