import random
from datasets import load_dataset

def greet_user():
    """Function to greet the user."""
    responses = ["Hello!", "Hi!", "Hey there!", "Greetings!"]

    return random.choice(responses)

def farewell_user():
    """Function to say goodbye to the user."""
    responses = ["Goodbye!", "Farewell!", "See you later!", "Take care!"]
    return random.choice(responses)

def chatbot_response(user_input):
    """Function to generate a response from the chatbot based on user input."""
    user_input = user_input.lower()
    if 'hello' in user_input or 'hi' in user_input or 'hey' in user_input:
        return greet_user()
    elif 'bye' in user_input or 'goodbye' in user_input or 'see you' in user_input:
        return farewell_user()
    else:
        return "I'm sorry, I don't understand that. Can you please rephrase?"

def main():
    print("Chatbot: Hello! I am your friendly chatbot. How can I assist you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)
        print()

def add(a, b):
    c = a + b
    return c

def subtract(a, b):
    c = a - b
    return c

def multiply(a, b):
    c = a * b
    return c

def divide(a, b):
    c = a / b
    return c

def power(a, b):
    c = a ** b
    return c

def factorial(a):
    c = 1
    for i in range(1, a + 1):
        c = c * i
    return c
    

if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoTokenizer, AutoModel
    from scipy.stats import pearsonr
    from scipy.stats import spearmanr
    import matplotlib
    from fastparquet import ParquetFile

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.to(device)


    # Create latent representation for text
    def code_embedding(function):
        tokens = tokenizer.tokenize(function)
        tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
        context_embeddings = model(torch.tensor(tokens_ids)[None,:])[0]
        print(context_embeddings.shape)

        return context_embeddings

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
        nominator = np.dot(a, b)
        denominator = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
        similarity = nominator / denominator
        print(similarity)
        
        return similarity

    # Calculate pearson correlation coefficient
    def pearson_similarity(A, B):
        corr, pval = pearsonr(A, B)
        print(corr)
        
        return corr, pval

    # Calculate spearman rank correlation 
    def spearman_similarity(a, b):
        corr, pval = spearmanr(a, b)
        print(corr)

        return corr, pval


    # Python
    python_function1 = '''
    def my_func():
        statement = "Hello World"
        print(statement)
    '''
    python_function2 = '''
    def my_func():
        print("Hello World")
    '''
    python_function3 = '''
    def my_func():
        print("Hello World")
    '''
    # # Generate text embeddings
    # python1 = code_embedding(python_function1)
    # python2 = code_embedding(python_function2)
    # python1, python2 = flatten_arrays(python1, python2)

    # # Calculate similarity scores
    # cosine_similarity = cosine_similarity(python1, python2)
    # pearson_coefficient, _ = pearson_similarity(python1, python2) 
    # spearman_rank, _ = spearman_similarity(python1, python2)

    # df = pd.read_parquet('snip.parquet')
    # df.to_csv('out.csv')
    pf = ParquetFile('snip.parquet')
    # print(pf.columns)
    df = pf.to_pandas(columns=['content'])
    print(df.shape)
    print(df.head(66)[35:56])
    with open('out.txt', 'a') as f:
        df_string = df.head(66)[35:56].to_string(header=False, index=False)
        f.write(df_string)

    string = '''    
    def parse (self) :
        root  = self.tree.getroot ()
        self.wlans  = []
        self.routes = {}
        for div in root.findall ("".//%s"" % tag (""div"")) :
            id = div.get ('id')
            if id == 'cbi-wireless' :
                wlan_div = div
            elif id == 'cbi-routes' :
                route_div = div
            self.try_get_version (div)
        for d in self.tbl_iter (wlan_div) :
            for k, newkey in pyk.iteritems (self.wl_names) :
                if k in d :
                    d [newkey] = d [k]
            wl = WLAN_Config (** d)
            self.wlans.append (wl)
        for d in self.tbl_iter (route_div) :
            iface = d.get ('iface')
            gw    = d.get ('gateway')
            if iface and gw :
                self.routes [iface] = gw
        self.set_version (root)
    # end def parse'''

    # python1 = code_embedding(string)
    # test2 = ''
    # with open('out.txt', 'r') as f:
    #     for line in f[32:56]:
    #         test2 += line
    # python2 = code_embedding(test2)
    # python1, python2 = flatten_arrays(python1, python2)

    # # Calculate similarity scores
    # cosine_similarity = cosine_similarity(python1, python2)
    # pearson_coefficient, _ = pearson_similarity(python1, python2) 
    # spearman_rank, _ = spearman_similarity(python1, python2)

    # print(f'cos: {cosine_similarity}, pear: {pearson_coefficient}, spear: {spearman_rank}')

