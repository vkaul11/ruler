"""
This file contains code that is derived from the RULER project by Jackson Hsieh,
available at https://github.com/hsiehjackson/RULER, which is licensed under the Apache License 2.0.

Modifications were made to simply and customize it for our use cases 
"""


"""
Create a dataset jsonl file for needle in a haystack.
"""
import os
import re
import json
import uuid
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import wonderwords
import sys
import tiktoken
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize
# Load Tokenizer by default use tiktoken as it is only used to compute length of synthetic sequence
TOKENIZER = tiktoken.get_encoding("cl100k_base")
# Words
nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
# verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))
# Positions (Inserting the needle between 25th and 75th percentile of the text to make it lost in the middle problem)
DEPTHS = list(np.round(np.linspace(25, 75, num=40, endpoint=True)).astype(int))


def _generate_random_number(num_digits: int = 7) -> str:
    lower_bound = 10**(num_digits - 1)
    upper_bound = 10**num_digits - 1
    return str(random.randint(lower_bound, upper_bound))

def _generate_random_word() -> str:
    word = random.choice(words)
    return word

def _generate_random_uuid() -> str:
    return str(uuid.UUID(int=random.getrandbits(128), version=4))

def _generate_random(type_needle: str) -> str:
    if type_needle == 'numbers':
        return _generate_random_number()
    elif type_needle == 'words':
        return _generate_random_word()
    elif type_needle == 'uuids':
        return _generate_random_uuid()
    else:
        raise NotImplementedError(f'{type_needle} is not implemented.')

def _generate_input_output(config: DictConfig,num_haystack: int,haystack: list[str]) -> tuple[str, list[str]]:
    keys, values, needles = [], [], []
    # extracting the data required from the config files 
    type_needle_k = config.task_complexity.type_needle_k
    type_needle_v = config.task_complexity.type_needle_v
    num_needle_k = config.task_complexity.num_needle_k
    num_needle_v = config.task_complexity.num_needle_v
    num_needle_q = config.task_complexity.num_needle_q
    random_seed = config.random_seed 
    type_haystack = config.task_complexity.type_haystack
    template = config.template 
   

    for _ in range(num_needle_k):
        keys.append(_generate_random(type_needle_k))
        value = []
        needle = "One of the special magic {type_needle_v} for {key} is: {value}."
        for _ in range(num_needle_v):
            value.append(_generate_random(type_needle_v))
            needles.append(needle.format(
                type_needle_v=type_needle_v,
                key=keys[-1], 
                value=value[-1],
            ))
        values.append(value)
    
    random.Random(random_seed).shuffle(needles)
    
    # Context
    if  type_haystack == 'essay':
        text = " ".join(haystack[:num_haystack])
        document_sents = sent_tokenize(text.strip())
        insertion_positions = [0] + \
                              sorted([int(len(document_sents) * (depth / 100)) for depth in random.sample(DEPTHS, len(needles))]) + \
                              [len(document_sents)]
        document_sents_list = []
        for i in range(1,len(insertion_positions)):
            last_pos = insertion_positions[i-1]
            next_pos = insertion_positions[i]
            document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
            if i-1 < len(needles):
                document_sents_list.append(needles[i-1])
        context = " ".join(document_sents_list)

    else:
        if type_haystack == 'repeat':
            sentences = [haystack] * num_haystack
        elif type_haystack == 'needle':
            sentences = [haystack.format(
                type_needle_v=type_needle_v,
                key=_generate_random(type_needle_k),
                value=_generate_random(type_needle_v),
            ) for _ in range(num_haystack)]

            
        indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
        for index, element in zip(indexes, needles):
            sentences.insert(index, element)
        context = "\n".join(sentences)


    ## Query and Answer
    indices = random.sample(range(num_needle_k), num_needle_q)
    queries = [keys[i] for i in indices]
    answers = [a for i in indices for a in values[i]]
    query = ', '.join(queries[:-1]) + ', and ' + queries[-1] if len(queries) > 1 else queries[0]
    
   
    if num_needle_q * num_needle_v == 1:
        template = template.replace('Some', 'A')
        template = template.replace('are all', 'is')
        template = template.replace('are', 'is')
        template = template.replace('answers', 'answer')
        type_needle_v = type_needle_v[:-1] # remove "s"

    input_text = template.format(
        type_needle_v=type_needle_v,
        context=context,
        query=query,
    )

    return input_text, answers


def _generate_samples(config: DictConfig) -> list[dict[str, any]]:
    write_jsons = []
    tokens_to_generate = config.simulation.tokens_to_generate
    max_seq_length = config.simulation.max_seq_length
    num_samples = config.simulation.num_samples
    type_haystack = config.task_complexity.type_haystack
    remove_newline_tab = config.remove_newline_tab
   

    if type_haystack == 'essay':
        incremental = 500
    elif type_haystack == 'repeat':
        incremental = 25
    elif type_haystack == 'needle':
        incremental = 25
        
    if type_haystack != 'essay' and max_seq_length < 4096:
        incremental = 5

    # Define Needle/Haystack Format
    needle = "One of the special magic {type_needle_v} for {key} is: {value}."
    if config.task_complexity.type_haystack == 'essay':
        essay_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/PaulGrahamEssays.json")
        with open(essay_path, 'r') as file:
            essay = json.load(file)['text']
        haystack = re.sub(r'\s+', " ", essay).split(" ")
    elif config.task_complexity.type_haystack == 'repeat':
        haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    elif config.task_complexity.type_haystack == 'needle':
        haystack = needle
    else:
        raise NotImplementedError(f"{config.task_complexity.type_haystack} is not implemented.")

    #Number of times haystack has to be increased each time to get to max sequence length
    num_haystack = incremental
        
    total_tokens = 0  # Track the total tokens generated for the first example
    while total_tokens + tokens_to_generate < max_seq_length :  
        input_text, answer = _generate_input_output(config,num_haystack,haystack)
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER.encode(input_text + ' '.join(answer)))
        print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Haystack: {num_haystack}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_haystack -= incremental
            break
    
        if type_haystack == 'essay' and num_haystack > len(haystack):
            num_haystack = len(haystack)
            break
        
        num_haystack += incremental

    print('Num haystack:', num_haystack)
    
    # Generate samples
    for index in tqdm(range(num_samples)):
        used_haystack = num_haystack
        while(True):
            try:
                input_text, answer  = _generate_input_output(config,used_haystack,haystack)
                length = len(TOKENIZER.encode(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_haystack > incremental:
                    used_haystack -= incremental
        
        if remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())

        formatted_output = {
            'index': index,
            "input": input_text,
            "outputs": answer,
            "length": length,
        }
        write_jsons.append(formatted_output)

    return write_jsons

@hydra.main(config_path="conf", config_name="config", version_base=None)
def _app(config: DictConfig):
    save_dir = Path(config.save_dir)
    save_file = save_dir / f'{config.save_name}' / f'{config.subset}.jsonl'
    save_file.parent.mkdir(parents=True, exist_ok=True)
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    # Ensure num_needle_k is properly set
    config.task_complexity.num_needle_k = max(config.task_complexity.num_needle_k, config.task_complexity.num_needle_q)
    print("Configuration:", OmegaConf.to_yaml(config))
    write_jsons = _generate_samples(config)
    print(f'{save_file} save_file')
    with open(save_file, 'w', encoding='utf-8') as f:
        for entry in write_jsons:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

if __name__ == "__main__":
    _app()
