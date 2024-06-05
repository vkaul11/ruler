# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

"""
Create a dataset jsonl file for QA task.

python qa.py \
    --save_dir=./ \
    --save_name=qa \
    --tokenizer_path=tokenizer.model \
    --tokenizer_type=nemo \
    --max_seq_length=4096 \
    --tokens_to_generate=128 \
    --num_samples=10 \
    --template="Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {query} Answer:"
"""
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import random
import numpy as np
import sys
import tiktoken
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")) 


parser = argparse.ArgumentParser()
# Basic Configurations
parser.add_argument("--save_dir", type=Path, required=True, help='dataset folder to save dataset')
parser.add_argument("--save_name", type=str, required=True, help='name of the save dataset jsonl file')
parser.add_argument("--subset", type=str, default='validation', help='Options: validation or test')
parser.add_argument("--max_seq_length", type=int, required=True, help='max sequence length including all input tokens and generated tokens.')
parser.add_argument("--tokens_to_generate", type=int, required=True, help='expected generated token amount.')
parser.add_argument("--num_samples", type=int, required=True, help='number of samples to generate')
parser.add_argument("--pre_samples", type=int, default=0, help='number of samples are already generated')
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--template", type=str, required=True, help='prompt template')
parser.add_argument("--remove_newline_tab", action='store_true', help='remove `\n` and `\t` in all strings.')

# Complexity Configurations
parser.add_argument("--dataset", type=str, required=True, help='dataset file')

# Load Tokenizer by default use tiktoken as it is only used to compute length of synthetic sequence
TOKENIZER = tiktoken.get_encoding("cl100k_base")



# Read SQuAD QA dataset
def read_squad(file):
    with open(file) as f:
        data = json.load(f)
        
    total_docs = [p['context'] for d in data['data'] for p in d['paragraphs']]
    total_docs = sorted(list(set(total_docs)))
    total_docs_dict = {c: idx for idx, c in enumerate(total_docs)}

    total_qas = []
    for d in data['data']:
        more_docs = [total_docs_dict[p['context']] for p in d['paragraphs']]
        for p in d['paragraphs']:
            for qas in p['qas']:
                if not qas['is_impossible']:
                    total_qas.append({
                        'query': qas['question'],
                        'outputs': [a['text'] for a in qas['answers']],
                        'context': [total_docs_dict[p['context']]],
                        'more_context': [idx for idx in more_docs if idx != total_docs_dict[p['context']]]
                    })
                        
    return total_qas, total_docs

# Move this global code later ! 
DOCUMENT_PROMPT = "Document {i}:\n{document}"
QAS, DOCS = read_squad(os.path.join(os.path.dirname(os.path.abspath(__file__)), "json/squad.json"))

def generate_input_output(config: DictConfig,index: int, num_docs: int) -> tuple[str,str]:
    random_seed = config.random_seed 
    template = config.template
    curr_q = QAS[index]['query']
    curr_a = QAS[index]['outputs']
    curr_docs = QAS[index]['context']
    curr_more = QAS[index].get('more_context', [])
    if num_docs < len(DOCS):
        if (num_docs - len(curr_docs)) > len(curr_more):
            addition_docs = [i for i, d in enumerate(DOCS) if i not in curr_docs + curr_more]
            all_docs = curr_docs + curr_more + random.sample(addition_docs, max(0, num_docs - len(curr_docs) - len(curr_more)))
        else:
            all_docs = curr_docs + random.sample(curr_more, num_docs - len(curr_docs))
    
        all_docs = [DOCS[idx] for idx in all_docs]
    else:
        all_docs = DOCS
        
    random.Random(random_seed).shuffle(all_docs)
    
    context = '\n\n'.join([DOCUMENT_PROMPT.format(i=i+1, document=d) for i, d in enumerate(all_docs)])
    input_text = template.format(
        context=context, 
        query=curr_q
    )
    return input_text, curr_a


def _generate_samples(config: DictConfig, incremental: int = 10) -> list:
    
    write_jsons = []
    tokens_to_generate = config.simulation.tokens_to_generate
    max_seq_length = config.simulation.max_seq_length
    num_samples = config.simulation.num_samples 
    remove_newline_tab = config.remove_newline_tab
    
    # Find the perfect num_docs
    num_docs = incremental
    
    total_tokens = 0  # Track the total tokens generated for this example
    while total_tokens + tokens_to_generate < max_seq_length :  
        input_text, answer = generate_input_output(config, 0, num_docs)
        # Calculate the number of tokens in the example
        total_tokens = len(TOKENIZER.encode(input_text + f' {answer}'))
        print(f'Max length {max_seq_length} | Current length {total_tokens + tokens_to_generate} | Docs: {num_docs}')
        if total_tokens + tokens_to_generate > max_seq_length:
            num_docs -= incremental
            break
            
        num_docs += incremental
        if num_docs > len(DOCS):
            num_docs = len(DOCS)
            break
    print('Number of documents:', num_docs)
    
    # Generate samples
    for index in tqdm(range(num_samples)):
        used_docs = num_docs
        while(True):
            try:
                input_text, answer = generate_input_output(config, index, used_docs)
                length = len(TOKENIZER.encode(input_text)) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except:
                if used_docs > incremental:
                    used_docs -= incremental
        
        if remove_newline_tab:
            input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').strip().split())
        
        formatted_output = {
            "index": index,
            "input": input_text,
            "outputs": answer,
            "length": length
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
    write_jsons = _generate_samples(config)
    print(f'{save_file} save_file')
    with open(save_file, 'w', encoding='utf-8') as f:
        for entry in write_jsons:
            json_line = json.dumps(entry)
            f.write(json_line + '\n')

if __name__=="__main__":
    _app()
