#!/usr/bin/env python
import accelerate
import logging
import pandas as pd
from transformers import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    BitsAndBytesConfig,
    set_seed
    )
import torch

from prompting import read_data, few_shot, create_prompt


# random seed for reproducibility
set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# Model names: "chrisyuan45/TimeLlama-7b-chat", "chrisyuan45/TimeLlama-13b-chat"
model_name = "chrisyuan45/TimeLlama-7b-chat"
quantization_config = BitsAndBytesConfig.from_dict({
    'load_in_4bit': True,
    'bnb_4bit_compute_dtype': torch.float16,
    'bnb_4bit_quant_type': 'nf4',
    'bnb_4bit_use_double_quant':True})

model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        #load_in_8bit=True,
        quantization_config = quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True)
logging.info(f'Model {model} loaded.')
tokenizer = LlamaTokenizer.from_pretrained(model_name)
logging.info('Tokenizer loaded.')

def generate(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    ids = model.generate(input_ids,
                        max_new_tokens=100,
                        num_return_sequences=5,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=10,
                        repetition_penalty=1.0,
                        length_penalty=1,
                        no_repeat_ngram_size=2)
    output = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(len(ids))]
    logging.info('Generated.')
    return output

def run_dataset(dir, prompting="5-shot", output_file='outputs/ordering_mcq.json'):
    ds = read_data(dir)
    outputs = []
    for d in ds.iterrows():
        prompt = create_prompt(d, shots=5, shot_dir='data/ordering/ordering_shots_mcq.csv')
        output = generate(model, tokenizer, prompt)
        output_wo_prompt = [o.replace(prompt, '') for o in output]
        outputs.append(output_wo_prompt)
        if d[0] == 50:
            break
    output_df = pd.DataFrame(outputs, columns=[f'g{i}' for i in range(len(outputs[0]))])
    output_df.to_json(output_file)#, encoding='utf-8')


if __name__=='__main__':
    logging.basicConfig(filename='../log/test.log', format=f'%(levelname)s: %(message)s', level=logging.INFO, filemode='w')
    run_dataset('data/ordering/ordering_mcq.csv')
