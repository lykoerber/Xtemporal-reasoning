#!/usr/bin/env python
import accelerate
import logging
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM, BitsAndBytesConfig
import torch


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
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    # inputs.input_ids = inputs.input_ids.to('meta')
    ids = model.generate(inputs, max_length=50, num_return_sequences=3, top_k=50)
    output = [tokenizer.decode(ids[i], skip_special_tokens=True) for i in range(len(ids))]
    print(ids, output)
    logging.info('Generated.')
    return output


if __name__=='__main__':
    logging.basicConfig(filename='log/test.log', format=f'%(levelname)s: %(message)s', level=logging.INFO, filemode='w')
    prompt = "How long did Cannes Film Festival 2019 last?\n"  # ,12 days,Facts
    generate(model, tokenizer, prompt)
    prompt2 = "Pourquoi tu marches pas en fait?\n"
    generate(model, tokenizer, prompt2)
    prompt3 = "Hi! How are you?"
    generate(model, tokenizer, prompt3)
