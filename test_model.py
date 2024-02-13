# import accelerate?
from transformers import LlamaConfig, LlamaTokenizer, LlamaForCausalLM


# Model names: "chrisyuan45/TimeLlama-7b-chat", "chrisyuan45/TimeLlama-13b-chat"
model_name = "chrisyuan45/TimeLlama-7b-chat"
model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained(model_name)


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    # inputs.input_ids = inputs.input_ids.to('meta')
    ids = model.generate(inputs.input_ids, max_length=30)
    output = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(ids, output)
    return output


if __name__=='__main__':
    prompt = "How long did Cannes Film Festival 2019 last?"  # ,12 days,Facts
    generate(model, tokenizer, prompt)