import sys

from tqdm import tqdm
import os
import json
import fire
import torch
from peft import PeftModel
import transformers
import gradio as gr
from dataset import get_examples

assert (
        "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

def main(
        data_path: str = "MATH_data/Raw",
        load_8bit: bool = True,
        base_model: str = "llama-7b",
        lora_weights: str = "./llama-7b_MATH-ChatGPT",
        batch_size: int = 16,
):
    print('load_8bit', load_8bit)
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
         
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
     
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": 0},
        )
    
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.eos_token_id = 2
    model.config.bos_token_id = 1
    
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
            inputs,
            instruction=None,
            temperature=0.0,
            top_p=0.9,
            top_k=3,
            num_beams=1,
            max_new_tokens=1024,
            **kwargs,
    ):
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            try:
                generation_output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    return_dict_in_generate=False,
                    output_scores=False,
                    max_new_tokens=max_new_tokens,
                    eos_token_id = [model.config.eos_token_id, model.config.pad_token_id],
                    pad_token_id = model.config.pad_token_id
                )
            except:
                return None
        '''
        generation = []
        for i,out in enumerate(generation_output):
            print(out.size(0))
            if out.size(0) < 510:
                generation.append(out)
        '''
        output = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        return output

    all_cate = ["counting_and_probability", "intermediate_algebra", "number_theory", "precalculus", "prealgebra",
                "geometry", "algebra"]
    cate = data_path.split('/')[-1]
    if cate in all_cate:
        cate_kind = cate
    else:
        cate_kind = ''
    with open(os.path.join(lora_weights, f"test_{cate_kind}.jsonl"), 'w') as fh:
        test_examples = get_examples(data_path + '/*', if_eval=True)
        tqdm_total = len(test_examples) // batch_size
        if len(test_examples) % batch_size != 0: tqdm_total += 1
        for i in tqdm(range(0, len(test_examples), batch_size), total=tqdm_total):
            questions = []
            q_batch = []
            a_batch = []
            for k in range(batch_size):
                if (i + k >= len(test_examples)): break

                q = test_examples[i + k]['question']
                q_batch.append(q)
                a = test_examples[i + k]['answer']
                a_batch.append(a)

                questions.append(q)
            inputs = tokenizer(questions, padding=True, return_tensors="pt")
            outputs = evaluate(inputs)
            if outputs:
                for q, a, ans_ in zip(questions, a_batch, outputs):
                    answer = ans_.replace(q, '')
                    sample = {"question": q, "answer": a, "generated_answer":answer}
                    fh.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    fire.Fire(main)
