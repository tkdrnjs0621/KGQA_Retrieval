import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from functools import partial
import time
import logging
import os
from torch.utils.data import DataLoader
from datetime import timedelta
import random
import argparse
random.seed(42)

def set_file_handler(logger, path, level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"):
    os.makedirs(os.path.dirname(path + "/run.log"), exist_ok=True)
    handler = logging.FileHandler(path + "/run.log")
    handler.setLevel(level)
    formatter = logging.Formatter(format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
def build_prompt(example, tokenizer):
    prompt = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    return {"prompt": prompt, "prompt_tokens": prompt_tokens}


def collate_fn(batch, tokenizer):
    prompt = [example["prompt"] for example in batch]
    inputs = tokenizer(prompt, add_special_tokens=False, padding=True, return_tensors="pt")
    return inputs


def generate(model, tokenizer, dataloader, logger, log_every, **kwargs):
    start = time.time()
    output_ids = []
    for i, inputs in tqdm(enumerate(dataloader, start=1)):
        inputs = inputs.to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, **kwargs)
        output_ids.extend(outputs[:, inputs["input_ids"].size(1) :].tolist())
        if i % log_every == 0:
            end = time.time()
            elapsed = end - start
            total = elapsed * (len(dataloader) / i)
            logger.info(f"Done {i}/{len(dataloader)} steps - {str(timedelta(seconds=int(elapsed)))}/{str(timedelta(seconds=int(total)))}.")
    return output_ids


def decode(example, tokenizer, feature):
    text = tokenizer.decode(example[feature + "_ids"], skip_special_tokens=True)
    return {feature: text}
def build_messages(example,no_rag,zero_shot,fs_case_1,fs_case_2):

    q = example["question"]
    if(no_rag):
        messages = [
            {"role": "system", "content": "Answer the given question with your knowledge."},
            {"role": "user", "content": f"{q}?"},
        ]
    else:
        def get_knowledge_text(data):
            triplet_list = data["retrieved_triplets"]
            random.shuffle(triplet_list)
            triplet_list = ["("+", ".join(t)+")" for t in triplet_list]
            txt="\n".join(triplet_list)
            return txt
        
        txt = get_knowledge_text(example)
        
        if zero_shot:
            messages = [
            {"role": "system", "content": "Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer the question with these triples and your own knowledge."},
            {"role": "user", "content": f"""[Knowledge Triples]
{txt}
[Question]
{q}?"""}
            ]      
        else:
            txt_1 = get_knowledge_text(fs_case_1)
            q_1 = fs_case_1["question"]
            answer_1 = fs_case_1["a_entity"][0]
            txt_2 = get_knowledge_text(fs_case_2)
            q_2 = fs_case_2["question"]
            answer_2 = fs_case_2["a_entity"][0]

            messages = [
            {"role": "system", "content": "Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer the question with these triples and your own knowledge."},
            {"role": "user", "content": f"""[Knowledge Triples]
{txt_1}

[Question]
{q_1}?"""},
            {"role": "assistant", "content": f"The answer is {answer_1}"},
            {"role": "user", "content": f"""[Knowledge Triples]
{txt_2}

[Question]
{q_2}?"""},
            {"role": "assistant", "content": f"The answer is {answer_2}"},
            {"role": "user", "content": f"""[Knowledge Triples]
{txt}

[Question]
{q}?"""},
            ]
    example["messages"] =messages
    return example

def ifhit(example):
    if  any([answer in example["model_answer"] for answer in example["a_entity"]]):
        example["hit"]=1
    else:
        example["hit"]=0
    return example
def main():
    parser = argparse.ArgumentParser(description="Processing")
    parser.add_argument("--no_rag", action="store_true", help="directory")
    parser.add_argument("--zero_shot", action="store_true", help="directory")
    parser.add_argument("--original_dataset", type=str, default="tkdrnjs0621/webqsp_retrieved_naive_contriver", help="directory")
    parser.add_argument("--original_dataset_split", type=str, default="test", help="directory")
    parser.add_argument("--original_dataset_config", type=str, default="ra_triplets1", help="directory")
    parser.add_argument("--output_directory", type=str, default="./data/", help="directory")
    parser.add_argument("--max_tokens", type=int, default=150, help="generation config; max new tokens")
    parser.add_argument("--do_sample", type=bool, default=True, help="generation config; whether to do sampling, greedy if not set")
    parser.add_argument("--temperature", type=float, default=0.5, help="generation config; temperature")
    parser.add_argument("--top_k", type=int, default=50, help="generation config; top k")
    parser.add_argument("--n_hops", type=int, default=1, help="n_hops")
    parser.add_argument("--top_p", type=float, default=0.5, help="generation config; top p, nucleus sampling")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for inference")
    parser.add_argument("--num_proc", type=int, default=16, help="number of processors for processing datasets")
    parser.add_argument("--log_every", type=int, default=20, help="logging interval in steps")

    args = parser.parse_args()

    dataset = load_dataset(args.original_dataset,args.original_dataset_config,split=args.original_dataset_split)
    case_1 = dataset[0]
    case_2 = dataset[1]
    # dataset = dataset.select(range(2,len(dataset)))
    # dataset_1hop = dataset.filter(lambda x: x["gt_path_triplet"]is not None and len(x["gt_path_triplet"])==1)
    # dataset = dataset.filter(lambda x: x["gt_path"]is not None and len(x["gt_path"])==args.n_hops)

    logger = logging.getLogger("inference")
    logger.setLevel(logging.DEBUG)
    set_file_handler(logger, "./logs")

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token

    dataset=dataset.map(partial(build_messages,no_rag=args.no_rag,zero_shot=args.zero_shot,fs_case_1=case_1,fs_case_2=case_2))
    dataset = dataset.map(partial(build_prompt, tokenizer=tokenizer), num_proc=16)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_proc, collate_fn=partial(collate_fn, tokenizer=tokenizer), pin_memory=True)  # type: ignore
    print("Len data",len(dataset),len(dataloader))

    output_ids = generate(model, tokenizer, dataloader, logger, args.log_every, max_new_tokens=args.max_tokens, do_sample=args.do_sample, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

    dataset = dataset.add_column("model_answer_ids", output_ids)  # type: ignore
    dataset = dataset.map(partial(decode, tokenizer=tokenizer, feature="model_answer"), num_proc=args.num_proc)
    dataset = dataset.map(ifhit)
    # dataset = dataset.select_columns(["question", "q_entity", "a_entity", "retrieved_triplets", "model_answer", "messages","hit"])
    dataset.save_to_disk(args.output_directory)


    print(f"Hit ratio : {sum(dataset['hit'])/len(dataset):.2f}")
    
if __name__ == "__main__":
    main()