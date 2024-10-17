import json
import argparse
from datasets import Dataset
from datasets import load_dataset, Dataset, load_from_disk
import torch
from transformers import AutoModel, AutoTokenizer
from functools import partial
def map_triplets(example):

    graph_dict = {}
    for triplet in example["graph"]:
        key = triplet[0]
        value = triplet
        graph_dict.setdefault(key,[]).append(value)
    example["graph_dict"]=graph_dict
    return example

def main():
    parser = argparse.ArgumentParser(description="Processing")
    parser.add_argument("--input_directory", type=str, default="./datasets/AlignData/Rog-webqsp/Rog-webqsp_test.jsonl", help="directory")
    parser.add_argument("--original_dataset", type=str, default="rmanluo/RoG-webqsp", help="directory")
    parser.add_argument("--original_dataset_split", type=str, default="test", help="directory")
    parser.add_argument("--output_directory", type=str, default="./datasets/Rog-webqsp_test_rr", help="directory")

    args = parser.parse_args()
    new_dict={}
    final_dict = {"question":[],"gt_path":[]}
    with open(args.input_directory, 'r') as file:
        for line in file:
            data = json.loads(line)  # Each line is a valid JSON object
            new_dict.setdefault(data["question"],[]).append(data["path"])
    for k, v in new_dict.items():
        final_dict["question"].append(k)
        final_dict["gt_path"].append([list(k) for k in set(tuple(a) for a in v)])
        # print([list(k) for k in set(tuple(a) for a in v)])
    
    dataset_path = Dataset.from_dict(final_dict).to_pandas()
    dataset_original = load_dataset("rmanluo/RoG-webqsp",split="test").to_pandas()
    dataset_merged = Dataset.from_pandas(dataset_original.merge(dataset_path, on="question", how='inner'))
    dataset_triplet_mapped = dataset_merged.map(map_triplets)
    dataset_triplet_mapped.save_to_disk(args.output_directory)

if __name__=='__main__':
    main()