import json
import argparse
from datasets import Dataset
from datasets import load_dataset, Dataset, load_from_disk
import torch
from transformers import AutoModel, AutoTokenizer
from functools import partial
import copy

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_score(_texts,batch_size__,tokenizer,contriever,output_query):
    def batchify(lst, batch_size_):
        return [lst[i:i+batch_size_] for i in range(0, len(lst), batch_size_)]
    
    batched_message = batchify(_texts,batch_size__)

    scores=[]
    for batch in batched_message:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriever(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())

    return scores

def _get_retrieved(example,contriever_tokenizer,contriever_model,batch_size,retrieving_depth_1hop,retrieving_depth_2hop,force_2hop):
    n_hops= len(example["gt_path"][0])
    
    query = example["question"]
    graph_dict = {}
    for triplet in example["graph"]:
        key = triplet[0]
        value = triplet
        graph_dict.setdefault(key,[]).append(value)

    tokenizer = contriever_tokenizer 
    contriever = contriever_model 
    contriever.eval()
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    output_query = contriever(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])

    texts_1hop = []
    original = []
    for triplet_a in graph_dict[example["q_entity"][0]]:
        txt = " ".join(triplet_a)
        texts_1hop.append(txt)
        original.append(triplet_a)
    
    scores = get_score(texts_1hop,batch_size,tokenizer,contriever,output_query)

    if(len(scores)<retrieving_depth_1hop):
        top_triplets = original
    else:
        _, top_k_indices = torch.topk(torch.tensor(scores), retrieving_depth_1hop)
        top_triplets = [original[k] for k in top_k_indices]
    
    top_triplets_1hop=copy.deepcopy(top_triplets)
    if(n_hops==1 and not force_2hop):
        top_triplets_te = [t[2] for t in top_triplets]
        example["is_gt_retrieved"] = 1 if any((ea in top_triplets_te) for ea in example["a_entity"]) else 0
        example["retrieved_triplets"] = top_triplets_1hop
        return example
    
    else:
        texts_2hop=[]
        original=[]
        for n in top_triplets:
            if n[2]==n[0] or n[2] not in graph_dict: # remove self-edge and invalidity
                continue
            for triplet_n in graph_dict[n[2]]:
                txt = " ".join(triplet_n)
                texts_2hop.append(txt)
                original.append(triplet_n)

        scores = get_score(texts_2hop,batch_size,tokenizer,contriever,output_query)
                    
        if(len(scores)<retrieving_depth_2hop):
            top_triplets = original# [i for i in original if i[1] not in [p[1] for p in example["gt_path"] if len(p)>0]]
        else:
            _, top_k_indices = torch.topk(torch.tensor(scores), retrieving_depth_2hop)
            top_triplets = [original[k] for k in top_k_indices]
        
        top_triplets_te = [t[2] for t in top_triplets]+[t[2] for t in top_triplets_1hop] if force_2hop else [t[2] for t in top_triplets]
        example["is_gt_retrieved"] = 1 if any((ea in top_triplets_te) for ea in example["a_entity"]) else 0
        example["retrieved_triplets"] = top_triplets+top_triplets_1hop
        return example
        

def main():
    parser = argparse.ArgumentParser(description="Processing")
    parser.add_argument("--original_dataset", type=str, default="tkdrnjs0621/webqsp_gt_gn_paths_revised", help="directory")
    parser.add_argument("--original_dataset_split", type=str, default="test", help="directory")
    parser.add_argument("--output_directory", type=str, default="./data/", help="directory")
    parser.add_argument("--force_2hop", action="store_true", help="directory")
    parser.add_argument("--batch_size", type=int, default=8, help="number of processors for processing datasets")
    parser.add_argument("--retrieving_depth", type=int, default=8, help="number of processors for processing datasets")

    args = parser.parse_args()
    
    dataset = load_dataset(args.original_dataset,split=args.original_dataset_split)

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever",clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained("facebook/contriever").to('cuda')

    dataset_check_retrieved = dataset.map(partial(_get_retrieved, contriever_tokenizer=tokenizer,contriever_model=model,batch_size=args.batch_size,retrieving_depth_1hop=args.retrieving_depth,retrieving_depth_2hop=args.retrieving_depth,force_2hop=args.force_2hop))
    dataset_check_retrieved = dataset_check_retrieved.select_columns(["question", "q_entity", "a_entity", "gt_path_triplet", "gn_path_triplet", "is_gt_retrieved","retrieved_triplets","n_hops"])
    dataset_check_retrieved.save_to_disk(args.output_directory)
    hr1 = (sum(dataset_check_retrieved.filter(lambda x: x["n_hops"]==1)["is_gt_retrieved"])/len(dataset_check_retrieved))
    hr2 =  (sum(dataset_check_retrieved.filter(lambda x: x["n_hops"]==2)["is_gt_retrieved"])/len(dataset_check_retrieved))
    hrt=(sum(dataset_check_retrieved["is_gt_retrieved"])/len(dataset_check_retrieved))
    print(f"Hit ratio Total/1hop/2hop: {hrt:.2f}/{hr1:.2f}/{hr2:.2f}")

if __name__=='__main__':
    main()