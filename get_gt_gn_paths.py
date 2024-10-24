import json
import argparse
from datasets import Dataset
from datasets import load_dataset, Dataset, load_from_disk
import torch
from transformers import AutoModel, AutoTokenizer
from functools import partial

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

def _get_negatives(example,contriever_tokenizer,contriever_model,batch_size,retrieving_depth_1hop,retrieving_depth_2hop,n_negative_1hop, n_negative_2hop):
    n_hops= len(example["gt_path"][0])
    if(n_hops==0):
        example["gn_path_triplet"]=None
        example["gt_path_triplet"]=None
        example["n_hops"]=n_hops
        return example
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
    gt_1hop_triplet = []
    for triplet_a in graph_dict[example["q_entity"][0]]:
        txt = " ".join(triplet_a)
        texts_1hop.append(txt)
        original.append(triplet_a)
        if(triplet_a[1] in [tmp[0] for tmp in example["gt_path"] if len(tmp)>0]):
            gt_1hop_triplet.append(triplet_a)
    
    if(len(gt_1hop_triplet)==0):
        example["gn_path_triplet"]=None
        example["gt_path_triplet"]=None
        example["n_hops"]=n_hops
        return example
    
    scores = get_score(texts_1hop,batch_size,tokenizer,contriever,output_query)
    if(len(scores)<retrieving_depth_1hop):
        # remove all gt
        top_negative_triplets = [i for i in original if i[1] not in [p[0] for p in example["gt_path"] if len(p)>0]]
    else:
        _, top_k_indices = torch.topk(torch.tensor(scores), retrieving_depth_1hop)
        # remove all gt
        top_negative_triplets = [original[k] for k in top_k_indices if  original[k][1] not in [p[0] for p in example["gt_path"] if len(p)>0]] 
    
    if(len(top_negative_triplets)>=n_negative_1hop):
        negative_1hop_triplet = top_negative_triplets[:n_negative_1hop]
    else:
        example["gn_path_triplet"]=None
        example["gt_path_triplet"]=None
        example["n_hops"]=n_hops
        return example

    if(n_hops==1):
        example["gn_path_triplet"]=negative_1hop_triplet
        example["gt_path_triplet"]=[gt_1hop_triplet[0]] # Choose the first GT triplet
        example["n_hops"]=n_hops
        return example
    
    else:
        
        gt_1hop_triplet_chosen=None
        for gt_1hop_triplet_candidate in gt_1hop_triplet:
            if(gt_1hop_triplet_candidate[2] in graph_dict):
                gt_1hop_triplet_chosen = gt_1hop_triplet_candidate # Choose the first valid gt 1hop
                break
        
        if(gt_1hop_triplet_chosen==None): # if 2hop for valid gt 1hop does not exists
            example["gn_path_triplet"]=None
            example["gt_path_triplet"]=None
            example["n_hops"]=n_hops
            return example
        
        gt_2hop_triplet = []
        for triplet_a in graph_dict[gt_1hop_triplet_chosen[2]]: #2-hop triplets for chosen 1hop
            if(triplet_a[1] in [tmp[1] for tmp in example["gt_path"] if len(tmp)>1]):
                gt_2hop_triplet.append(triplet_a)

        if(len(gt_2hop_triplet)==0):
            example["gn_path_triplet"]=None
            example["gt_path_triplet"]=None
            example["n_hops"]=n_hops
            return example
        
        texts_2hop=[]
        original=[]
        for n in negative_1hop_triplet:
            if n[2]==n[0] or n[2] not in graph_dict: # remove self-edge and invalidity
                continue
            for triplet_n in graph_dict[n[2]]:
                if(triplet_n[2]!=gt_1hop_triplet_chosen[0]): #remove back edge
                    txt = " ".join(triplet_n)
                    texts_2hop.append(txt)
                    original.append(triplet_n)

        scores = get_score(texts_2hop,batch_size,tokenizer,contriever,output_query)
                    
        if(len(scores)<retrieving_depth_2hop):
            #remove gt
            top_negative_triplets = [i for i in original if i[1] not in [p[1] for p in example["gt_path"] if len(p)>1]]
        else:
            _, top_k_indices = torch.topk(torch.tensor(scores), retrieving_depth_2hop)
            #remove gt
            top_negative_triplets = [original[k] for k in top_k_indices if original[k][1] not in [p[1] for p in example["gt_path"] if len(p)>1]]
     
        if (len(top_negative_triplets)>=n_negative_2hop):
            negative_2hop_triplet = top_negative_triplets[:n_negative_2hop]
        else:
            example["gn_path_triplet"]=None
            example["gt_path_triplet"]=None
            example["n_hops"]=n_hops
            return example
        
        example["gn_path_triplet"]=negative_1hop_triplet+negative_2hop_triplet
        example["gt_path_triplet"]=[gt_1hop_triplet_chosen,gt_2hop_triplet[0]] # choose the first gt 2hop triplet
        example["n_hops"]=n_hops
        
        return example

def map_gt_path(example, dictionary):
    example["gt_path"]=dictionary[example["question"]] if example["question"] in dictionary else None
    return example

def main():
    parser = argparse.ArgumentParser(description="Processing")
    parser.add_argument("--input_directory", type=str, default="./datasets/AlignData/Rog-webqsp/Rog-webqsp_test.jsonl", help="directory")
    parser.add_argument("--original_dataset", type=str, default="rmanluo/RoG-webqsp", help="directory")
    parser.add_argument("--original_dataset_split", type=str, default="test", help="directory")
    parser.add_argument("--output_directory", type=str, default="./datasets/Rog-webqsp_test_rr", help="directory")

    args = parser.parse_args()
    path_dict={}
    prev_lines=[]
    with open(args.input_directory, 'r') as file:
        for line in file:
            if line not in prev_lines:
                data = json.loads(line)
                prev_lines.append(line)
                path_dict.setdefault(data["question"],[]).append(data["path"])
                
    dataset_original = load_dataset(args.original_dataset,split=args.original_dataset_split)
    dataset_merged = dataset_original.map(partial(map_gt_path,dictionary=path_dict)).filter(lambda x:x["gt_path"] is not None)
    # print(dataset_merged[1]["gt_path"])
    # assert()
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever",clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained("facebook/contriever").to('cuda')

    dataset_merged = dataset_merged.map(partial(_get_negatives, contriever_tokenizer=tokenizer,contriever_model=model,batch_size=8,retrieving_depth_1hop=15,n_negative_1hop=8,retrieving_depth_2hop=15,n_negative_2hop=8))
    dataset_merged=dataset_merged.filter(lambda x: x["gt_path_triplet"] is not None)
    dataset_merged.save_to_disk(args.output_directory)
    dataset_merged.push_to_hub("tkdrnjs0621/webqsp_gt_gn_paths_revised",split="test")

if __name__=='__main__':
    main()