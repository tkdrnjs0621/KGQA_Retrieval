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

def _get_negatives(example,contriever_tokenizer,contriever_model,batch_size,retrieving_depth_1hop,retrieving_depth_2hop,n_negative_1hop, n_negative_2hop):
    n_hops= len(example["gt_path"][0])
    query = example["question"]
    graph_dict = example["graph"]

    texts = []
    original = []
    gt_1hop_triplet = None
    for triplet_a in graph_dict[example["q_entity"][0]]:
        txt = " ".join(triplet_a)
        texts.append(txt)
        original.append(triplet_a)
        if(gt_1hop_triplet==None and triplet_a[1] in [tmp[0] for tmp in example["gt_path"]]):
            gt_1hop_triplet=triplet_a
    
    gt_2hop_triplet = None
    for triplet_a in graph_dict[example[gt_1hop_triplet[2]]]:
        if(gt_2hop_triplet==None and triplet_a[1] in [tmp[1] for tmp in example["gt_path"]]):
            gt_2hop_triplet=triplet_a
        

    tokenizer = contriever_tokenizer 
    contriever = contriever_model 
    contriever.eval()
    query_tokens = tokenizer([query], padding=True, truncation=True, return_tensors='pt').to("cuda")
    output_query = contriever(**query_tokens)
    output_query = mean_pooling(output_query.last_hidden_state, query_tokens['attention_mask'])
    
    def batchify(lst, batch_size_):
        return [lst[i:i+batch_size_] for i in range(0, len(lst), batch_size_)]
    
    batched_message = batchify(texts,batch_size)

    scores=[]
    for batch in batched_message:
        tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
        outputs_batch = contriever(**tokens_batch)
        outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
        temp_scores = output_query.squeeze() @ outputs_batch.T
        scores.extend(temp_scores.tolist())
    
    _, top_k_indices = torch.topk(torch.tensor(scores), retrieving_depth_1hop)

    top_negative_triplets = [original[k] for k in top_k_indices if original[k][1] not in [p[0] for p in example["gt_path"]]]
    assert len(top_negative_triplets)>=n_negative_1hop, "number top negative triplets is less than number of negative 1hop"
    negative_1hop_triplet = top_negative_triplets[:n_negative_1hop]

    if(n_hops==1):
        example["gn_path_triplet"]=negative_1hop_triplet
        example["gt_path_triplet"]=gt_1hop_triplet

        print(example["gn_path_triplet"])
        print(example["gt_path_triplet"])
        assert()
        return example
    
    else:
        texts=[]
        original=[]
        for n in negative_1hop_triplet:
            for triplet_n in graph_dict[n[0]]:
                txt = " ".join(triplet_n)
                texts.append(txt)
                original.append(triplet_n)

        batched_message = batchify(texts,batch_size)
        scores=[]
        for batch in batched_message:
            tokens_batch = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to("cuda:0")
            outputs_batch = contriever(**tokens_batch)
            outputs_batch = mean_pooling(outputs_batch.last_hidden_state, tokens_batch['attention_mask'])
            temp_scores = output_query.squeeze() @ outputs_batch.T
            scores.extend(temp_scores.tolist())
                
        _, top_k_indices = torch.topk(torch.tensor(scores), retrieving_depth_2hop)
        top_negative_triplets = [original[k] for k in top_k_indices if original[k][1] not in [p[1] for p in example["gt_path"]]]
        assert len(top_negative_triplets)>=n_negative_2hop, "number top negative triplets is less than number of negative 2hop"
        negative_2hop_triplet = top_negative_triplets[:n_negative_1hop]

        example["gn_path_triplet"]=negative_1hop_triplet+negative_2hop_triplet
        example["gt_path_triplet"]=gt_1hop_triplet+gt_2hop_triplet

        print(example["gn_path_triplet"])
        print(example["gt_path_triplet"])
        assert()
        return example


def main():
    parser = argparse.ArgumentParser(description="Processing")
    parser.add_argument("--original_dataset", type=str, default="rmanluo/RoG-webqsp", help="directory")
    parser.add_argument("--original_dataset_split", type=str, default="test", help="directory")
    parser.add_argument("--input_merged_dataset_directory", type=str, default="./datasets/Rog-webqsp_test_rr", help="directory")

    args = parser.parse_args()
    dataset_merged = load_from_disk(args.input_merged_dataset_directory)

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever",clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained("facebook/contriever").to('cuda')
    dataset_merged.map(partial(_get_negatives, contriever_tokenizer=tokenizer,contriever_model=model,batch_size=8,retrieving_depth_1hop=15,n_negative_1hop=8,retrieving_depth_2hop=30,n_negative_2hop=16))

if __name__=='__main__':
    main()