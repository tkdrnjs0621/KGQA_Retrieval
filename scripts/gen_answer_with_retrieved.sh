
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets1 --output_directory "./data/gen_ra_triplets1"
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets2 --output_directory "./data/gen_ra_triplets2"
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets3 --output_directory "./data/gen_ra_triplets3"
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets4 --output_directory "./data/gen_ra_triplets4"
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets5 --output_directory "./data/gen_ra_triplets5"
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets7 --output_directory "./data/gen_ra_triplets7"
# CUDA_VISIBLE_DEVICES="0,1,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets9 --output_directory "./data/gen_ra_triplets9"
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets15 --output_directory "./data/gen_ra_triplets15" --batch_size 16
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets20 --output_directory "./data/gen_ra_triplets20" --batch_size 8
# CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_retrieved.py --zero_shot --original_dataset_config ra_triplets50 --output_directory "./data/gen_ra_triplets50" --batch_size 4
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_retrieved.py --no_rag --original_dataset_config ra_triplets50 --output_directory "./data/gen_ra_triplets50" --batch_size 32