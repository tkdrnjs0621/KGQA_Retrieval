
# echo k1s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 1 --output_directory "./data/ra1"
# echo k2s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 2 --output_directory "./data/ra2"
# echo k3s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 3 --output_directory "./data/ra3"
# echo k4s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 4 --output_directory "./data/ra4"
# echo k5s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 5 --output_directory "./data/ra5"
# echo k7s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 7 --output_directory "./data/ra7"
# echo k9s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 9 --output_directory "./data/ra9"
# echo k15s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 15 --output_directory "./data/ra15"
# echo k20s
# CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 20 --output_directory "./data/ra20"
echo k50s
CUDA_VISIBLE_DEVICES="0" python src/get_retrievable_answers.py --retrieving_depth 50 --output_directory "./data/ra50"
