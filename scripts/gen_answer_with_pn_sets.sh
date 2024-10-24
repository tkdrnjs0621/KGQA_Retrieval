
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_pn_sets.py --zero_shot --n_negative 0 --output_directory "./data/gen_pn_0"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_pn_sets.py --zero_shot --n_negative 0 --output_directory "./data/gen_pn_1"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_pn_sets.py --zero_shot --n_negative 0 --output_directory "./data/gen_pn_2"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_pn_sets.py --zero_shot --n_negative 0 --output_directory "./data/gen_pn_4"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python gen_answer_with_pn_sets.py --zero_shot --n_negative 0 --output_directory "./data/gen_pn_8"