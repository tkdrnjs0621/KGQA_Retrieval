cleanup() {
    echo "Stopping all background processes..."
    kill $(jobs -p)
    wait
    echo "All processes stopped."
}

# Use trap to catch SIGINT (Ctrl+C)
trap cleanup SIGINT



# CUDA_VISIBLE_DEVICES="0" python retrieve_triplets.py --retrieving_depth 1 --output_directory "./data/ra_triplets1" --force_2hop &

# CUDA_VISIBLE_DEVICES="1" python retrieve_triplets.py --retrieving_depth 2 --output_directory "./data/ra_triplets2" --force_2hop &

# CUDA_VISIBLE_DEVICES="2" python retrieve_triplets.py --retrieving_depth 3 --output_directory "./data/ra_triplets3" --force_2hop &

# CUDA_VISIBLE_DEVICES="3" python retrieve_triplets.py --retrieving_depth 4 --output_directory "./data/ra_triplets4" --force_2hop &

# CUDA_VISIBLE_DEVICES="4" python retrieve_triplets.py --retrieving_depth 5 --output_directory "./data/ra_triplets5" --force_2hop &


# CUDA_VISIBLE_DEVICES="5" python retrieve_triplets.py --retrieving_depth 7 --output_directory "./data/ra_triplets7" --force_2hop &

# CUDA_VISIBLE_DEVICES="6" python retrieve_triplets.py --retrieving_depth 9 --output_directory "./data/ra_triplets9" --force_2hop &

# CUDA_VISIBLE_DEVICES="7" python retrieve_triplets.py --retrieving_depth 15 --output_directory "./data/ra_triplets15" --force_2hop &
# # echo k20s
# CUDA_VISIBLE_DEVICES="0" python retrieve_triplets.py --retrieving_depth 20 --output_directory "./data/ra_triplets20" --force_2hop
# echo k50s
CUDA_VISIBLE_DEVICES="3" python retrieve_triplets.py --retrieving_depth 50 --output_directory "./data/ra_triplets50"
