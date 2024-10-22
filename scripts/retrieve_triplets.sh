cleanup() {
    echo "Stopping all background processes..."
    kill $(jobs -p)
    wait
    echo "All processes stopped."
}

# Use trap to catch SIGINT (Ctrl+C)
trap cleanup SIGINT

{
echo k1s
CUDA_VISIBLE_DEVICES="1" python src/retrieve_triplets.py --retrieving_depth 1 --output_directory "./data/ra_triplets1"
echo k2s
CUDA_VISIBLE_DEVICES="1" python src/retrieve_triplets.py --retrieving_depth 2 --output_directory "./data/ra_triplets2"
echo k3s
CUDA_VISIBLE_DEVICES="1" python src/retrieve_triplets.py --retrieving_depth 3 --output_directory "./data/ra_triplets3"
echo k4s
CUDA_VISIBLE_DEVICES="1" python src/retrieve_triplets.py --retrieving_depth 4 --output_directory "./data/ra_triplets4"
echo k5s
CUDA_VISIBLE_DEVICES="1" python src/retrieve_triplets.py --retrieving_depth 5 --output_directory "./data/ra_triplets5"
echo k7s
} &

{
CUDA_VISIBLE_DEVICES="0" python src/retrieve_triplets.py --retrieving_depth 7 --output_directory "./data/ra_triplets7"
echo k9s
CUDA_VISIBLE_DEVICES="0" python src/retrieve_triplets.py --retrieving_depth 9 --output_directory "./data/ra_triplets9"
echo k15s
CUDA_VISIBLE_DEVICES="0" python src/retrieve_triplets.py --retrieving_depth 15 --output_directory "./data/ra_triplets15"
echo k20s
CUDA_VISIBLE_DEVICES="0" python src/retrieve_triplets.py --retrieving_depth 20 --output_directory "./data/ra_triplets20"
echo k50s
CUDA_VISIBLE_DEVICES="0" python src/retrieve_triplets.py --retrieving_depth 50 --output_directory "./data/ra_triplets50"
}