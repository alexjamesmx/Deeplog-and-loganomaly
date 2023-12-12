#Generate embeddings first
python -u ./dataset/generate_embeddings.py oct1-2 average 
python -u main.py --config_file=config/loganomaly.yaml
