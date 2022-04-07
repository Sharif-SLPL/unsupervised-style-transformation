BASE_CONFIG = {
    "model_path": "erfan226/persian-t5-paraphraser",
    "local_model_path": "/gd2/MyDrive/models/persian-t5-style-paraphraser" # You must change this to your own directory
}
TRAIN_CONFIG = {
    "paraphrase_dataset_path": "data/paraphrase_data.txt", # For the transfer task
    "formal_dataset_path": "data/formal_data.txt", # For the classification task
    "informal_dataset_path": "data/informal_data.txt", # For the classification task
    "dataset_limit": 5000,
}
GENERATION_CONFIG = {
    "text_similarity": 0.8,
    "num_beams": 5,
    "do_sample": True,
    "temperature": 0.9,
    "top_p": 0.6,
    "top_k": 50,
}