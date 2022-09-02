# Paraphrase & Style Transfer model path
BASE_CONFIG = {
    "st_seed": 1,
    "para_model_path": "erfan226/persian-t5-paraphraser", # load the base model if you want to fine-tune the model with your own data -> Ahmad/parsT5-base
    "local_para_model_path": "/gd/MyDrive/models/persian-t5-paraphraser", # Direcotry to save the trained/downloaded (outside of HuggingFace) model. You must change this to your own directory.
    "st_model_path": "erfan226/persian-t5-paraphraser",
    "local_st_model_path": "/gd2/MyDrive/models/persian-t5-style-paraphraser" # Direcotry to save the trained/downloaded (outside of HuggingFace) model. You must change this to your own directory.
}
TRAIN_CONFIG = {
    "paraphrase_dataset_path": "data/paraphrase_data.txt", # For the training of the transfer model
    "formal_dataset_path": "data/formal_data.txt", # For training of the classification model
    "informal_dataset_path": "data/informal_data.txt", # For training of the classification model
    "dataset_limit": -1, # Use all of available data. Limit it if needed
    "loss_threshold": 0.1,
    "learning_rate": None,
    "warmup_init": True
}
GENERATION_CONFIG = {
    "text_similarity": 0.8,
    "num_beams": 5,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.92,
    "top_k": 30,
}