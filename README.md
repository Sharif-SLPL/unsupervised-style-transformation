# Persian Formality Style Transfer
This project aims to do formality style transfer on Persian language using the T5 transformer architecture.  

### Available Tasks
| Tasks | Description |
| ----------- | ----------- |
| Style Transfer | Convert an informal text/document to a formal style |
| Style Classification | Classify the style of an input text/document |  

### Used Datasets
1. For the informal dataset we have used a dataset of Persian product reviews from Digikala, an Iraninan e-commerce company.
2. For the formal dataset we used the Tapaco dataset which has a paraphrase for every instance that was created by our T5-based paraphraser.

## 1. How to Run
1. First do a `pip install -r requirements.txt`
2. Modify hyperparameters in the `config.py` if needed
3. Run `prep.sh` to install & download required datasets & packages
4. Run main.py
    - **Task:** Pass the task argument for the desired task to be performed. `transfer` for style transfer (Change the style of input text(s) to formal) or `classify` to classify the style of input text(s).
    - **Mode:** Pass the mode argument to train or test the models.
    - **Input:** Pass the input argument for either task in test mode (Can be a single line or path to a file with each sentence in one line). This input is only used for the test mode. To change the input data for training, please see section 2, Custom Datasets.
    ### Example:
    ```
    * Transfering a single input: python main.py transfer test --input 'من این بچه رو دوست دارم'
    * Classifing a whole document: python main.py classify test --input doc.txt
    ```
5. **Note:** You must change the `BASE_CONFIG/local_model_path` in `confing.py` to point to your own directory, Google Drive, etc. 
5. **Note 2:** The output labels of classification task might need to be swapped after each training.  

## 2. Custom Datasets
Put your custom datasets in under the data directory. You could also check the data folder to see example files. To change the path to your custom datasets, please modify values of TRAIN_CONFIG in config.py.
### 2.1 Style Transfer
For the transfer task you would need a text file (paraphrase_data.txt) with each line containing of two comma-separated instances which are the paraphrase of each other. No labels are required for this to work.
### 2.2 Style Classification
For the classification task you would need two text files (informal_data.txt & formal_data.txt) with each line containing only one instance. No labels are required for this to work either.  

## 3. Evaluation
TBI