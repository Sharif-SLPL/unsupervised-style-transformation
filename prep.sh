mkdir data
mkdir models
mkdir outputs

pip install sentencepiece transformers datasets
pip install -U sentence-transformers
pip install hazm

# Hazm pos tagger
cd models
wget https://github.com/sobhe/hazm/releases/download/v0.5/resources-0.5.zip
unzip resources-0.5.zip
cd ..
# Informal dataset - Not needed (Only need to be trained on formal dataset (In our case, Tapaco & the paraphrases of Tapaco))
# wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3195/lscp-0.5-fa-normalized.7z
# sudo apt-get install p7zip
# 7za e lscp-0.5-fa-normalized.7z