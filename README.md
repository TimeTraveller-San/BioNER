# Named Entity Recognition for Medical NEs

## Files
- `combine_data.sh`: Combine all the training datasets into one set of train, devel and test set.
- `dataloader.py`: PyTorch custom dataloader.
- `model.py`: PyTorch models.
- `ner.py`: Get inference results.
- `train.py`: Train the Medical NER model.
- `run.sh`: Run training and a simple inference.
- `utils.py`: Various utility files.
- `data/`: Directory containing all the data.
- `models/`: Directory containing all the trained `*.pth` pytorch models and related data. 

## Notebooks
- EDA: `notebooks/1. EDA.ipynb`, also available as pdf in `1. EDA.pdf`.
- Classical ML models (lightGBM and CRF): `notebooks/2. Classical Models.ipynb`.
- Bi-LSTM development notebook: `notebooks/3. Deep Learning.ipynb`.

## Get NEs
```
python ner.py --f <model_path> --s "Optum is trying the best to help people suffering Angelman syndrome and colorectal, endometrial cancer. Treat with paracetamol"
```

## Train
- Run the following commands
```
bash download.sh #to download data [1]
bash combine_data.sh #to combine the downloaded data
bash run.sh #to start training
```
- By default, the bi-lstm model trains on all the combined data. Run the following to train on specific data:
```
python train.py --train_f <train_data_path> --dev_f <devel_data_path>
```
[1]: The data and the download scripts are borrowed from here: https://github.com/dmis-lab/biobert

## Note
- The models are not optimal, not much effort has been spent on tuning them.
- Number of layers, epochs and other model complexities have been kept to minimal due to computational limits.

## TODO:
- Explore more classical models.
- Other seq2seq models like transformers and fine-tuned BERT or fine tuning modern models like GPT-2.
- Log files.
- Config files for hyperparamters.
- Conduct tests using [checklist](https://www.aclweb.org/anthology/2020.acl-main.442/)