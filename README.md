# Cooperative Generator-Discriminator Networks 

This repository contains the code used in the paper:

Discourse Understanding and Factual Consistency in Abstractive Summarization. Saadia Gabriel, Antoine Bosselut, Jeff Da, Ari Holtzman, Jan Buys, Kyle Lo, Asli Celikyilmaz and Yejin Choi. EACL 2021. 

The repo contains 

1. A cooperative generator-discriminator summarization framework for generating abstracts of scientific papers with various discourse and factuality objectives; and 
2. A large-scale dataset of ArXiv scientific papers covering a broad range of domains 

[Dataset](https://drive.google.com/drive/u/0/folders/1VEBEuH3sJKZErt_9UF6bIrgag_ws6GXC) (Currently contains all ArXiv examples for the intro -> abstract task)

## Dataset Description 

See [here](https://arxiv.org/category_taxonomy) for a list of ArXiv paper categories

The dataset contains a json file for each paper intro/abstract pair with the category, article split into sentences, abstract split into sentences, and an id 

## Instructions 

Setting up data: 

mkdir data 

cd data 

Download data from Google Drive 

for train, test and val splits: tar -xvf {split}.tar


Training Generator: 

Example w/ Bio summarization model: python train.py --data_dir ../data --dataset bio 

Example w/ CS summarization model: python train.py --data_dir ../data --dataset cs 

Decoding from Generator:

Example: python decode.py --load_epoch 12 --dataset bio --data_dir ../data --topk 3 --num_cands 30 --split test --model_dir ./model

Generations for each example are located in the file at ./generator/{dataset}_gpt2_gen/{split}/{split}-{id}.txt and the file contains a list of generated candidate summaries with every line of the file containing a different candidate summary  

Discriminator Models:

The sequence prediction model (Cohan et al., 2019) for predicting abstract discourse role tags can be found with instructions and requirements [here](https://github.com/skgabriel/coopnet/tree/main/discriminators/seq_tagging)

The training script for the adjacency model is [here](https://github.com/skgabriel/coopnet/tree/main/discriminators/adj) 

The token classification model for the factuality discriminator is [here](https://github.com/skgabriel/coopnet/tree/main/discriminators/factuality)

