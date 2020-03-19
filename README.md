# MGTC

This the the repository for the CAiSE-2020 paper "***An Approach for Process Model Extraction By Multi-Grained Text Classification***". In this paper, we formalize the PME task into the multi-grained text classification problem, and propose a neural network and the coarse-to-fine (grained) learning mechanism to effectively extract multi-grained procedural knowledge.


## Overview
- code/ 
  This directory contains the source code of our approach.
- data/ 
  This directory contains two datasets used for evaluation.


## Datasets
The dataset named as X-Y denotes a text classification task Y on data source X. For two sentence-level tasks -- SC (sentence classification) and SSR (sentence semantics recognition), an example <x,y> in each line denotes a sentence x and its label y. each of them can be solely used to evaluate single-sentence classification tasks. For the word-level task -- SRL (semantic role labeling), an example <x,i,y> in each line denotes a sentence x, a subordinate word index i and a corresponding label y. It can be further used to evaluate sequential-text classification tasks. We will keep updating them to provide more reliable version(s), including correcting wrongly-annotated labels and adding more training/testing examples. The up-to-date version can be directly downloaded from this repository.


## Reqirements
* Python (3.6.8, 3.7.3 or above)
* PyTorch (1.0.1) 
* [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html). Note that the embedding layer can be replaced by other models such as [BERT-Base](https://github.com/google-research/bert). If necessary, kindly replace corresponding lines of the code to do so.


### Citation
When referencing, please cite this paper as:

```
@inproceedings{MGTC,
  title={An Approach for Process Model Extraction By Multi-Grained Text Classification},
  author={Chen Qian and Lijie Wen and Akhil Kumar and Leilei Lin and Li Lin and Zan Zong and Shuang Li and Jianmin Wang},
  booktitle={Proceedings of The 32nd International Conference on Advanced Information Systems Engineering (CAiSE)},
  year={2020}
}
```

