# CQD-SHAP

This repository contains the code to reproduce the results from the paper "CQD-SHAP: Explainable Complex Query Answering with Shapley Values".

**Google Colab Notebook:** You can test CQD-SHAP directly in the Google Colab environment using the following link. Colab environment has already been set up with all the necessary packages we used in our experiments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anonymscientist/CQD-SHAP/blob/main/example_usage.ipynb).

## Prerequisites

### Environment Setup

We recommend using a conda environment with python `3.10`. You can use the following commands to set up the environment:

```bash
conda create -n xcqa python=3.10
```

To activate the environment, use:

```bash
conda activate xcqa
```

The list of required packages is provided in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

### Data Preparation

You can download the dataset from Google Drive using [gdown](https://github.com/wkentaro/gdown).

```bash
gdown https://drive.google.com/file/d/1yoZFUAY7DLOj4fC78pIU32SUSAEWRmLw
```

**Note:** Our original data is based on the [CQD](https://github.com/uclnlp/cqd/) repository. However, we made a few changes to NELL dataset to have the same format as FB15k-237 for a unified data loading process. Furthermore, we enriched FB15k-237 with titles of entities based on [KNN-KG repository](https://github.com/zjunlp/KNN-KG/tree/main/dataset/FB15k-237).


After downloading, run the following command to extract the data (this will create a `data` directory):

```bash
unzip data.zip
```

### Pre-trained Models

We also use pre-trained models from [CQD](https://github.com/uclnlp/cqd/). We've provided a new file that contains only the necessary models to reduce the download size. You can download the models from Google Drive using the following command:

```bash
gdown https://drive.google.com/file/d/1ot3CuVk4DorVu3JiHKzdumzGNaTREAU3
```

After downloading, run the following command to extract the models (this will create a `models` directory):

```bash
unzip models.zip
```

**Note:** We use the following pre-trained models for our experiments:
- FB15k-237: `models/FB15k-model-rank-1000-epoch-100-1602520745.pt`
- NELL995: `models/NELL-model-rank-1000-epoch-100-1602499096.pt`

## Necessary and Sufficient Explanations

The result for necessary and sufficient explanations evaluation can be reproduced by the `evaluation.py` script. The script takes the following arguments:

| Argument | Description | Value |
|----------|-------------|-------|
| `query_type` | The type of query to evaluate | `2p`, `3p`, `2i`, `3i`, `2u`, `pi` (i.e., 1p2i), `ip` (i.e., 2i1p), `up` (i.e., 2u1p) |
| `--data_dir` | The directory where the data is stored | e.g. `data/FB15k-237` (default) or `data/NELL995` |
| `--model_path` | The path to the pre-trained model | e.g. `models/FB15k-model-rank-1000-epoch-100-1602520745.pt` (default) or `models/NELL-model-rank-1000-epoch-100-1602499096.pt` |
| `--k` | Value of k for top-k beam search | Default is `10` |
| `--t-norm` | The t-norm to use for evaluation | `prod` (default), `min`, `max` |
| `--t-conorm` | The t-conorm to use for evaluation | `prob` (default), `max`, `min` |
| `--split` | The data split to use for evaluation | `test` (default), `valid` |
| `--method` | The method to use for generating explanations | `shapley` (default), `score`, `random`, `last`, `first` |
| `--explanation` | The type of explanation to evaluate | `necessary` (default), `sufficient` |
| `--output_path` | The path to save the evaluation results | Default is `output.json` |

An example command to run the evaluation for necessary explanations on 2p queries using the NELL dataset is as follows:

```bash
python evaluation.py 2p --k 10 --method shapley --explanation necessary --output_path eval/nell/necessary_2p_shapley.json --data_dir data/NELL995 --model_path models/NELL-model-rank-1000-epoch-100-1602499096.pt
```