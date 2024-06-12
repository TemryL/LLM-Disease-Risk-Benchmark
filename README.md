# LLM-Disease-Risk-Benchmark

This repository contains the code for the paper "Benchmarking LLMs for Disease Risk Prediction: Zero-shot and Few-shot Analysis on UK BioBank" presented at [NeurIPS 2024 Datasets and Benchmarks Track](https://neurips.cc/Conferences/2024/CallForDatasetsBenchmarks). The benchmark evaluates the performance of open-source LLMs in predicting disease risk based on individual-level characteristics from the [UK Biobank dataset](https://www.ukbiobank.ac.uk/). The goal is to provide a comprehensive, open-source benchmark for disease risk prediction using LLMs. Our results can be found on a live leaderboard hosted on [Hugging Face](https://huggingface.co/spaces/TemryL/LLM-Disease-Risk-Leaderboard).

## Table of Contents
1. [Introduction](#introduction)
2. [Repository Structure](#repository-structure)
3. [Setup Instructions](#setup-instructions)
4. [Data Preparation](#data-preparation)
5. [Running the Benchmark](#running-the-benchmark)
6. [Citation](#citation)
7. [License](#license)

## Introduction
Large language models (LLMs) have shown impressive abilities in solving complex tasks across various fields, including healthcare. While significant efforts have been made to assess the performance of LLMs in question-answering within the health domain, there is no open-source benchmark that evaluates the performance of LLMs in predicting disease conditional on individual-level characteristics. This project aims to fill that gap by providing a benchmark for evaluating LLMs' predictive capabilities in disease risk prediction using participant data from the UK Biobank.

## Repository Structure
```
├── LICENSE
├── README.md
├── UKB-Tools
│   ├── LICENSE
│   ├── README.md
│   ├── commands
│   │   ├── create_data.py
│   │   ├── create_eu_set.py
│   │   └── get_newest_baskets.py
│   ├── requirements.txt
│   └── ukb_tools
│       ├── __init__.py
│       ├── data.py
│       ├── logger.py
│       ├── preprocess
│       │   ├── filtering.py
│       │   ├── labeling.py
│       │   └── utils.py
│       └── tools.py
├── configs
│   ├── predictor_cfgs.py
│   └── preprocess_cfg.py
├── data
│   ├── request_examples
│   │   ├── LogReg.json
│   │   ├── XGBoost.json
│   │   └── meditron-7b.json
│   └── ukb_fields.txt
├── requirements.txt
└── src
    ├── __init__.py
    ├── commands
    │   ├── __init__.py
    │   ├── compute_risk_scores.py
    │   ├── generate_results.py
    │   └── preprocess_raw_data.py
    ├── evaluate.py
    ├── logger.py
    ├── models
    │   ├── __init__.py
    │   ├── few_shot_predictor.py
    │   ├── json_encoder.py
    │   ├── log_reg_predictor.py
    │   ├── risk_predictor.py
    │   └── xgb_predictor.py
    ├── preprocess.py
    └── utils.py
```

## Setup Instructions

### Cloning the Repository
First, clone the repository using the following command:
```bash
git clone --recurse-submodules -j8 https://github.com/TemryL/LLM-Disease-Risk-Benchmark.git
```

### Installing Dependencies
Install the required dependencies with Python 3.11:
```bash
pip install -r requirements.txt
```

## Data Preparation

### Setting Environment Variables
Set your environment variables:
```bash
export UKB_FOLDER=...
export PROJECT_ID=...
```

### Creating Raw Data
Follow these steps to create raw data. More details can be found in the [UKB-Tools README](https://github.com/TemryL/UKB-Tools/blob/main/README.md):
```bash
cd UKB-Tools
python commands/get_newest_baskets.py $UKB_FOLDER $PROJECT_ID ../data/ukb_fields.txt ../data/field_to_basket.json
python commands/create_data.py $UKB_FOLDER ../data/field_to_basket.json ../data/raw_data.csv
```

### Preprocessing Raw Data
Preprocess the raw data with the following command:
```bash
cd ..
python src/commands/preprocess_raw_data.py data/raw_data.csv configs/preprocess_cfg.py data/preprocessed_data.csv
```

## Running the Benchmark

### Preparing Environment Variables
Create a `.env` file with your Hugging Face access token:
```
HF_TOKEN = '...'
```

### Computing Risk Scores
Compute risk scores using Logistic Regression and Meditron-7B models as examples:
```bash
python src/commands/compute_risk_scores.py data/request_examples/LogReg.json output/risk_scores
python src/commands/compute_risk_scores.py data/request_examples/meditron-7b.json output/risk_scores
```

### Evaluating Results
Evaluate the risk scores of a single experiement using the following commands:
```bash
python -m src.evaluate output/risk_scores/LogReg/rs_diabetes_baseline.json
python -m src.evaluate output/risk_scores/epfl-llm/meditron-7b/rs_asthma_baseline_float16_0-shots.json
```

### Generating Final Results
Generate the final results for each experiments of a given model:
```bash
python src/commands/generate_results.py output/risk_scores output/results LogReg
python src/commands/generate_results.py output/risk_scores output/results epfl-llm/meditron-7b
```

## Citation
If you use this benchmark in your research, please cite our paper:
```
@article{TBA,
  title={Benchmarking LLMs for Disease Risk Prediction: Zero-shot and Few-shot Analysis on UK BioBank},
  author={Tom Mery, Chirag J. Patel},
  year={2024}
}
```

## License
This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.