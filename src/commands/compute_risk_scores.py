import sys

sys.path.append(".")
sys.path.append("..")

from dotenv import load_dotenv

load_dotenv()

import os
import json
import argparse
import pandas as pd
from huggingface_hub import login
from importlib.machinery import SourceFileLoader
from src.logger import logger
from src.utils import get_torch_dtype
from src.models.xgb_predictor import XGBoostPredictor
from src.models.log_reg_predictor import LogRegPredictor
from src.models.few_shot_predictor import FewShotPredictor


def load_predictor(model_name, config):
    if model_name == "LogReg":
        cfg = config.log_reg_cfg
        predictor = LogRegPredictor(**cfg)
    elif model_name == "XGBoost":
        predictor = XGBoostPredictor()
    else:
        try:
            cfg = config.few_shot_cfg
            llm_id = model_name
            predictor = FewShotPredictor(llm_id, **cfg)
        except Exception as e:
            logger.error(f"Error loading predictor: {e} ")
            raise
    return predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to compute and save risk scores for a given phenotype using specified model and features."
    )
    parser.add_argument("request_file", help="Path to request file")
    parser.add_argument("out_dir", help="Output directory to save risk scores.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/preprocessed_data.csv",
        help="Path to CSV file containing preprocessed UKB data.",
    )
    parser.add_argument(
        "--nb_shots",
        type=int,
        default=0,
        help="Number of few-shots to inject in the prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="configs/predictor_cfgs.py",
        help="Path to config file.",
    )
    parser.add_argument(
        "--subset",
        help="Whether to compute scores only for a subset (default: False)",
        action="store_true",
    )
    return parser.parse_args()


def main():
    # Parse arguments:
    logger.info("Parsing arguments ...")
    try:
        args = parse_args()
        request_file = args.request_file
        out_dir = args.out_dir
        data_path = args.data_path
        config = SourceFileLoader("config", args.config_file).load_module()
        nb_shots = args.nb_shots
        batch_size = args.batch_size
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")
        raise

    # Load request file:
    with open(request_file, "r") as f:
        request_config = json.load(f)
    model_name = request_config["model"]

    # Update config:
    if model_name not in ["LogReg", "XGBoost"]:
        try:
            torch_dtype = get_torch_dtype(request_config["precision"])
        except ValueError as e:
            logger.error(e)
            raise
        request_config["nb_shots"] = nb_shots
        config.few_shot_cfg["nb_shots"] = nb_shots
        config.few_shot_cfg["batch_size"] = batch_size
        config.few_shot_cfg["torch_dtype"] = torch_dtype

    # Load data:
    logger.info("Loading data ...")
    try:
        if args.subset:
            data = pd.read_csv(data_path, nrows=50000)
        else:
            data = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    # Load predictor:
    predictor = load_predictor(model_name, config)

    for phenotype in config.phenotypes:
        for feature_set, features in config.features.items():
            # Compute risk scores:
            logger.info(f"Computing risk scores for {phenotype} with {model_name} ...")
            try:
                predictor.compute_scores(
                    data,
                    features,
                    phenotype,
                    split_seed=config.split_seed,
                    train_size=config.train_size,
                    test_size=config.test_size,
                )
            except Exception as e:
                logger.error(f"Error computing risk scores: {e}")
                raise

            # Evaluate predictor:
            logger.info(f"Evaluating {model_name} ...")
            try:
                auroc, auprc = predictor.evaluate()
                logger.info("AUROC: {:.2f}".format(auroc))
                logger.info("AUPRC: {:.2f}".format(auprc))
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                raise

            # Save risk scores:
            logger.info("Saving risk scores ...")
            request_config["phenotype"] = phenotype
            request_config["feature_set"] = feature_set
            predictor.save_scores(os.path.join(out_dir, model_name), request_config)


if __name__ == "__main__":
    login(token=os.getenv("HF_TOKEN"))
    main()
