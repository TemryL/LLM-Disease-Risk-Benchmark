import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("UKB-Tools")

import argparse
from ukb_tools.data import UKB
from ukb_tools.logger import logger
from src.preprocess import preprocess_pipeline
from importlib.machinery import SourceFileLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data", help="Path to CSV file containing raw UKB data.")
    parser.add_argument("config", help="Path to the preprocessing config file.")
    parser.add_argument(
        "out_file",
        help="File to write the resulting dataframe.",
        default="preprocessed_data.csv",
        nargs="?",
        const=1,
    )
    return parser.parse_args()


def main():
    # Parse arguments:
    args = parse_args()
    raw_data = args.raw_data
    out_file = args.out_file
    preprocess_cfg = SourceFileLoader("config", args.config).load_module()

    # Load data:
    logger.info("Loading UKB data ...")
    ukb = UKB(raw_data)
    ukb.load_data(instance="0")

    # Preprocess data:
    ukb.preprocess(preprocess_pipeline, [preprocess_cfg])

    # Save to CSV file:
    logger.info(f"Saving data to {out_file} ...")
    ukb.data.to_csv(out_file)
    logger.info("Data saved successfully.")


if __name__ == "__main__":
    main()
