import sys

sys.path.append(".")
sys.path.append("..")

import os
import json
import argparse
from src.logger import logger
from src.evaluate import evaluate
from src.models.json_encoder import NoIndentEncoder


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to evaluate predicted risk scores and generate results."
    )
    parser.add_argument("risk_scores_dir", help="Risk scores directory.")
    parser.add_argument("result_dir", help="Directory to store results.")
    parser.add_argument("model_name", help="Model to evaluate.")
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number bootstrapping iteration",
    )
    parser.add_argument(
        "--n_interp",
        type=int,
        default=100,
        help="Number interpolation point for ROC and PR curves",
    )
    parser.add_argument(
        "--confidence",
        type=int,
        default=95,
        help="Confidence of the confidence intervals",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed",
    )
    return parser.parse_args()


def main():
    # Parse arguments:
    logger.info("Parsing arguments ...")
    try:
        args = parse_args()
        risk_scores_dir = args.risk_scores_dir
        model_name = args.model_name
        result_dir = args.result_dir
        n_bootstrap = args.n_bootstrap
        n_interp = args.n_interp
        confidence = args.confidence
        seed = args.seed
    except Exception as e:
        logger.error(f"Error parsing arguments: {e}")

    # Initialize the output dictionary
    output = {}

    # List risk scores files
    for file in os.listdir(os.path.join(risk_scores_dir, model_name)):
        # Load risk scores
        with open(os.path.join(risk_scores_dir, model_name, file), "r") as f:
            risk_scores = json.load(f)
            y_true = risk_scores["y_true"]
            y_scores = risk_scores["y_scores"]
            config = risk_scores["config"]
            phenotype = config["phenotype"]
            feature_set = config["feature_set"]
            weight_precision = config.get("precision", None)
            nb_shots = config.get("nb_shots", None)
            if "status" in config.keys():
                del config["status"]
            if "submitted_time" in config.keys():
                del config["submitted_time"]
            if "phenotype" in config.keys():
                del config["phenotype"]

        # Evaluate predicted risk scores
        logger.info("Evaluating predicted risk scores ...")
        try:
            results = evaluate(
                y_true,
                y_scores,
                n_bootstrap=n_bootstrap,
                n_interp=n_interp,
                seed=seed,
                confidence=confidence,
                verbose=True,
            )
        except Exception as e:
            logger.error(f"Error parsing evaluating predicted risk scores: {e}")
            raise

        # Create a key from the combination of feature_set, weight_precision, and nb_shots
        key = (feature_set, weight_precision, nb_shots)

        # Update results, initialize the structure if key does not exist
        if key not in output:
            output[key] = {"results": {}, "config": None, "configs": []}

        output[key]["results"][phenotype] = results
        output[key]["configs"].append(config)

    # Save results in output dir:
    output_dir = os.path.join(result_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    for (feature_set, weight_precision, nb_shots), value in output.items():
        # Generate filename:
        filename = f"results_{feature_set}"
        if weight_precision:
            filename += f"_{weight_precision}"
        if nb_shots is not None:
            filename += f"_{nb_shots}-shots"
        filename += ".json"

        # Check that config file was the same for each phenotype
        configs = value["configs"]
        if all(config == configs[0] for config in configs):
            value["config"] = configs[0]
            del value["configs"]
        else:
            raise ValueError(
                f"Inconsistent configs found for key {(feature_set, weight_precision, nb_shots)}"
            )

        # Save:
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w") as f:
            f.write(json.dumps(value, cls=NoIndentEncoder, indent=2))


if __name__ == "__main__":
    main()
