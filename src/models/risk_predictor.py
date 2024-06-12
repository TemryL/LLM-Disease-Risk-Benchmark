import os
import json
import pickle
from ..logger import logger
from .json_encoder import NoIndent, NoIndentEncoder
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


class RiskPredictor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.feature_set = None
        self.eids = None
        self.y_scores = None
        self.y_true = None

    def compute_scores(self, data):
        raise NotImplementedError

    def evaluate(self):
        if self.y_scores is None or self.y_true is None:
            logger.error("Predictions or true labels are not available.")
            raise ValueError("Predictions or true labels are missing.")

        auroc_score = roc_auc_score(self.y_true, self.y_scores)
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_scores
        )
        auprc_score = auc(recall, precision)
        return auroc_score, auprc_score

    def save_scores(self, out_dir, config):
        if self.y_scores is None:
            logger.error("No scores to save.")
            raise ValueError("No scores available to save.")

        # Unpack config:
        phenotype = config['phenotype']
        feature_set = config['feature_set']
        precision = config.get('precision', None)
        nb_shots = config.get('nb_shots', None)
        
        # Creating the full path to save the file;
        normalized_phenotype = phenotype.lower().replace(" ", '-')
        filename = f"rs_{normalized_phenotype}_{feature_set}"
        if precision:
            filename += f"_{precision}"
        if nb_shots is not None:
            filename += f"_{nb_shots}-shots"
        filename += ".json"
        full_path = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)

        # Save output in JSON file:
        output = {
            "config": config,
            "eids": NoIndent(self.eids),
            "y_true": NoIndent([int(i) for i in self.y_true]),
            "y_scores": NoIndent([float(i) for i in self.y_scores])
        }
        with open(full_path, 'w') as f:
            f.write(json.dumps(output, cls=NoIndentEncoder, indent=2))

        logger.info(f"Scores saved successfully in {full_path}")
        return full_path

    def load_scores(self, file, verbose=True):
        try:
            with open(file, "rb") as f:
                self.eids, self.y_true, self.y_scores = pickle.load(f)
            if verbose:
                logger.info("Scores loaded successfully from file.")
        except FileNotFoundError:
            logger.error(f"File {file} not found.")
            raise
        except Exception as e:
            logger.error(f"Failed to load scores from file: {str(e)}")
            raise
