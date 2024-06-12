import sys
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from .risk_predictor import RiskPredictor
from ..logger import logger


class XGBoostPredictor(RiskPredictor):
    def __init__(self):
        super().__init__(model_name="XGBoost")
        self.clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")

    def prepare_data(
        self, data, features, phenotype, split_seed, train_size, test_size
    ):
        # Encode string values:
        le = LabelEncoder()
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = le.fit_transform(data[col])
        
        # Split data:
        df_train, df_val = train_test_split(
            data, train_size=train_size, test_size=test_size, random_state=split_seed
        )
        
        # Convert to numpy array:
        feature_names = [feature.name for feature in features]
        X_train = df_train[feature_names].to_numpy()
        X_val = df_val[feature_names].to_numpy()
        y_train = df_train[phenotype].astype(int).to_numpy().squeeze()
        y_val = df_val[phenotype].astype(int).to_numpy().squeeze()

        # Save eids and true target
        self.eids = list(df_val["eid"])
        self.y_true = list(y_val)

        return X_train, X_val, y_train, y_val

    def compute_scores(
        self, data, features, phenotype, split_seed, train_size, test_size
    ):
        X_train, X_val, y_train, y_val = self.prepare_data(
            data, features, phenotype, split_seed, train_size, test_size
        )

        # Train:
        logger.info("Training XGBoost ...")
        try:
            self.clf.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            sys.exit()

        # Predict:
        self.y_scores = list(self.clf.predict_proba(X_val)[:, 1])
        return self.y_scores
