import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("UKB-Tools")

from tqdm import tqdm
from ukb_tools.preprocess.utils import rename_features
from ukb_tools.preprocess.labeling import match_phenotype
from ukb_tools.preprocess.filtering import filter_partially_populated_rows


def preprocess_pipeline(df, preprocess_cfg):
    # Unpack config vars:
    features = preprocess_cfg.features
    phenotypes_ids = preprocess_cfg.phenotype_ids

    # Filter participant with valid phenotype entries:
    eids_valid_pheno = filter_partially_populated_rows(df, phenotypes_ids)
    df = df.loc[list(eids_valid_pheno)]

    # Label the participant based on whether they have been diagnosed or not:
    matching_rules = preprocess_cfg.phenotype_matching_rules
    for phenotype, rules in tqdm(
        matching_rules.items(), desc="Labeling participant's phenotypes"
    ):
        df[f"{phenotype}"] = df.apply(match_phenotype, phenotype_rules=rules, axis=1)

    # Rename feature columns:
    features_dict = {feat.name: feat.field_id for feat in features}
    df, feature_names = rename_features(df, features_dict)

    # Keep features and phenotypes columns:
    phenotype_names = list(matching_rules.keys())
    df = df[feature_names + phenotype_names]

    # Compute average diastolic blood pressure:
    df["Diastolic blood pressure"] = (
        (df["Diastolic blood pressure_0"] + df["Diastolic blood pressure_1"])
    ) / 2
    df = df.drop(columns=["Diastolic blood pressure_0", "Diastolic blood pressure_1"])

    # Filter participant with valid features:
    for feat in tqdm(features, desc="Filtering features"):
        df = df[df[feat.name].apply(feat.is_valid)]

    # Decode features:
    for feat in tqdm(features, desc="Decoding features"):
        df[feat.name] = df[feat.name].apply(
            lambda x: feat.decode_map[x] if feat.decode_map else x
        )

    return df
