import torch
import string
import pandas as pd
from typing import Callable, List
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from .logger import logger


@dataclass
class Feature:
    name: str
    field_id: str
    is_valid: Callable
    unit: str = None
    decode_map: dict = None


def load_llm(model_id, freeze=True, inference=True, torch_dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, device_map="auto", padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch_dtype
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.bos_token_id

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    if inference:
        model = model.eval()

    logger.info(f'{model_id.split("/")[-1]} loaded succesfully.')
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            logger.info(
                "Memory allocated on cuda device {}: {:.1f} GB".format(
                    i, torch.cuda.memory_allocated(i) * 1e-9
                )
            )

    return tokenizer, model


def get_torch_dtype(precision: str) -> torch.dtype:
    if precision == "float32":
        return torch.float32
    elif precision == "float16":
        return torch.float16
    elif precision == "bfloat16":
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported precision type: {precision}")


def serialize_features(row: pd.Series, features: List[Feature]) -> str:
    row_dict = row.to_dict()
    if not row_dict:
        return ""

    formatted_items = []
    for feature in features:
        value = row_dict[feature.name]
        if isinstance(value, str):
            if feature.unit:
                value = "{}: {} {};\n".format(feature.name, value, feature.unit)
            else:
                value = "{}: {};\n".format(feature.name, value)
        else:
            if feature.unit:
                value = "{}: {:.1f} {};\n".format(feature.name, value, feature.unit)
            else:
                value = "{}: {:.1f};\n".format(feature.name, value)
        formatted_items.append(value)

    serialized_features = "".join(formatted_items)
    return serialized_features


def normalize_string(value: str) -> str:
    value = value.lower()
    value = value.replace(" ", "")
    value = value.translate(str.maketrans("", "", string.punctuation))
    return value


def prettify(phenotype):
    if phenotype[0].isupper():
        return " ".join(phenotype.split("-"))
    else:
        return " ".join(phenotype.split("-")).capitalize()


def check_icd_code(value, code_list):
    return any(value.startswith(prefix) for prefix in code_list)


def is_valid_date(value):
    return not (
        pd.isna(value)
        or value
        in [
            "1900-01-01",
            "1901-01-01",
            "1902-02-02",
            "1903-03-03",
            "1909-09-09",
            "2037-07-07",
        ]
    )
