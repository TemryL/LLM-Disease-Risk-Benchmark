from src.utils import prettify, Feature


train_size = 10000
test_size = 10000
split_seed = 0


features = {
    "baseline": [
        Feature(name="Age", field_id="21003", unit="years", is_valid=lambda x: x >= 0),
        Feature(
            name="Sex",
            field_id="31",
            unit=None,
            is_valid=lambda x: x in [0, 1],
            decode_map={0: "Male", 1: "Female"},
        ),
        Feature(name="BMI", field_id="21001", unit="kg/m²", is_valid=lambda x: x >= 0),
    ],
    "expanded": [
        Feature(name="Age", field_id="21003", unit="years", is_valid=lambda x: x >= 0),
        Feature(
            name="Sex",
            field_id="31",
            unit=None,
            is_valid=lambda x: x in [0, 1],
            decode_map={0: "Male", 1: "Female"},
        ),
        Feature(name="BMI", field_id="21001", unit="kg/m²", is_valid=lambda x: x >= 0),
        Feature(
            name="HDL cholesterol",
            field_id="30760",
            unit="mmol/L",
            is_valid=lambda x: x >= 0,
        ),
        Feature(
            name="LDL cholesterol",
            field_id="30780",
            unit="mmol/L",
            is_valid=lambda x: x >= 0,
        ),
        Feature(
            name="Total cholesterol",
            field_id="30690",
            unit="mmol/L",
            is_valid=lambda x: x >= 0,
        ),
        Feature(
            name="Triglycerides",
            field_id="30870",
            unit="mmol/L",
            is_valid=lambda x: x >= 0,
        ),
        Feature(
            name="Diastolic blood pressure",
            field_id="4079",
            unit="mmHg",
            is_valid=lambda x: x >= 0,
        ),
        Feature(
            name="Ever smoked",
            field_id="20160",
            unit=None,
            is_valid=lambda x: x in [0, 1],
            decode_map={0: "No", 1: "Yes"},
        ),
        Feature(
            name="Snoring",
            field_id="1210",
            unit=None,
            is_valid=lambda x: x in [1, 2],
            decode_map={1: "Yes", 2: "No"},
        ),
        Feature(
            name="Insomnia",
            field_id="1200",
            unit=None,
            is_valid=lambda x: x in [1, 2, 3],
            decode_map={1: "Never/rarely", 2: "Sometimes", 3: "Usually"},
        ),
        Feature(
            name="Daytime napping",
            field_id="1190",
            unit=None,
            is_valid=lambda x: x in [1, 2, 3],
            decode_map={1: "Never/rarely", 2: "Sometimes", 3: "Usually"},
        ),
        Feature(
            name="Sleep duration",
            field_id="1160",
            unit="hours/day",
            is_valid=lambda x: x >= 0,
        ),
        Feature(
            name="Chronotype",
            field_id="1180",
            unit=None,
            is_valid=lambda x: x in [1, 2, 3, 4],
            decode_map={
                1: "Definitely a 'morning' person",
                2: "More a 'morning' than 'evening' person",
                3: "More an 'evening' than a 'morning' person",
                4: "Definitely an 'evening' person",
            },
        ),
    ],
}
phenotypes = [
    "Asthma",
    "Cataract",
    "Diabetes",
    "GERD",
    "Hay-fever & Eczema",
    "Major depression",
    "Myocardial infarction",
    "Osteoarthritis",
    "Pneumonia",
    "Stroke",
]


log_reg_cfg = dict(random_state=42, max_iter=1000)


few_shot_cfg = dict(
    system_prompt="Given the following health information:\n",
    instruction=lambda phenotype: f"predict whether the individual has the following condition or not. Respond Yes or No.\n{prettify(phenotype)}: ",
    positive_labels=["Yes"],
    negative_labels=["No"],
    max_new_tokens=1,
    debug=False,
)
