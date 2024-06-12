import torch
import pandas as pd
from tqdm import tqdm
from typing import List
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from ..logger import logger
from .risk_predictor import RiskPredictor
from ..utils import serialize_features, load_llm, normalize_string


class FewShotPredictor(RiskPredictor):
    def __init__(self, llm_id, nb_shots, batch_size, system_prompt, instruction, positive_labels, negative_labels, max_new_tokens, torch_dtype, debug=False):
        llm_name = llm_id.split("/")[-1]
        super().__init__(model_name=llm_name)
        self.llm_id = llm_id
        self.nb_shots = nb_shots
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.instruction = instruction
        self.max_new_tokens = max_new_tokens
        self.debug = debug
        self.tokenizer, self.model = load_llm(model_id=llm_id, torch_dtype=torch_dtype)
        self.vocab = self.tokenizer.get_vocab()
        self.positive_tokens = self._get_matching_tokens(positive_labels)
        self.negative_tokens = self._get_matching_tokens(negative_labels)
        
    def prepare_data(
        self, data, features, phenotype, split_seed, train_size, test_size
    ):
        # Split data:
        df_train, df_val = train_test_split(
            data, train_size=train_size, test_size=test_size, random_state=split_seed
        )
        feature_names = [feature.name for feature in features]
        
        # Sample few shots:
        few_shots_prompt = ""
        if self.nb_shots > 0:
            nb_controls = int(self.nb_shots / 2)
            nb_cases = self.nb_shots - nb_controls 
            df_controls = df_train[~df_train[phenotype]].sample(n=nb_controls, random_state=split_seed)
            df_cases = df_train[df_train[phenotype]].sample(n=nb_cases, random_state=split_seed)
            few_shots = pd.concat([df_controls, df_cases]).sample(frac=1).reset_index(drop=True)
            
            few_shots['prompt'] = few_shots.apply(lambda x: "".join([self.system_prompt, serialize_features(x, features) , self.instruction(phenotype)]), axis=1)
            few_shots['prompt'] = few_shots.apply(lambda x: "".join([x["prompt"], "Yes."]) if x[phenotype] else "".join([x["prompt"], "No."]), axis=1)
            few_shots_prompt = "\n".join(few_shots['prompt']) + "\n"
            
        # Construct prompts:
        prompts = df_val[feature_names].apply(
            lambda x: "".join([few_shots_prompt, self.system_prompt, serialize_features(x, features), self.instruction(phenotype)]), axis=1
        ).to_list()
        if self.debug:
            logger.debug(f"\nPrompt example: {prompts[0]}\n")

        # Save eids and true target:
        self.eids = list(df_val["eid"])
        self.y_true = list(df_val[phenotype])

        return prompts

    def compute_scores(
        self, data, features, phenotype, split_seed, train_size, test_size
    ):
        prompts = self.prepare_data(data, features, phenotype, split_seed, train_size, test_size)

        with torch.no_grad():
            self.y_scores = []
            for i in tqdm(range(0, len(prompts), self.batch_size), desc="Processing batches"):
                # Create batch:
                batch = prompts[i : i + self.batch_size]
 
                # Tokenize prompts:
                tokenized_input = self.tokenizer(batch, return_tensors="pt", padding=True)
                
                # Move to device:
                if torch.cuda.is_available():
                    tokenized_input = tokenized_input.to(0)

                # Generate output:
                output = self.model.generate(**tokenized_input, output_logits=True, return_dict_in_generate=True, max_new_tokens=self.max_new_tokens, pad_token_id=self.tokenizer.eos_token_id)
                
                # Compute risk scores:
                positive_probs = self._get_next_token_probs(output, self.positive_tokens, debug=self.debug if i == 0 else False)
                negative_probs = self._get_next_token_probs(output, self.negative_tokens)
                positive_probs = positive_probs.view(self.batch_size, -1).sum(dim=1)
                negative_probs = negative_probs.view(self.batch_size, -1).sum(dim=1)
                risk_scores = positive_probs / (positive_probs + negative_probs)
                self.y_scores.extend(risk_scores.tolist())

        return self.y_scores
    
    def _get_matching_tokens(self, labels):
        normalized_labels = [normalize_string(label) for label in labels]
        matching_tokens = []
        for token_label in self.vocab.keys():
            normalized_token_label = normalize_string(token_label)
            if normalized_token_label in normalized_labels:
                matching_tokens.append(token_label)
        return matching_tokens
    
    def _get_next_token_probs(self, output, target_tokens: List[str], debug=False) -> float:
        # Encode target words:
        target_ids = [self.vocab[word] for word in target_tokens]
        
        # Compute target probabilities:
        probs = F.softmax(output.logits[0], dim=-1)
        target_probs = probs[:, target_ids]
        
        if debug:
            scores, token_ids = torch.sort(probs[0,:].view(-1), descending=True)
            token_labels = [self.tokenizer.decode([token_id]) for token_id in token_ids[:20]]
            for score, label, token_id in zip(scores, token_labels, token_ids):
                logger.debug(repr("{:.2f} {} {}".format(score.item(), label, token_id.item())))
        
        return target_probs.squeeze()