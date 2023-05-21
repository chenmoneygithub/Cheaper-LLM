import re

import langchain
import torch

from cheaper_llm.cache import Cache
from cheaper_llm.model_cascading.model_collection import MODEL_COLLECTION
from cheaper_llm.model_cascading.model_collection import MODEL_ORDER
from cheaper_llm.model_cascading.prompts_utils import generate_scoring_prompt


def is_gpu_available():
    return torch.cuda.is_available()


def get_model_response(model, prompt):
    model_meta = MODEL_COLLECTION[model]
    source = model_meta["source"]
    if source == "huggingface":
        tokenizer_info = model_meta["huggingface"]["tokenizer"]
        model_info = model_meta["huggingface"]["model"]

        tokenizer = tokenizer_info["class"].from_pretrained(
            *tokenizer_info["args"], **tokenizer_info["kwargs"]
        )
        model = model_info["class"].from_pretrained(
            *model_info["args"], **model_info["kwargs"]
        )

        inputs = tokenizer(prompt, return_tensors="pt")
        if is_gpu_available:
            inputs = inputs.to_device("cuda")
        outputs = model.generate(inputs, penalty_alpha=0.2, top_k=5)
        return tokenizer.decode(outputs[0])
    elif source == "langchain":
        model_info = model_meta["langchain"]["model"]
        model = model_info["class"](*model_info["args"], **model_info["kwargs"])
        return model(prompt)


class ModelCascading:
    def __init__(self, candidate_models=None, scoring_model=None):
        if candidate_models is None:
            self.candidate_models = MODEL_ORDER
        else:
            self.candidate_models = candidate_models

        if scoring_model is None:
            # If no scoring model is set, use a manually picked one.
            # For demo we are using openai's text-babbage-001.
            self.scoring_model = langchain.llms.OpenAI(
                model_name="text-babbage-001"
            )
        else:
            self.scoring_model = scoring_model

        self.cache = Cache()

    def __call__(self, prompt):
        rough_cost = 0
        for model in self.candidate_models:
            price = MODEL_COLLECTION[model]["price"]
            response = get_model_response(model, prompt).strip()
            response_score = self.calculate_score(prompt, response)
            # 1.15 is a rough estimate of multiplier from text to tokens when
            # using BPE tokenizer or SPE tokenizer.
            total_tokens = (len(prompt) + len(response)) * 1.15
            rough_cost += total_tokens * price / 1000.0
            if response_score > 70:
                return {
                    "response": response,
                    "model": model,
                    "cost": f"{rough_cost:.5f}",
                }
        return {
            "response": response,
            "model": model,
            "cost": f"{rough_cost:.5f}",
        }

    def calculate_score(self, prompt, response):
        rating_prompt = generate_scoring_prompt(prompt, response)
        search_pattern = "[0-9]+"
        score_response = self.scoring_model(rating_prompt)
        return int(re.search(search_pattern, score_response).group())
