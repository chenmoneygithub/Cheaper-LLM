import re

import langchain
import torch
from absl import logging

from cheaper_llm.cache import PromptCache
from cheaper_llm.model_cascading.model_collection import MODEL_COLLECTION
from cheaper_llm.model_cascading.model_collection import MODEL_ORDER
from cheaper_llm.model_cascading.prompts_utils import generate_scoring_prompt


def is_gpu_available():
    return torch.cuda.is_available()


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

        self.in_memory_models = {}
        for model in self.candidate_models:
            model_meta = MODEL_COLLECTION[model]
            source = model_meta["source"]
            if source == "huggingface":
                self.load_in_memory_models(model, source)

        try:
            # If there is a redis server running.
            self.cache = PromptCache()
        except:
            self.cache = None

    def load_in_memory_models(self, model_id, source):
        if source != "huggingface":
            raise ValueError(
                f"Source must be 'huggingface', but received: {source}."
            )
        model_meta = MODEL_COLLECTION[model_id]
        model_info = model_meta["huggingface"]["model"]
        model = model_info["class"].from_pretrained(
            *model_info["args"], **model_info["kwargs"]
        )
        self.in_memory_models[model_id] = model

    def get_model_response(self, model, prompt):
        logging.info(f"Trying model {model}...")
        model_meta = MODEL_COLLECTION[model]
        source = model_meta["source"]
        if source == "huggingface":
            tokenizer_info = model_meta["huggingface"]["tokenizer"]
            model_info = model_meta["huggingface"]["model"]

            tokenizer = tokenizer_info["class"].from_pretrained(
                *tokenizer_info["args"], **tokenizer_info["kwargs"]
            )
            if model in self.in_memory_models:
                model = self.in_memory_models[model]
            else:
                model = model_info["class"].from_pretrained(
                    *model_info["args"], **model_info["kwargs"]
                )

            inputs = tokenizer(prompt, return_tensors="pt")
            if is_gpu_available:
                inputs = inputs.to("cuda")
            generate_info = model_meta["huggingface"]["generate"]
            outputs = model.generate(**inputs, **generate_info["kwargs"])
            decoded = tokenizer.decode(outputs[0])
            decoded = decoded[len(prompt) :].strip()
            return decoded
        elif source == "langchain":
            model_info = model_meta["langchain"]["model"]
            model = model_info["class"](
                *model_info["args"], **model_info["kwargs"]
            )
            return model(prompt)

    def __call__(self, prompt):
        rough_cost = 0
        cache_read = None if self.cache is None else self.cache.get(prompt)
        candidate_models = self.candidate_models
        if cache_read is not None:
            score = float(cache_read["score"])
            model_used = cache_read["model"]
            if score > 0.97:
                logging.info("Cache hit! Reading from cache...")
                return {
                    "response": cache_read["content"],
                    "model": cache_read["model"],
                    "cost": 0,
                }
            elif score > 0.5:
                if model_used in candidate_models:
                    # Start from a decent model.
                    index = candidate_models.index(model_used)
                    candidate_models = candidate_models[index:]

        for model in self.candidate_models:
            price = MODEL_COLLECTION[model]["price"]
            response = self.get_model_response(model, prompt).strip()
            response_score = self.calculate_score(prompt, response)
            # 1.15 is a rough estimate of multiplier from text to tokens when
            # using BPE tokenizer or SPE tokenizer.
            total_tokens = (len(prompt) + len(response)) * 1.15
            rough_cost += total_tokens * price / 1000.0
            if response_score > 70:
                if self.cache is not None:
                    self.cache.put(prompt, response, model)
                return {
                    "response": response,
                    "model": model,
                    "cost": f"{rough_cost:.5f}",
                }
        if self.cache is not None:
            self.cache.put(prompt, response, model)
        return {
            "response": response,
            "model": model,
            "cost": f"{rough_cost:.5f}",
        }

    def calculate_score(self, prompt, response):
        rating_prompt = generate_scoring_prompt(prompt, response)
        search_pattern = "[0-9]+"
        score_response = self.scoring_model(rating_prompt)
        score = re.search(search_pattern, score_response)
        if score is None:
            return 0
        return int(score.group())
