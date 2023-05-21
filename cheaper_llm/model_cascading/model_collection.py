import langchain
import torch
from transformers import AutoTokenizer
from transformers import GPTNeoXForCausalLM

MODEL_COLLECTION = {
    "pythia-6.9b": {
        "source": "huggingface",
        "huggingface": {
            "model": {
                "class": GPTNeoXForCausalLM,
                "args": {
                    "EleutherAI/pythia-6.9b",
                },
                "kwargs": {
                    "torch_dtype": torch.float16,
                    "device_map": "auto",
                },
            },
            "tokenizer": {
                "class": AutoTokenizer,
                "args": {
                    "EleutherAI/pythia-6.9b",
                },
                "kwargs": {},
            },
        },
        "price": 0,  # Per 1k tokens.
    },
    "text-babbage-001": {
        "source": "langchain",
        "langchain": {
            "model": {
                "class": langchain.llms.OpenAI,
                "args": {},
                "kwargs": {
                    "model_name": "text-babbage-001",
                    "temperature": 0.9,
                },
            }
        },
        "price": 0.0005,  # Per 1k tokens.
    },
    "gpt-3.5-turbo": {
        "source": "langchain",
        "langchain": {
            "model": {
                "class": langchain.llms.OpenAI,
                "args": {},
                "kwargs": {
                    "model_name": "gpt-3.5-turbo",
                    "temperature": 0.9,
                },
            }
        },
        "price": 0.0005,  # Per 1k tokens.
    },
}

MODEL_ORDER = ["pythia-6.9b", "text-babbage-001", "gpt-3.5-turbo"]
