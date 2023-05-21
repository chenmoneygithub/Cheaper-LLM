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
    },
    "gpt-3.5": {
        "source": "openai",
        "id": "gpt-3.5",
    },
}

MODEL_ORDER = ["pythia-6.9b", "gpt-3.5"]
