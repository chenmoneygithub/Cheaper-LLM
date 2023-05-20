
MODELS = {
    "gpt-neo": {
        "source": "huggingface",
        "id": "EleutherAI/gpt-neo-2.7B",
    },
    "alpaca": {
        "source": "huggingface",
        "id": "chavinlo/alpaca-native",
    },
    "pythia-6.9b": {
        "source": "huggingface",
        "id": "EleutherAI/pythia-6.9b",
    },
    "gpt-3.5": {
        "source": "openai",
        "id": "gpt-3.5",
    }
}

MODEL_ORDER = ["gpt-neo", "alpaca", "pythia-6.9b", "gpt-3.5"]