from cheaper_llm.model_cascading.model_collection import MODEL_COLLECTION
from cheaper_llm.model_cascading.model_collection import MODEL_ORDER

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
    else:
        raise NotImplementedError
        
        
            
class ModelCascading:
    def __init__(self, candidate_models=None):
        if candidate_models is None:
            self.candidate_models = MODEL_ORDER
        else:
            self.candidate_models = candidate_models

    def __call__(self, prompt):
        for model in self.candidate_models:
            response = get_model_response(model, prompt)
            response_score = calculate_score(prompt, response)
