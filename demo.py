import cheaper_llm

candidate_models = ["pythia-6.9b", "text-babbage-001", "gpt-3.5-turbo"]
# candidate_models = ["pythia-6.9b"]

model_cascading = cheaper_llm.model_cascading.ModelCascading(candidate_models)

print(
    model_cascading(
        "What would be a good company name for a company that makes colorful socks?"
    )
)
