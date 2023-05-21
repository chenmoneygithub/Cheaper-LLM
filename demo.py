import cheaper_llm

candidate_models = ["text-babbage-001", "gpt-3.5-turbo"]

model_cascading = cheaper_llm.model_cascading.ModelCascading(candidate_models)

print(
    model_cascading(
        "What would be a good company name for a company that makes colorful socks?"
    )
)
