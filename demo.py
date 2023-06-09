import cheaper_llm

candidate_models = ["pythia-2.8b", "text-babbage-001", "gpt-3.5-turbo"]

model_cascading = cheaper_llm.model_cascading.ModelCascading(candidate_models)

responses = []
responses.append(model_cascading("I like hotpot because"))
responses.append(model_cascading("Why did the chicken cross the kitchen?"))
responses.append(model_cascading(
        (
            "I am planning for my holiday party, please give "
            "me a plan. I would like the party to be of size 30, "
            "and in western style."
        )
    ))

for response in responses:
    print(response)
