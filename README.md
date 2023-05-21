# Cheaper-LLM

## Roadmap

### Cache
Don't fire any queries if we can find a similar one. 

### Model Cascading
Pick up the right model from our model collection.


## Quickstart

```shell
pip install -q git+https://github.com/chenmoneygithub/Cheaper-LLM.git
```

```python
import cheaper_llm

candidate_models = ["pythia-2.8b", "text-babbage-001", "gpt-3.5-turbo"]

model_cascading = cheaper_llm.model_cascading.ModelCascading(candidate_models)

print(model_cascading("I like hotpot because"))
print(model_cascading("Why did the chicken cross the kitchen?"))

```
