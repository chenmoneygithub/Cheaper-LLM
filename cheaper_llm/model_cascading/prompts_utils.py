def generate_scoring_prompt(prompt, response):
    fused_prompt = (
        "Please rate the response for given prompt in a scale of 1 "
        f"to 100, precision 0.1. Prompt: {prompt}. Response: {response}."
    )
    return fused_prompt
