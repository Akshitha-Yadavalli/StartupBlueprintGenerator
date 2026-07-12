from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

credentials = Credentials(
    url="https://au-syd.ml.cloud.ibm.com",
    api_key="OLvaYW3ViUfr3M9jOGfU7wGPS65OttkWEToMWeuWi23P"
)

model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=credentials,
    project_id="65b44493-11a6-4ccb-9448-e2a5103bb6ee"
)

prompt = """
You are an AI Startup Mentor.

A user has this idea:

AI-powered crop disease detection for farmers.

Give:
1. Startup Summary
2. Target Customers
3. Revenue Model
"""

response = model.generate_text(prompt=prompt)

print(response)