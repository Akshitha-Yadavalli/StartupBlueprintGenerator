import os
import streamlit as st
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from rag import retrieve_context

# -----------------------------
# Load credentials
# -----------------------------

try:
    API_KEY = st.secrets["IBM_API_KEY"]
    PROJECT_ID = st.secrets["IBM_PROJECT_ID"]
    URL = st.secrets["IBM_URL"]
    MODEL_ID = st.secrets["MODEL_ID"]

except Exception:
    load_dotenv()

    API_KEY = os.getenv("IBM_API_KEY")
    PROJECT_ID = os.getenv("IBM_PROJECT_ID")
    URL = os.getenv("IBM_URL")
    MODEL_ID = os.getenv("MODEL_ID")

# -----------------------------
# Debug (remove later)
# -----------------------------

print("API_KEY Loaded:", API_KEY is not None)
print("PROJECT_ID:", PROJECT_ID)
print("URL:", URL)
print("MODEL_ID:", MODEL_ID)

if not API_KEY:
    raise ValueError("IBM_API_KEY is missing.")

if not PROJECT_ID:
    raise ValueError("IBM_PROJECT_ID is missing.")

if not URL:
    raise ValueError("IBM_URL is missing.")

if not MODEL_ID:
    raise ValueError("MODEL_ID is missing.")

# -----------------------------
# IBM Credentials
# -----------------------------

credentials = Credentials(
    url=URL,
    api_key=API_KEY
)

model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    project_id=PROJECT_ID
)

# -----------------------------
# Generate Blueprint
# -----------------------------

def generate_blueprint(startup_idea):

    context = retrieve_context(startup_idea)

    prompt = f"""
You are an expert Startup Business Consultant.

Use the following knowledge while answering.

Knowledge:
{context}

Startup Idea:
{startup_idea}

Generate a COMPLETE Startup Blueprint.

Include:

1. Executive Summary
2. Problem Statement
3. Proposed Solution
4. Target Customers
5. Market Analysis
6. Competitor Analysis
7. Business Model Canvas
8. Revenue Model
9. Estimated Budget
10. Funding Sources
11. Government Schemes
12. Legal Requirements
13. Go-to-Market Strategy
14. Technology Stack
15. Future Scope

Provide detailed answers.
"""

    response = model.generate(
        prompt=prompt,
        params={
            "temperature": 0.5,
            "max_new_tokens": 1000
        }
    )

    return response["results"][0]["generated_text"]