import os
import streamlit as st

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from rag import retrieve_context

# -----------------------------
# Load credentials
# -----------------------------

try:
    # Streamlit Cloud
    API_KEY = st.secrets["IBM_API_KEY"]
    PROJECT_ID = st.secrets["IBM_PROJECT_ID"]
    URL = st.secrets["IBM_URL"]
    MODEL_ID = st.secrets["MODEL_ID"]

except Exception:
    # Local Development
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    API_KEY = os.getenv("IBM_API_KEY")
    PROJECT_ID = os.getenv("IBM_PROJECT_ID")
    URL = os.getenv("IBM_URL")
    MODEL_ID = os.getenv("MODEL_ID")

# -----------------------------
# Validate credentials
# -----------------------------

if not API_KEY:
    raise Exception("IBM_API_KEY not found.")

if not PROJECT_ID:
    raise Exception("IBM_PROJECT_ID not found.")

if not URL:
    raise Exception("IBM_URL not found.")

if not MODEL_ID:
    raise Exception("MODEL_ID not found.")

# -----------------------------
# IBM Watsonx Credentials
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
# Generate Startup Blueprint
# -----------------------------

def generate_blueprint(startup_idea):

    context = retrieve_context(startup_idea)

    prompt = f"""
You are an expert Startup Business Consultant.

Use the following retrieved knowledge while generating the answer.

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

Provide detailed and well-structured answers.
"""

    response = model.generate(
        prompt=prompt,
        params={
            "temperature": 0.5,
            "max_new_tokens": 1000
        }
    )

    return response["results"][0]["generated_text"]