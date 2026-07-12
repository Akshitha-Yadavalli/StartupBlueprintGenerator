import os
from dotenv import load_dotenv

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from rag import retrieve_context

load_dotenv()

API_KEY = os.getenv("IBM_API_KEY")
PROJECT_ID = os.getenv("IBM_PROJECT_ID")
URL = os.getenv("IBM_URL")
MODEL_ID = os.getenv("MODEL_ID")

credentials = Credentials(
    url=URL,
    api_key=API_KEY
)

model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    project_id=PROJECT_ID,
    params={
        "max_new_tokens": 1000,
        "temperature": 0.5
    }
)

def generate_blueprint(startup_idea):

    # Retrieve relevant content from your PDFs
    context = retrieve_context(startup_idea)

    prompt = f"""
You are an expert Startup Business Consultant.

Below is some knowledge retrieved from official startup documents.

==========================
{context}
==========================

Use the above information whenever it is relevant.

Generate a COMPLETE startup blueprint.

Startup Idea:
{startup_idea}

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

If government schemes, funding opportunities, Startup India benefits, legal requirements, or regulations are present in the retrieved documents, include them in your answer.

Generate detailed answers.
"""

    response = model.generate_text(prompt=prompt)

    return response