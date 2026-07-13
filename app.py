import streamlit as st
from agent import generate_blueprint

st.set_page_config(
    page_title="Startup Blueprint Generator",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Startup Blueprint Generator Agent")

st.write(
    "Generate a complete startup business plan using IBM watsonx.ai."
)

idea = st.text_area(
    "Enter your Startup Idea",
    height=180,
    placeholder="Example: AI-powered crop disease detection for farmers"
)

if st.button("Generate Blueprint"):

    if idea.strip() == "":
        st.warning("Please enter a startup idea.")

    else:

        with st.spinner("Generating Blueprint..."):

            try:
                result = generate_blueprint(idea)

                st.success("Blueprint Generated Successfully!")

                st.markdown(result)

            except Exception as e:
                st.error(f"Error: {e}")