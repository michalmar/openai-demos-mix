import streamlit as st
import os
import json
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI

from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")

# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# os.environ["OPENAI_API_BASE"] = os.environ['AZURE_OPENAI_ENDPOINT']
# os.environ["OPENAI_API_KEY"] = os.environ['AZURE_OPENAI_API_KEY']
# os.environ["OPENAI_MODEL"] = os.environ['AZURE_OPENAI_MODEL_NAME']
MODEL = os.environ['AZURE_OPENAI_MODEL_NAME']



if "db" not in st.session_state:
    st.session_state.db = None

if "info" not in st.session_state:
    st.session_state.info = None
#################################################################################
# App elements

st.set_page_config(layout="wide")
st.title("DALL-E 3 Demo")
st.caption(f"powered by Azure OpenAI's {MODEL} model")

# Accept user input
if prompt := st.text_input("What you want to see? (ex. 'A cat made of pizza')"):
    if prompt:

        client = AzureOpenAI(
            api_version="2023-12-01-preview",
            azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
        )

        result = client.images.generate(
            model="dall-e-3", # the name of your DALL-E 3 deployment
            prompt=prompt,
            n=1
        )

        image_url = json.loads(result.model_dump_json())['data'][0]['url']

        st.image(image_url, width=300, caption=f"Image generated from prompt: {prompt}", use_column_width=True)

