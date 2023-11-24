import streamlit as st
import os

if "info" not in st.session_state:
    st.session_state.info = None
#################################################################################
# App elements

st.set_page_config(layout="wide")
st.title("Azure OpenAI Demos")

# # read the README.md file and display as markdown
# with open("readme.md", "r") as f:
#     st.markdown(f.read())

text = '''
This is a demo showcase of using Azure OpenAI API in simple web app.

Supported scenarios & APIs:

- [Chat](./ChatGPT) simple ChatGPT-like app where you can modfify settings such as `Temperature`, `Model`, `System message`.
- [Chat with file sources](./ChatGPT-LangChain) simple ChatGPT-like where you can add your own file (`.txt`, `.md`)
- [Dall-e V3.0](./Dall-e_3.0) simple image generation application using newest DALL-E 3.0 model
'''

st.markdown(text)