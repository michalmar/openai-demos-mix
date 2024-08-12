import streamlit as st
import os

#################################################################################
# App elements

st.title("Azure OpenAI Demos")

# # read the README.md file and display as markdown
# with open("readme.md", "r") as f:
#     st.markdown(f.read())

text = '''
This is a demo showcase of using Azure OpenAI API in simple web app.

Supported scenarios & APIs:
- :star: [Chat with you Data in AI Search](./Chat_with_Data) Using existing index in AI Search to chat with your data
- **NEW** :star: [Chat with Assistants API](./ChatGPT_with_Assistants_API) Using existing index in AI Search to chat with your data
- :speech_balloon: [Chat](./ChatGPT) simple ChatGPT-like app where you can modify settings such as `Temperature`, `Model`, `System message`.
- :file_folder: [Chat with file sources](./ChatGPT-LangChain) simple ChatGPT-like where you can add your own file (`.txt`, `.md`)
- :frame_with_picture: [Dall-e V3.0](./Dall-e_3.0) simple image generation application using newest DALL-E 3.0 model
- :eye: [GPT-4 Vision](./GPT-X) Showcase of GPT-4 Vision API
- :telephone_receiver: [Functions Calling](./ChatGPT-Functions) simple showcase of Function Calling (plugins) from ChatGPT conversation
'''

st.markdown(text)

