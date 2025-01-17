import streamlit as st
import os
import json
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI

from prompts import PROMPTS_SYSTEM_LIST

from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")


import doc_utils as doc_utils

import rag_utils as rag_utils



# MODEL = os.environ['AZURE_OPENAI_MODEL_NAME']
# MODEL = "gpt-35-turbo"

SYSTEM_DEFAULT_PROMPT = "Assistant is a large language model trained by OpenAI."



# Store the initial value of the text area in session state if not already stored
if "initial_system_prompt" not in st.session_state:
    st.session_state.initial_system_prompt = PROMPTS_SYSTEM_LIST[list(PROMPTS_SYSTEM_LIST.keys())[0]]

if "system_custom_prompt" not in st.session_state:
    st.session_state.system_custom_prompt = st.session_state.initial_system_prompt

if "info" not in st.session_state:
    st.session_state.info = None
if "SYSTEM_PROMPT" not in st.session_state:
    st.session_state.SYSTEM_PROMPT = SYSTEM_DEFAULT_PROMPT
if "messages" not in st.session_state:
    st.session_state.messages = [
                    {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                ]
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 200

if "index_filled" not in st.session_state:
    st.session_state.index_filled = False


AZURE_OPENAI_MODEL_NAME_LIST = os.getenv("AZURE_OPENAI_MODEL_NAME_LIST").split(",")

#################################################################################
# App elements

st.title("ChatGPT Demo")

def update_prompt():
    st.session_state.system_custom_prompt = PROMPTS_SYSTEM_LIST[st.session_state.selected_option]


with st.sidebar:
    st.caption("Settings")

    # add check box to recreate the index
    st.checkbox("Recreate index", key="recreate_index")

    # add checkbox to use Semantic chunking
    st.checkbox("Use Semantic Chunking", key="use_semantic_chunking")

    # upload a file
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx", "pptx"], accept_multiple_files=True)
    status_placeholder = st.empty()
    if uploaded_file and not st.session_state.index_filled:
        with st.spinner(f"Uploading & chunking {'(semantically)' if st.session_state.use_semantic_chunking else ''}..."):

            if st.session_state.recreate_index:
                res = doc_utils.create_index(recreate=True)
                status_placeholder.success(f"Index {res.name} re-created succesfully.")

            # st.write("File uploaded")
            # st.write(uploaded_file)
            # paths = [f["upload_url"] for f in uploaded_file]
            # st.write(paths)
            num_docs, num_chunks = doc_utils.process_and_upload_files(uploaded_file, 10, use_semantic_chunking=st.session_state.use_semantic_chunking)
            status_placeholder.success(f"uploaded: {num_docs} docs in {num_chunks} chunks")
            st.session_state.index_filled = True


    st.session_state.model = st.selectbox("Select a model", AZURE_OPENAI_MODEL_NAME_LIST)
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
    st.session_state.max_tokens = st.slider("Max tokens", 10, 4000, 400, 5)


    # Display the text area with the initial value
    st.text_area(
        "Enter your SYSTEM message", 
        key="system_custom_prompt", 
        value=st.session_state.initial_system_prompt
    )

    if st.button("Apply & Clear Memory"):
        # save the text from the text_area to SYSTEM_PROMPT
        st.session_state.SYSTEM_PROMPT = st.session_state.system_custom_prompt
        st.session_state.messages = [
                        {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                    ]
    st.caption("Refresh the page to reset to default settings")

    

st.caption(f"powered by Azure OpenAI's {st.session_state.model} model")
# st.caption(f"powered by Azure OpenAI's {MODEL} model")

for message in st.session_state.messages:
    if message["role"] == "system":
        pass
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What is up?"):
    if prompt:
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            response = rag_utils.get_chat_response(chat_input = prompt, chat_history=st.session_state.messages[-1])
            
            for part in response["reply"]:
                # full_response += part.choices[0].delta.content or ""
                full_response += part or ""
                message_placeholder.markdown(full_response + "▌")

            # final response
            # message_placeholder.markdown(full_response)
            message_placeholder = st.expander("Citations", expanded=False)
            # message_placeholder.write(full_response)
            for doc in response["context"]:
                # message_placeholder.write(doc) # TODO: format the doc to be more readable
                for k,v in doc.items():
                    # icon of file
                    
                    message_placeholder.markdown(f"📄 {k} [ {v['title']}](http://www.example.com)")
                    # message_placeholder.write(f"Document {k}: {v['title']}")
                    # message_placeholder.write(f"Content: {v['content']}")
                    # message_placeholder.write(f"URL: {v['url']}")

                



            # add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
             
    
