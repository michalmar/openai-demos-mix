import streamlit as st
import os
import json
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI

from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")


import doc_utils as doc_utils
import rag_utils as rag_utils
import re



if "info" not in st.session_state:
    st.session_state.info = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"


# this is used for storing the original search phrase
if "search_phrase" not in st.session_state:
    st.session_state.search_phrase = None    

# this is  used for filling the prompty template
if "docs_string" not in st.session_state:
    st.session_state.docs_string = None

# this holds the actual list of docuemnts - in list of dictionaries format
if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

AZURE_OPENAI_MODEL_NAME_LIST = os.getenv("AZURE_OPENAI_MODEL_NAME_LIST").split(",")

#################################################################################
# App elements

st.title("Search assistant Demo")

def extract_article(full_response, docs):
    match = re.search(r'###(.*?)###', full_response)
    if match:
        extracted_content = match.group(1)
        for doc in docs:
            if extracted_content in doc['header']:
                return (full_response, doc)
    else:
        return (full_response, None)
    return (None, None)

with st.sidebar:
    st.caption("Settings")

    st.session_state.model = st.selectbox("Select a model", AZURE_OPENAI_MODEL_NAME_LIST)
    
    if st.button("Apply & Clear Memory"):
        # save the text from the text_area to SYSTEM_PROMPT
        st.session_state.messages = [
                        # {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                    ]
    st.caption("Refresh the page to reset to default settings")

st.caption(f"powered by Azure OpenAI's {st.session_state.model} model")

# checkbox to indicate whether to use assistance in search
use_assistance = st.checkbox("Use assistance in search", value=False)

if not st.session_state.search_phrase:

    if search_phrase := st.text_input("What are you looking for?"):
        if search_phrase:
            # Display user message in chat message container
            st.write("User: " + search_phrase)
            with st.spinner("Searching for documents..."):
                docs = rag_utils.get_documents(search_query=search_phrase, num_docs=10)
                _docs = []
                _docs_string = ""
                # reformat the docs to be more readable
                for doc in docs:
                    for k,v in doc.items():
                        _docs.append({
                            "header": v["title"],
                            "description": v["content"],
                            "url": v["url"],
                            # "id": k
                        })
                        _docs_string += f"Header: {v['title']}\nDescription: {v['content']}\n\n"
                docs = _docs


            if use_assistance:
                st.session_state.search_phrase = search_phrase
                st.session_state.docs_string = _docs_string
                st.session_state.docs_list = docs
                st.rerun()
                
            else:
                # Display documents as a nice list with title and content
                for doc in docs:
                    with st.container(border=True):
                        st.markdown(f"[{doc['header']}]({doc['url']})")
                        st.caption(f"{doc['description']}")

        
else:
    import search_utils

    for message in st.session_state.messages:
        if message["role"] == "system":
            pass
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Type your answer?"):
        # subsequent user input - refining search term / guided by assistant
        if prompt:
            _display_history_message = prompt
            _chat_input = prompt
    else:
        # FIRST TIME -> NO USER INPUT, ONLY ORIGINAL SEARCH QUERY
        _display_history_message = st.session_state.search_phrase
        _chat_input = ""
    
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(_display_history_message)
        st.session_state.messages.append({"role": "user", "content": _display_history_message})
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # get all items in the chat history but last one
        response = search_utils.get_search_assistant_response(chat_input = _chat_input, 
                                                                chat_history=st.session_state.messages[:-1], 
                                                                results=st.session_state.docs_string, 
                                                                original_search_phrase=st.session_state.search_phrase,
                                                                azure_deployment=st.session_state.model,
                                                                )
        
        for part in response["reply"]:
            # full_response += part.choices[0].delta.content or ""
            full_response += part or ""
            message_placeholder.markdown(full_response + "▌")

        full_response, selected_doc = extract_article(full_response, st.session_state.docs_list)
        if selected_doc:
            message_placeholder.markdown(f"Your go-to article is: [{selected_doc['header']}](https://www.example.com)")
        else:
            message_placeholder.markdown(full_response)
        # add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

