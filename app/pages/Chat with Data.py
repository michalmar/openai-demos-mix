import streamlit as st
import os
import json
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI

from jinja2 import Environment, FileSystemLoader
from promptflow.connections import AzureOpenAIConnection
from promptflow.connections import CognitiveSearchConnection

from rag_utils.RetrieveDocuments import search
from rag_utils.FormatRetrievedDocuments import format_retrieved_documents

from rag_utils.convert_template import convert_jinja_to_messages

from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")

# MODEL = os.environ['AZURE_OPENAI_MODEL_NAME']
# MODEL = "gpt-35-turbo"

SYSTEM_DEFAULT_PROMPT = "Assistant is a large language model trained by OpenAI."


if "info" not in st.session_state:
    st.session_state.info = None
if "SYSTEM_PROMPT" not in st.session_state:
    st.session_state.SYSTEM_PROMPT = SYSTEM_DEFAULT_PROMPT
if "messages" not in st.session_state:
    st.session_state.messages = [
                    {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                ]
if "model" not in st.session_state:
    st.session_state.model = "gpt-35-turbo"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 600

if "retreived_docs" not in st.session_state:
    st.session_state.retreived_docs = None




#################################################################################
# App elements


st.title("ChatGPT Demo on Your data")
st.warning("This demo uses the RAG model to chat with your data. Assuming the data has been **already** uploaded and indexed in Ai Search (setup previously).")
st.caption(f'Currently using index: **{os.environ["AZURE_SEARCH_INDEX"]}** from Azure AI Search {os.environ["AZURE_SEARCH_ENDPOINT"]}')

with st.sidebar:
    st.caption("Settings")
    st.session_state.model = st.selectbox("Select a model", ["gpt-35-turbo", "gpt-35-turbo-16k","gpt-4", "gpt-4-turbo"])
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
    # st.session_state.max_tokens = st.slider("Max tokens", 100, 4000, 600, 50)
        
    # st.text_area("Enter your SYSTEM message", key="system_custom_prompt", value=st.session_state.SYSTEM_PROMPT)
    # if st.button("Apply & Clear Memory"):
    #     # save the text from the text_area to SYSTEM_PROMPT
    #     st.session_state.SYSTEM_PROMPT = st.session_state.system_custom_prompt
    #     st.session_state.messages = [
    #                     {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
    #                 ]
    st.caption("Refresh the page to reset to default settings")

    

# st.caption(f"powered by Azure OpenAI's {st.session_state.model} model")
# st.caption(f"powered by Azure OpenAI's {MODEL} model")


search_index_name = os.environ["AZURE_SEARCH_INDEX"]; # Add your Azure AI Search index name here
search_conn = CognitiveSearchConnection(api_key=os.environ["AZURE_SEARCH_KEY"], api_base=os.environ["AZURE_SEARCH_ENDPOINT"])
embedding_model = "text-embedding-ada-002"
embedding_conn = AzureOpenAIConnection(api_base=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"])

# performs search over documents and returns top X results
def do_rag(query):
    ret = search(queries = query, 
        searchConnection =  search_conn, 
        indexName = search_index_name, 
        queryType = "vector", 
        topK = 3, 
        semanticConfiguration = None, 
        vectorFields = "contentVector", 
        embeddingModelConnection = embedding_conn, 
        embeddingModelName = embedding_model)
    # st.session_state.retreived_docs = ret
    return ret


def get_last_user_question():
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            return message["content"]
    return None

from rag_utils.ExtractIntent import extract_intent

# display chat messages
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

            message_placeholder.markdown("Formulating query...")
            
            # extract intent from user input
            # environment = Environment(loader=FileSystemLoader("./"))
            # template = environment.get_template(os.path.join("rag_utils","DetermineIntent.jinja2"))
            # context = {
            #     "query": prompt,
            #     "chat_history": None
            # }
            # jinja_prompt_template = template.render(context)

            # # print the prompt preserving the newlines on standard output
            # for x in jinja_prompt_template.split("\n"):
            #     print(x)


            # msgs = convert_jinja_to_messages(jinja_prompt_template)

            # client = AzureOpenAI(
            #     api_version="2023-05-15",
            #     azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
            #     api_key=os.environ["AZURE_OPENAI_API_KEY"],
            # ) 
            
            # response = client.chat.completions.create(
            #     model = st.session_state.model ,
            #     # messages=st.session_state.messages,
            #     messages=msgs,
            #     stream=False,
            #     temperature=0,
            #     max_tokens=50,
            # )

            # retrieved_intents = response.choices[0].message.content
            # intent = extract_intent(retrieved_intents, prompt)


            doc_query = prompt

            message_placeholder.markdown("Retrieving documents...")

            ret = do_rag(str(doc_query))
            RETRIEVAL_MAX_TOKENS = 50000
            
            message_placeholder.markdown("Formating documents...")
            docs = format_retrieved_documents(ret, RETRIEVAL_MAX_TOKENS)

            message_placeholder.markdown("Formulating the prompt...")  

            environment = Environment(loader=FileSystemLoader("./"))
            template = environment.get_template(os.path.join("rag_utils","DetermineReply.jinja2"))

            # context = {
            #     "query": "Hi, how are you doing?",
            #     "chat_history": None
            # }
            
            # convert current messages to conversation string
            conversation = ""
            for message in st.session_state.messages:
                if message["role"] == "system":
                    pass
                else:
                    conversation += f"{message['role']}\n\n{message['content']}\n"

            context = {
                "conversation": conversation,
                "documentation": docs,
                "user_query": prompt
            }

            jinja_prompt_template = template.render(context)

            # # print the prompt preserving the newlines on standard output
            # for x in jinja_prompt_template.split("\n"):
            #     print(x)


            msgs = convert_jinja_to_messages(jinja_prompt_template)

            # read prompt and add to chat history
            

            # st.session_state.messages.append({"role": "user", "content": prompt})

            message_placeholder.markdown("Generating response...")  
            client = AzureOpenAI(
                api_version="2023-05-15",
                azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
            ) 
            
            response = client.chat.completions.create(
                model = st.session_state.model ,
                # messages=st.session_state.messages,
                messages=msgs,
                stream=True,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
            
            for part in response:
                full_response += part.choices[0].delta.content or ""
                message_placeholder.markdown(full_response + "▌")

            # final response
            message_placeholder.markdown(full_response)

            if ret is not None:
                with st.expander(f"Retrieved docs for {prompt}", expanded=False):
                    # key:id
                    # key:url
                    # key:filepath
                    # key:content
                    # key:title
                    # key:chunk_id
                    # key:search_score
                    for index, doc in enumerate(ret):
                        with st.container():
                            st.markdown(f"[doc{index}] - {doc['title']}") 
                            st.caption(f"Score: {doc['search_score']}, filepath: {doc['filepath']}")

            # add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
             
    


