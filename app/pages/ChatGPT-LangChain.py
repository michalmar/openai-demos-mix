import streamlit as st
import os
import json
from openai import AzureOpenAI
import uuid

from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
MODEL = "gpt-35-turbo"

from langchain.chains import LLMChain
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)
from langchain.memory import ConversationBufferWindowMemory
from langchain.docstore.document import Document
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


def get_prompt(template):
    PROMPT = ChatPromptTemplate(
                    messages=[
                        SystemMessagePromptTemplate.from_template(
                            template
                        ),
                        # The `variable_name` here is what must align with memory
                        MessagesPlaceholder(variable_name="chat_history"),
                        HumanMessagePromptTemplate.from_template("{question}")
                    ]
                )
    return PROMPT

def ask_gpt(llm, QUESTION, session_id):
    chatgpt_chain = LLMChain(
                            llm=llm,
                            prompt=get_prompt(st.session_state.SYSTEM_PROMPT),
                            # prompt=QUESTION,
                            verbose=False,
                            memory=st.session_state.memory_dict[session_id]
                            )
    answer = chatgpt_chain.run(QUESTION)                        
    return answer

from chat_utils import get_search_results, process_file, generate_index, generate_doc_id, format_response

def ask_gpt_with_sources(llm, QUESTION, session_id):
    # remove the /file prefix
    # QUESTION = self.QUESTION[5:].strip()
    
    # query = "What did the president say about Ketanji Brown Jackson"
    # docs = self.db.similarity_search_with_score(query)
    vector_indexes = [generate_index()]

    ordered_results = get_search_results(QUESTION, vector_indexes, 
                                            k=6,
                                            reranker_threshold=0.1, #1
                                            vector_search=True, 
                                            similarity_k=6,
                                            #query_vector = embedder.embed_query(QUESTION)
                                            query_vector= []
                                            )
    # COMPLETION_TOKENS = 1000
    # llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.5, max_tokens=COMPLETION_TOKENS)

    top_docs = []
    for key,value in ordered_results.items():
        location = value["location"] if value["location"] is not None else ""
        # top_docs.append(Document(page_content=value["chunk"], metadata={"source": location+os.environ['BLOB_SAS_TOKEN']}))
        top_docs.append(Document(page_content=value["chunk"], metadata={"source": value["name"]}))
            
        print("Number of chunks:",len(top_docs))

    chain_type = "stuff"
    
    if chain_type == "stuff":
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                        prompt=COMBINE_PROMPT)
    elif chain_type == "map_reduce":
        chain = load_qa_with_sources_chain(llm, chain_type=chain_type, 
                                        question_prompt=COMBINE_QUESTION_PROMPT,
                                        combine_prompt=COMBINE_PROMPT,
                                        return_intermediate_steps=True)


    response = chain({"input_documents": top_docs, "question": QUESTION, "language": st.session_state.language})
    text_output = format_response(response['output_text'])
    return text_output


#################################################################################
# Session state

if "SYSTEM_PROMPT" not in st.session_state:
    st.session_state.SYSTEM_PROMPT = CUSTOM_CHATBOT_PREFIX

if "memory_dict" not in st.session_state:
    st.session_state.memory_dict = {}

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.memory_dict[st.session_state.session_id] = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True, k=3)

if "info" not in st.session_state:
    st.session_state.info = None

if "db" not in st.session_state:
    st.session_state.db = None

if "model" not in st.session_state:
    st.session_state.model = "gpt-35-turbo"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5

if "language" not in st.session_state:
    st.session_state.language = "English"

if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 500

#################################################################################
# App elements

st.title("ChatGPT Demo with LangChain (with sources)")

with st.sidebar:
    st.caption("Settings")
    st.session_state.model = st.selectbox("Select a model", ["gpt-35-turbo", "gpt-35-turbo-16k","gpt-4", "gpt-4-turbo"])
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
    st.session_state.max_tokens = st.slider("Max tokens", 100, 4000, 500, 5)
    st.session_state.language = st.selectbox("Select a language", ["English", "Spanish", "Czech"])
    
    if st.button("New Conversation"):
        st.session_state.session_id = uuid.uuid4().hex
        st.session_state.memory_dict[st.session_state.session_id] = ConversationBufferWindowMemory(memory_key="chat_history",input_key="question", return_messages=True, k=3)
        st.session_state.info = "Session restarted"
        st.session_state.db = None
    st.caption(f"I am using **{st.session_state.model}** model")
    st.caption(f"session: {st.session_state.session_id}")


    # with st.expander("Settings"):
    # add upload button
    uploaded_file = st.file_uploader("Upload a file to ground your answers", type=["txt", "md"])
    if uploaded_file is not None and  (st.session_state.db is None):
        # store the uploaded file on disk
        msg = process_file(uploaded_file)
        st.warning(msg)
        st.session_state.info = msg
    
    st.text_area("Enter your SYSTEM message", key="system_custom_prompt", value=CUSTOM_CHATBOT_PREFIX)

    # create a save button
    if st.button("Save"):
        # save the text from the text_area to SYSTEM_PROMPT
        st.session_state.SYSTEM_PROMPT = st.session_state.system_custom_prompt
        # delete memory / restart sesion
        st.session_state.memory_dict[st.session_state.session_id].clear()

if st.session_state.info is not None:
    st.info(st.session_state.info)
    st.session_state.info = None
st.caption(f"powered by Azure OpenAI's {st.session_state.model} model")

with st.container():
    # display messages from memory
    memory = st.session_state.memory_dict[st.session_state.session_id].load_memory_variables({})
    for message in memory["chat_history"]:
        # if message typ
        with st.chat_message(message.type):
            st.markdown(message.content)

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        if prompt:
            
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                stream_handler = StreamHandler(st.empty())
                llm = AzureChatOpenAI(deployment_name=st.session_state.model, temperature=st.session_state.temperature, max_tokens=st.session_state.max_tokens, streaming=True, callbacks=[stream_handler])
                
                # check if db is loaded - if so, use the qa_with_sources chain
                if (st.session_state.db is not None):
                    output = ask_gpt_with_sources(llm, prompt, st.session_state.session_id)
                else:
                    # Get response from GPT
                    output = ask_gpt(llm, prompt, st.session_state.session_id)       
                
                stream_handler.container.markdown(output, unsafe_allow_html=True)
    

