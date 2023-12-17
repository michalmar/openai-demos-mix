import streamlit as st
import os
import json
# from dotenv import load_dotenv
# load_dotenv("../credentials.env")


# os.environ["OPENAI_API_TYPE"] = "azure"
# os.environ["OPENAI_API_VERSION"] = "2023-05-15"
# # os.environ["OPENAI_API_BASE"] = os.environ['AZURE_OPENAI_ENDPOINT']
# # os.environ["OPENAI_API_KEY"] = os.environ['AZURE_OPENAI_API_KEY']
# # os.environ["OPENAI_MODEL"] = os.environ['AZURE_OPENAI_MODEL_NAME']
# MODEL = os.environ['AZURE_OPENAI_MODEL_NAME']

import openai


from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               MessagesPlaceholder,
                               SystemMessagePromptTemplate)

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document

from prompts import (CUSTOM_CHATBOT_PREFIX, WELCOME_MESSAGE)
from prompts import COMBINE_QUESTION_PROMPT, COMBINE_PROMPT
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
from collections import OrderedDict
import uuid
import markdownify


def process_file(file, chunk_size = 1000, chunk_overlap=100) -> str:
    '''
    Function to store the uploaded file in FAISS db.
    It stores the file in ./upload folder and then load it into FAISS db.

    Parameters:
    file: Attachment object with content_url and name
    '''
    file_folder = "./upload"
    content = file.read()
    doc = Document(page_content=content, metadata={"source": file.name})

    # # create folder if not exist
    # if not os.path.exists(file_folder):
    #     os.makedirs(file_folder)
    # file_path = os.path.join(file_folder, file.name)
    # open(file_path, "wb").write(content)

    
    # load the file int FAISS db
    # chunk_size = 1000
    # loader = TextLoader(file_path)
    # documents = loader.load()
    documents = [doc]
    # text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 2000,
        chunk_overlap  = 200,
        length_function = len,
        is_separator_regex = False,
        )
    docs = text_splitter.split_documents(documents)

    # embeddings = OpenAIEmbeddings()
    embeddings = AzureOpenAIEmbeddings()
    warning_msg = ""
    if (len(docs) > 16):
        warning_msg = f"\U0001F6D1 Only the first 16 chunks will be loaded (got {len(docs) } chunks for chunk size = {chunk_size})"
        docs = docs[:16]
    
    st.session_state.db = FAISS.from_documents(docs, embeddings)

    return f"&nbsp;\U00002705&nbsp;&nbsp;your chat is now grounded by '{file.name}' file!\n{warning_msg}"

def get_search_results(query: str, indexes: list, 
                    k: int = 5,
                    reranker_threshold: int = 1,
                    sas_token: str = "",
                    vector_search: bool = False,
                    similarity_k: int = 3, 
                    query_vector: list = []) -> List[dict]:

    # headers = {'Content-Type': 'application/json','api-key': os.environ["AZURE_SEARCH_KEY"]}
    # params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    agg_search_results = dict()
    
    for index in indexes:
        # search_payload = {
        #     "search": query,
        #     "queryType": "semantic",
        #     "semanticConfiguration": "my-semantic-config",
        #     "count": "true",
        #     "speller": "lexicon",
        #     "queryLanguage": "en-us",
        #     "captions": "extractive",
        #     "answers": "extractive",
        #     "top": k
        # }
        # if vector_search:
        #     search_payload["vectors"]= [{"value": query_vector, "fields": "chunkVector","k": k}]
        #     search_payload["select"]= "id, title, chunk, name, location"
        # else:
        #     search_payload["select"]= "id, title, chunks, language, name, location, vectorized"
        

        # resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
        #                 data=json.dumps(search_payload), headers=headers, params=params)

        # search_results = resp.json()
        docs = st.session_state.db.similarity_search_with_score(query)
        agg_search_results[index] = docs
    
    content = dict()
    ordered_content = OrderedDict()
    
    for index,search_results in agg_search_results.items():
        for doc in search_results:
            result = doc[0] # Document object
            relevance_score = doc[1] # Relevance score
            if relevance_score > reranker_threshold: # Show results that are at least N% of the max possible score=4
                tmp_id = generate_doc_id()
                content[tmp_id]={
                                        "title": result.metadata["source"], # result['title'], 
                                        "name": result.metadata["source"], # result['name'], 
                                        "location": "none", # result['location'] + sas_token if result['location'] else "",
                                        "caption": "none", # result['@search.captions'][0]['text'],
                                        "index": index
                                    }
                content[tmp_id]["chunk"]= result.page_content #result['chunk']
                content[tmp_id]["score"]= relevance_score # Uses the reranker score

                # if vector_search:
                #     content[tmp_id]["chunk"]= result['chunk']
                #     content[tmp_id]["score"]= result['@search.score'] # Uses the Hybrid RRF score
            
                # else:
                #     content[tmp_id]["chunks"]= result['chunks']
                #     content[tmp_id]["language"]= result['language']
                #     content[tmp_id]["score"]= relevance_score # Uses the reranker score
                #     content[tmp_id]["vectorized"]= result['vectorized']
                
    # After results have been filtered, sort and add the top k to the ordered_content
    if vector_search:
        topk = similarity_k
    else:
        topk = k*len(indexes)
        
    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding 5 results
            break

    return ordered_content

# currently a dummy function returns a random uuid
def generate_index():
    return str(uuid.uuid4())    

# currently a dummy function returns a random uuid
def generate_doc_id():
    return str(uuid.uuid4())

# format the response (convert html to markdown)
def format_response(response):
    # return re.sub(r"(\n\s*)+\n+", "\n\n", response).strip()

    # convert html tags to markdown
    response = markdownify.markdownify(response, heading_style="ATX")
    # response = response.replace("[[", " [").replace("]]", "]")
    response = response.replace("[[", " [[")

    return response.strip()
