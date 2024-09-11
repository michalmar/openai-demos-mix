
# %%
import os
import json
from pathlib import Path


from dotenv import load_dotenv
load_dotenv("../credentials.env")

# Set the environment variables
AZURE_OPENAI_ENDPOINT= os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_CHAT_DEPLOYMENT= os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_API_VERSION= os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_API_KEY= os.getenv("AZURE_OPENAI_API_KEY")

AZURE_OPENAI_EMBEDDING_ENDPOINT= os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT= os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZUREAI_SEARCH_INDEX_NAME= os.getenv("AZUREAI_SEARCH_INDEX_NAME")
AZURE_SEARCH_KEY= os.getenv("AZURE_SEARCH_KEY")

import os
# set environment variables before importing any other code
# from dotenv import load_dotenv
# load_dotenv()

from pathlib import Path
from typing import TypedDict
from openai import AzureOpenAI

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.tracing import trace

class ChatResponse(TypedDict):
    context: dict
    reply: str

@trace
# def get_chat_response(chat_input: str, chat_history: list = []) -> ChatResponse:
def get_chat_response(chat_input: str, chat_history: list = []):
    model_config = AzureOpenAIModelConfiguration(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )

    searchQuery = chat_input

    # Only extract intent if there is chat_history
    if len(chat_history) > 0:
        # extract current query intent given chat_history
        path_to_prompty = f"{Path(__file__).parent.absolute().as_posix()}/queryIntent.prompty" # pass absolute file path to prompty
        print(path_to_prompty)
        intentPrompty = Prompty.load(path_to_prompty, model={
            'configuration': model_config,
            'parameters': { 
                'max_tokens': 256,
            }
        })
        searchQuery = intentPrompty(query=chat_input, chat_history=chat_history)

    # retrieve relevant documents and context given chat_history and current user query (chat_input)
    documents = get_documents(searchQuery, 3)

    # send query + document context to chat completion for a response
    path_to_prompty = f"{Path(__file__).parent.absolute().as_posix()}/chat.prompty"
    chatPrompty = Prompty.load(path_to_prompty, model={
        'configuration': model_config,
        'parameters': { 
            'max_tokens': 3000,
            'temperature': 0.2,
            'stream': True # always stream responses, consumers/clients should handle streamed response
        }
    })
    result = chatPrompty(
        chat_history=chat_history,
        chat_input=chat_input,
        documents=documents
    )
    # for item in result:
    #     print(item, end="")

    return dict(reply=result, context=documents)
    # return result, documents

@trace
def get_documents(search_query: str, num_docs=3):

    index_name = os.environ["AZUREAI_SEARCH_INDEX_NAME"]


    #  retrieve documents relevant to the user's question from Cognitive Search
    search_credential = AzureKeyCredential(os.environ["AZURE_SEARCH_KEY"])
    search_client = SearchClient(
        endpoint=os.environ["AZURE_SEARCH_ENDPOINT"],
        # credential=DefaultAzureCredential(),
        credential=search_credential,
        index_name=index_name
        )

    aoai_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
        # azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"),
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"]
    )

    # generate a vector embedding of the user's question
    embedding = aoai_client.embeddings.create(
        input=search_query,
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
        )
    embedding_to_query = embedding.data[0].embedding

    context = ""
    # use the vector embedding to do a vector search on the index
    vector_query = VectorizedQuery(vector=embedding_to_query, k_nearest_neighbors=num_docs, fields="content_vector")
    results = trace(search_client.search)(
        search_text="",
        vector_queries=[vector_query],
        select=["id", "content","publicurl","title"])

    # context = {"retrieved_documents": []}
    context = []
    for idx, result in enumerate(results):
        # context["retrieved_documents"].append(
        context.append(
             {
                 f"[doc{idx}]": {
                "chunk_id": result["id"],
                "content": result["content"],
                "url":result["publicurl"],
                "title":result["title"]
                }
            }
        )
        # context += f'\n [Doc{idx}]: {result["id"]}\n "content": "{result["content"]}"'
        # context += f"\n [doc{idx}]: {result['content']}\n"

    return context

# %%
# import copilot
from IPython.display import Markdown
from promptflow.tracing import start_trace, trace

if __name__ == "__main__":
    start_trace()

    # response = copilot.get_chat_response("What types of tickets do you provide?")
    response = get_chat_response("What types of tickets do you provide? ANd where can I buy them?")


    full_response = ""
    for part in response["reply"]:
        full_response += part or ""
        print(part, end="")

    for doc in response["context"]:
        print(doc)

# path_to_prompty = f"{Path(__file__).parent.absolute().as_posix()}/chat.prompty"
# chatPrompty = Prompty.load(path_to_prompty, model={
#     'configuration': model_config,
#     'parameters': { 
#         'max_tokens': 3000,
#         'temperature': 0.2,
#         'stream': False # always stream responses, consumers/clients should handle streamed response
#     }
# })
# messages = chatPrompty.render()

# messages = eval(messages)
# messages.append({"role": "assistant", "content": full_response})


# response = get_chat_response(chat_input="Where can I buy yellow ticket?", chat_history=messages)


# full_response = ""
# for part in response["reply"]:
#     full_response += part or ""
#     print(part, end="")


