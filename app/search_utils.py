import os
import json
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI
from pathlib import Path
from typing import TypedDict


from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")


import doc_utils as doc_utils
import rag_utils as rag_utils
from doc_utils import generate_embedding

import base64
import random
import time


#################################################################################
# App elements

from doc_utils import create_index

from promptflow.core import Prompty, AzureOpenAIModelConfiguration
from promptflow.tracing import trace

@trace
# def get_chat_response(chat_input: str, chat_history: list = []) -> ChatResponse:
def get_search_assistant_response(
    original_search_phrase: str, 
    results: list = [], 
    chat_input: str = "", 
    chat_history: list = [],
    azure_deployment: str = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
    api_version: str = os.environ["AZURE_OPENAI_API_VERSION"],
    azure_endpoint: str = os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key: str = os.environ["AZURE_OPENAI_API_KEY"]
    ) -> dict:

    model_config = AzureOpenAIModelConfiguration(
        azure_deployment=azure_deployment,
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )

    # send query + document context to chat completion for a response
    path_to_prompty = f"{Path(__file__).parent.absolute().as_posix()}/search.prompty"
    chatPrompty = Prompty.load(path_to_prompty, model={
        'configuration': model_config,
        'parameters': { 
            'max_tokens': 1000,
            'temperature': 0.2,
            'stream': True # always stream responses, consumers/clients should handle streamed response
        }
    })

    result = chatPrompty(
        chat_history=chat_history,
        chat_input=chat_input,
        results=results,
        original_search_phrase=original_search_phrase
    )

    return dict(reply=result, context=results)


def fill_search():
    create_index(recreate=True, schema="schema.json")

    chunks = [

        {	"Header": "Driver and Vehicle Licensing Agency",
	"Description": "We’re the Driver and Vehicle Licensing Agency (DVLA), holding more than 50 million driver records and more than 40 million vehicle records. We collect over £7 billion a year in Vehicle Excise Duty (VED). DVLA is an executive agency, sponsored...",
},	
{	"Header": "License, sell or market your copyright material",
	"Description": "Guidance for copyright owners on how to grant a licence for, sell or market their work.",
},	
{	"Header": "Driving licences",
	"Description": "Apply for, renew or update your licence, view or share your driving licence, add new categories",
},	
{	"Header": "Licensing intellectual property",
	"Description": "Intellectual property can be bought, sold or licensed.",
},	
{	"Header": "Apply for your first provisional driving licence",
	"Description": "Apply for your first provisional driving licence from DVLA online to drive a car, motorbike or moped.",
},	
{	"Header": "View or share your driving licence information",
	"Description": "Find out what information DVLA holds about your driving licence or create a check code to share your driving record, for example to hire a car",
},	
{	"Header": "Renew your driving licence",
	"Description": "Apply online to renew your 10-year driving licence, full or provisional - cost, payment methods, documents and information you need",
},	
{	"Header": "Licensing bodies and collective management organisations",
	"Description": "Licensing bodies and collective management organisations can agree licences with users on behalf of owners and collect any royalties the owners are owed.",
},	
{	"Header": "Check someone's driving licence information",
	"Description": "Check someone's driving record - vehicles they can drive, penalty points and disqualifications",
},	
{	"Header": "Standard Essential Patent licensing",
	"Description": "This guidance relates to Standard Essential Patents licensing.",
},	
{	"Header": "Entertainment Licensing",
	"Description": "Information on whether you need approval to put on certain types of regulated entertainment.",
},	
{	"Header": "Get a licence to play live or recorded music",
	"Description": "You usually need a licence from PPL PRS to play live or recorded music in public - includes playing background music at your business, and staging live music or theatre productions.",
},	
{	"Header": "Strategic export controls: licensing data",
	"Description": "Reports and data on export control licensing compiled by the export control organisation.",
},	
{	"Header": "Using somebody else's intellectual property",
	"Description": "Buy or get someone's permission to use a patent, trade mark, design or work under copyright",
},	
{	"Header": "Getting your full driving licence",
	"Description": "How to get your full driving licence once you've passed your driving test - driving test certificate, photocard, provisional, paper licences",
},	
{	"Header": "Driver licences for taxis and private hire vehicles",
	"Description": "How to apply for a taxi or private hire vehicle (PHV) driving licences inside and outside London",
},	
{	"Header": "Alcohol licensing",
	"Description": "Information on the different types of alcohol licences available and guidance on how to apply for them.",
},	
{	"Header": "License, mortgage, sell, change ownership and market your design",
	"Description": "Benefit from your design by licensing, mortgaging, selling, changing ownership and exploit by marketing.",
},	
{	"Header": "Goods vehicle operator licensing guide",
	"Description": "Overview of the vehicle operator licensing system.",
},	
{	"Header": "Industrial hemp licensing",
	"Description": "Information for prospective growers of low THC cannabis (industrial hemp), for the production of seed and fibre only.",
},	
    ]

    docs = []
    chunk_num = 0
    page_num = 0
    filename = "test"
    counter = 0
    for chunk in chunks:
        chunk_num += 1
        d = {
            "chunk_id" : filename + '_' + str(page_num).zfill(2) +  '_' + str(chunk_num).zfill(2),
            "document_id": filename + '_' + str(page_num).zfill(2) +  '_' + str(chunk_num).zfill(2),
            "content": chunk['Description'],       
            "title": chunk['Header']
            }
        # d["dateTime"] = "dummy date"

        counter += 1

        # TODO: this is not valid url since we are not spliting by pages
        public_url = '#page=' + str(page_num) 

        v_contentVector = generate_embedding(d["content"])

        # it may happen that semantic chunking returns None (too small chunk)
        if v_contentVector == None:
            print (f'Error generating vector for chunk {chunk_num} with content: {d["content"]}' )
            continue

        docs.append(
        {
                "id": base64.urlsafe_b64encode(bytes(d["chunk_id"], encoding='utf-8')).decode('utf-8'),
                "chunk_id": d["chunk_id"],
                "document_id": d["document_id"],
                "title": d["title"],
                "content": d["content"],
                "sourceurl": filename, 
                "publicurl": public_url,
                # "dateTime": d["dateTime"],
                # "Person": d["Person"],
                # "Location": d["Location"],
                # "Organization": d["Organization"],
                # "URL": d["URL"],
                # "Email": d["Email"],
                # "PersonType": d["PersonType"],
                # "Event": d["Event"],
                # "Quantity": d["Quantity"],
                # "titleVector": v_titleVector,
                "category": random.choice(["news", "research", "blog"]),
                "content_vector": v_contentVector,
                "last_update": time.strftime("%Y-%m-%dT%H:%M:%S-00:00", time.localtime()),
                "index_date": time.strftime("%Y-%m-%dT%H:%M:%S-00:00", time.localtime())
        }
        )

    from doc_utils import ingest_docs
    from doc_utils import search_key, search_endpoint, search_index_name

    from azure.core.credentials import AzureKeyCredential  
    from azure.search.documents import SearchClient
    search_credential = AzureKeyCredential(search_key)
    client = SearchClient(search_endpoint, search_index_name, search_credential)
    _counter = ingest_docs(docs, client)

    print(f"Uploaded {_counter} documents")


if __name__ == "__main__":
    fill_search()