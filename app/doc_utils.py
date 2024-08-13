

from dotenv import load_dotenv
if not load_dotenv("../credentials.env", override=True):
    load_dotenv("credentials.env", override=True)

# Import required libraries  
import os  
import shutil  
import json  
# import openai
from openai import AzureOpenAI
from openai import APIError

# For LibreOffice Doc Conversion to PDF
import subprocess  
import pathlib
from pathlib import Path  
import os  
from azure.core.credentials import AzureKeyCredential  
# from azure.ai.textanalytics import TextAnalyticsClient  

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryCaptionResult,
    QueryAnswerResult,
    SemanticErrorMode,
    SemanticErrorReason,
    SemanticSearchResultsType,
    QueryType,
    VectorizedQuery,
    VectorQuery,
    VectorFilterMode,    
)
from azure.search.documents.indexes.models import (  
    ExhaustiveKnnAlgorithmConfiguration,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticPrioritizedFields,
    SemanticField,  
    SearchField,  
    SemanticSearch,
    VectorSearch,  
    HnswAlgorithmConfiguration,
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchAlgorithmKind,
    VectorSearchProfile,
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    ExhaustiveKnnParameters,
    SearchIndex,  
    SearchField,  
    SearchFieldDataType,  
    SimpleField,  
    SearchableField,  
    SearchIndex,  
    SemanticConfiguration,  
    SemanticField,  
    SearchField,  
    VectorSearch,  
    HnswParameters,  
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)  

from tenacity import retry, wait_random_exponential, stop_after_attempt 

from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter, PythonCodeTextSplitter

from langchain_experimental.text_splitter import SemanticChunker
# # from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

import tiktoken

import re
import PyPDF2  
import base64
import time
import uuid

try:
    import pymupdf as fitz  # available with v1.24.3
except ImportError:
    import fitz

search_endpoint =  os.getenv("AZURE_SEARCH_ENDPOINT")
search_key =  os.getenv("AZURE_SEARCH_KEY")
search_index_name = os.getenv("AZURE_SEARCH_INDEX")

openai_api_key  = os.getenv("AZURE_OPENAI_API_KEY")
openai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")

openai_embedding_api_base = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
openai_embedding_api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai_embedding_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
openai_embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL_NAME")

supported_conversion_types = ['.pptx', '.ppt', '.docx', '.doc', '.xlsx', '.xls', '.pdf']
image_path = 'images'
markdown_path = 'markdown'


embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=openai_embedding_api_base,
    api_key=openai_embedding_api_key,
    azure_deployment=openai_embedding_model,
    openai_api_version=openai_embedding_api_version,
)

# Function to generate vectors for title and content fields, also used for query vectors
max_attempts = 6
max_backoff = 60
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(max_attempts))
def generate_embedding(text: str) -> list:
    
    if text == None:
        return None
        
    if len(text) < 10:
        return None
        
    client = AzureOpenAI(
        api_version=openai_embedding_api_version,
        azure_endpoint=openai_embedding_api_base,
        api_key=openai_embedding_api_key
    )    
    counter = 0
    incremental_backoff = 1   # seconds to wait on throttline - this will be incremental backoff
    while True and counter < max_attempts:
        try:
            # text-embedding-3-small == 1536 dims
            response = client.embeddings.create(
                input=text,
                model=openai_embedding_model
            )
            return json.loads(response.model_dump_json())["data"][0]['embedding']
        except APIError as ex:
            # Handlethrottling - code 429
            if str(ex.code) == "429":
                incremental_backoff = min(max_backoff, incremental_backoff * 1.5)
                print ('Waiting to retry after', incremental_backoff, 'seconds...')
                time.sleep(incremental_backoff)
            elif str(ex.code) == "content_filter":
                print ('API Error', ex.code)
                return None
        except Exception as ex:
            counter += 1
            print ('Error - Retry count:', counter, ex)
    return None


def clean_spaces_with_regex(text):
    # Use a regular expression to replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', text)
    # Use a regular expression to replace consecutive dots with a single dot
    cleaned_text = re.sub(r'\.{2,}', '.', cleaned_text)
    return cleaned_text

def estimate_tokens(text):
    GPT2_TOKENIZER = tiktoken.get_encoding("gpt2")
    return(len(GPT2_TOKENIZER.encode(text)))

def chunk_data(text):
    text = clean_spaces_with_regex(text)
    SENTENCE_ENDINGS = [".", "!", "?"]
    WORDS_BREAKS = ['\n', '\t', '}', '{', ']', '[', ')', '(', ' ', ':', ';', ',']
    num_tokens = 1024 #500
    min_chunk_size = 10
    token_overlap = 128

    # splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(separators=SENTENCE_ENDINGS + WORDS_BREAKS,chunk_size=num_tokens, chunk_overlap=token_overlap)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=num_tokens, chunk_overlap=token_overlap)

    return(splitter.split_text(text))

def chunk_data_semantic(text):
    text = clean_spaces_with_regex(text)
    text_splitter = SemanticChunker(embeddings)

    docs = text_splitter.split_text(text)
    return docs


# This is just a hack - different paths to file when locally or in Azure
def find_schema_file(schema_file):
    if os.path.exists(schema_file):
        return schema_file
    elif os.path.exists(os.path.join('app', schema_file)):
        return os.path.join('app', schema_file)
    elif os.path.exists(os.path.join('..', 'app', schema_file)):
        return os.path.join('..', 'app', schema_file)
    else:
        return None

def create_index(index_name=search_index_name, recreate=False, schema = "schema.json"):
    print('Creating Index:', index_name)
    dims = len(generate_embedding('That quick brown fox.'))
    print ('Dimensions in Embedding Model:', dims)

    with open(find_schema_file(schema), "r") as f_in:
        index_schema = json.loads(f_in.read())
        index_schema['name'] = index_name
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['resourceUri'] = openai_embedding_api_base
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['deploymentId'] = openai_embedding_model
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['apiKey'] = openai_embedding_api_key
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['apiKey'] = openai_embedding_api_key
        index_schema['vectorSearch']['vectorizers'][0]['azureOpenAIParameters']['modelName'] = openai_embedding_model
    

    # Create the search index
    search_credential = AzureKeyCredential(search_key)

    index_client = SearchIndexClient(
        endpoint=search_endpoint, credential=search_credential)

    if recreate:
        try:
            index_client.delete_index(index_name)
        except Exception as e:
            print(e)
    
    # fields = [
    #     SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True, filterable=True, facetable=True),
    #     SearchableField(name="chunk_id", type=SearchFieldDataType.String),
    #     SearchableField(name="document_id", type=SearchFieldDataType.String),
    #     SearchableField(name="title", type=SearchFieldDataType.String),
    #     SearchableField(name="content", type=SearchFieldDataType.String),
    #     SearchableField(name="sourceurl", type=SearchFieldDataType.String),
    #     SearchableField(name="publicurl", type=SearchFieldDataType.String),
    #     # SimpleField(name="dateTime", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="Person", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="Location", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="Organization", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="URL", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="Email", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="PersonType", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="Event", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     # SimpleField(name="Quantity", type=SearchFieldDataType.Collection(SearchFieldDataType.String),Filterable=True,Sortable=True, Facetable=True),
    #     SearchField(name="titleVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    #                 searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile"),
    #     SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
    #                 searchable=True, vector_search_dimensions=1536, vector_search_profile_name="myHnswProfile")
    # ]

    # # Configure the vector search configuration  
    # vector_search = VectorSearch(
    #     algorithms=[
    #         HnswAlgorithmConfiguration(
    #             name="myHnsw",
    #             kind=VectorSearchAlgorithmKind.HNSW,
    #             parameters=HnswParameters(
    #                 m=4,
    #                 ef_construction=400,
    #                 ef_search=500,
    #                 metric=VectorSearchAlgorithmMetric.COSINE
    #             )
    #         ),
    #         ExhaustiveKnnAlgorithmConfiguration(
    #             name="myExhaustiveKnn",
    #             kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
    #             parameters=ExhaustiveKnnParameters(
    #                 metric=VectorSearchAlgorithmMetric.COSINE
    #             )
    #         )
    #     ],
    #     profiles=[
    #         VectorSearchProfile(
    #             name="myHnswProfile",
    #             algorithm_configuration_name="myHnsw",
    #         ),
    #         VectorSearchProfile(
    #             name="myExhaustiveKnnProfile",
    #             algorithm_configuration_name="myExhaustiveKnn",
    #         )
    #     ]
    # )

    # semantic_config = SemanticConfiguration(
    #     name="my-semantic-config",
    #     prioritized_fields=SemanticPrioritizedFields(
    #         title_field=SemanticField(field_name="title"),
    #         content_fields=[SemanticField(field_name="content")]
    #     )
    # )

    # # Create the semantic settings with the configuration
    # semantic_search = SemanticSearch(configurations=[semantic_config])

    # # Create the drafts search index with the semantic settings
    # index = SearchIndex(name=index_name, fields=fields,
    #                     vector_search=vector_search, semantic_search=semantic_search)
    
    index = SearchIndex.from_dict(index_schema)

    result = index_client.create_or_update_index(index)
    # print(f' {result.name} created')
    return result

# TODO: Change to use content types
def check_file_type(file_path):
    if file_path.endswith('.pdf'):
        return 'pdf'
    elif file_path.endswith('.docx'):
        return 'docx'
    elif file_path.endswith('.pptx'):
        return 'pptx'
    elif file_path.endswith('.txt'):
        return 'txt'
    else:
        return 'unknown'
    
def process_and_upload_files(paths, num_pages, use_semantic_chunking=False):
    upload_batch_size = 10
    search_credential = AzureKeyCredential(search_key)
    client = SearchClient(search_endpoint, search_index_name, search_credential)
    
    # docs = []
    num_files = 0
    counter = 0
    for _path in paths:
        num_files += 1
        # check if path is string
        if not isinstance(_path, str):
            # probably uploaded from Streamlit
            path = _path
            filename = _path.name
            file_type = check_file_type(_path.name)
             
        else:
            path = _path
            filename = _path.split('/')[-1] 
            file_type = check_file_type(_path)

        if file_type == 'pdf':
            docs = process_pdf(path, num_pages, filename, use_semantic_chunking)
        elif file_type == 'docx':
            docs = process_docx(path, num_pages, filename, use_semantic_chunking)
        elif file_type == 'pptx':
            docs = process_pptx(path, num_pages, filename, use_semantic_chunking)
        elif file_type == 'txt':
            docs = process_txt(path, num_pages, filename, use_semantic_chunking)
        else:
            print(f"Unknown file type: {file_type}")
            _counter = 0

        _counter = ingest_docs(docs, client)
        counter += _counter

    return num_files, counter

def convert_pdf_to_md(pdf_file):
    import pymupdf4llm


    if not isinstance(pdf_file, str):

        if not isinstance(pdf_file, fitz.Document):
            # pdf_file = fitz.open(pdf_file)
            pdf_file = fitz.open(stream=pdf_file.read(), filetype="pdf")

    md_text = pymupdf4llm.to_markdown(pdf_file)

    # # now work with the markdown text, e.g. store as a UTF8-encoded file
    # import pathlib
    # pathlib.Path("output.md").write_bytes(md_text.encode())

    # import os
    # os.chdir("/Users/mimarusa/Documents/PRJ/openai-demos-mix")
    return md_text

def reset_local_dirs():
    if os.path.exists('json'):
        remove_directory('json')
    if os.path.exists('images'):
        remove_directory('images')
    if os.path.exists('markdown'):
        remove_directory('markdown')
    if os.path.exists('pdf'):
        remove_directory('pdf')
    if os.path.exists('merged'):
        remove_directory('merged')
    if os.path.exists('tmp'):
        remove_directory('tmp')

# Create directory if it does not exist
def ensure_directory_exists(directory_path):  
    path = Path(directory_path)  
    if not path.exists():  
        path.mkdir(parents=True, exist_ok=True)  
        print(f"Directory created: {directory_path}")  
    else:  
        print(f"Directory already exists: {directory_path}")  
  
# Remove a dir and sub-dirs
def remove_directory(directory_path):  
    try:  
        if os.path.exists(directory_path):  
            shutil.rmtree(directory_path)  
            print(f"Directory '{directory_path}' has been removed successfully.")  
        else:  
            print(f"Directory '{directory_path}' does not exist.")  
    except Exception as e:  
        print(f"An error occurred while removing the directory: {e}")  
    
def convert_doc_to_pdf(input_path):  
    print('converting to PDF', input_path)

    file_suffix = pathlib.Path(input_path).suffix.lower()
    
    if file_suffix in supported_conversion_types:
        ensure_directory_exists('pdf')  
        
        output_file = input_path.replace(pathlib.Path(input_path).suffix, '')
        output_file = os.path.join('pdf', output_file + '.pdf')
    
        print ('Converting', input_path, 'to', output_file)
        if os.path.exists(output_file):
            os.remove(output_file)
    
        if file_suffix == '.pdf':
            # No need to convert, just copy
            shutil.copy(input_path, output_file)  
        else:
            # Command to convert pptx to pdf using LibreOffice  
            command = [  
                'soffice',  # or 'libreoffice' depending on your installation  
                '--headless',  # Run LibreOffice in headless mode (no GUI)  
                '--convert-to', 'pdf',  # Specify conversion format  
                '--outdir', os.path.dirname(output_file),  # Output directory  
                input_path  # Input file  
            ]  
              
            # Run the command  
            subprocess.run(command, check=True)  
            print(f"Conversion complete: {output_file}")  
    else:
        print ('File type not supported.')  
        return ""
    
    return output_file


# Convert pages from PDF to images
def extract_pdf_pages_to_images(pdf_path, image_dir):
    # Validate image_out directory exists
    doc_id = str(uuid.uuid4())
    image_out_dir = os.path.join(image_dir, doc_id)
    ensure_directory_exists(image_out_dir)  

    # Open the PDF file and iterate pages
    print ('Extracting images from PDF...')
    pdf_document = fitz.open(pdf_path)  

    for page_number in range(len(pdf_document)):  
        page = pdf_document.load_page(page_number)  
        image = page.get_pixmap()  
        image_out_file = os.path.join(image_out_dir, f'{page_number + 1}.png')
        image.save(image_out_file)  
        if page_number % 100 == 0:
            print(f'Processed {page_number} images...')  

    return doc_id

# Base64 encode images
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
# Find all files in a dir
def get_all_files(directory_path):  
    files = []  
    for entry in os.listdir(directory_path):  
        entry_path = os.path.join(directory_path, entry)  
        if os.path.isfile(entry_path):  
            files.append(entry_path)  
    return files  
  

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def extract_markdown_from_image(image_path):
    client = AzureOpenAI(
        api_version=openai_api_version,
        azure_endpoint=openai_api_base,
        api_key = openai_api_key
    )
    try:
        base64_image = encode_image(image_path)
        response = client.chat.completions.create(
            model="gpt-4o", # TODO: Change to use the model from env var
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [  
                    { 
                        "type": "text", 
                        "text": """Extract everything you see in this image to markdown. 
                            Convert all charts such as line, pie and bar charts to markdown tables and include a note that the numbers are approximate.
                        """ 
                    },
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                    }
                ] } 
            ],
            max_tokens=2000 
        )
        return response.choices[0].message.content
    except Exception as ex:
        return ""

def process_image(file, markdown_out_dir):
    if '.png' in file:
        print ('Processing:', file)
        markdown_file_out = os.path.join(markdown_out_dir, os.path.basename(file).replace('.png', '.txt'))
        print(markdown_file_out)
        if os.path.exists(markdown_file_out) == False:
            markdown_text = extract_markdown_from_image(file)
            with open(markdown_file_out, 'w') as md_out:
                md_out.write(markdown_text)
        else:
            print ('Skipping processed file.')
    else:
        print ('Skipping non PNG file:', file)

    return file
  
def process_docx(path, num_pages, filename, use_semantic_chunking=False):
    reset_local_dirs()
    out = convert_doc_to_pdf(path)
    docs = process_pdf(out, num_pages, filename, use_semantic_chunking=use_semantic_chunking)
    reset_local_dirs()
    return docs

def process_pptx(path, num_pages, filename, use_semantic_chunking=False):

    reset_local_dirs()
    import concurrent.futures  
    from functools import partial  
    
    pdf_path = convert_doc_to_pdf(path)
    # Extract PDF pages to images
    doc_id = extract_pdf_pages_to_images(pdf_path, image_path)
    pdf_images_dir = os.path.join(image_path, doc_id)
    print ('Images saved to:', pdf_images_dir)
    print ('Doc ID:', doc_id)
    files = get_all_files(pdf_images_dir)  
    total_files = len(files)
    print ('Total Image Files to Process:', total_files)

    # Convert the images to markdown using GPT-4o 
    # Process pages in parallel - adjust worker count as needed
    max_workers = 10

    markdown_out_dir = os.path.join(markdown_path, doc_id)
    ensure_directory_exists(markdown_out_dir)

    # Using ThreadPoolExecutor with a limit of max_workers threads  
    partial_process_image = partial(process_image, markdown_out_dir=markdown_out_dir)  
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:  
        # Map the partial function to the array of items  
        results = list(executor.map(partial_process_image, files))  
        
    print('Total Processed:', len(results))

    files = get_all_files(markdown_out_dir)  

    docs = []
    # for file in files:
    for idx, file in enumerate(files):
        with open(file, 'r') as f:
            text = f.read()
            docs += process_text_only(text, num_pages, filename, page_num=idx, use_semantic_chunking=use_semantic_chunking)
    reset_local_dirs()
    return docs

def get_text_from_txt(path):
    if not isinstance(path, str): # its the content
        return path.read().decode('utf-8')
    else:
        
        with open(path, 'r') as file:
            text = file.read()
        return text

def process_txt(path, num_pages, filename, use_semantic_chunking=False):
    text = get_text_from_txt(path)
    return process_text_only(text, num_pages, filename, use_semantic_chunking=use_semantic_chunking)

def process_pdf(path, num_pages, filename, use_semantic_chunking=False):
    text = convert_pdf_to_md(path)
    return process_text_only(text, num_pages, filename, use_semantic_chunking=use_semantic_chunking)

def process_text_only(text, num_pages, filename, page_num=0, use_semantic_chunking=False):

    docs = []
    counter = 0

    document_id = filename.replace('.pdf','')

    # df_file_metadata = df_metadata[df_metadata['grant_id']==document_id].iloc[0]
    df_file_metadata = {
        'grant_id': document_id,
        'title': filename,
        'publicurl': 'https://www.example.com/' + filename
    }
    if num_pages == 0:
        page_num = 0 # TODO: how to handle pages in MD text?
    else:
        page_num = page_num

    if use_semantic_chunking:
        chunks = chunk_data_semantic(text)
    else:          
        chunks = chunk_data(text)

    chunk_num = 0
    for chunk in chunks:
        chunk_num += 1
        d = {
            "chunk_id" : filename + '_' + str(page_num).zfill(2) +  '_' + str(chunk_num).zfill(2),
            "document_id": str(df_file_metadata['grant_id']),
            "content": chunk,       
            "title": df_file_metadata['title']
            }
        # d["dateTime"] = "dummy date"

        counter += 1

        # TODO: this is not valid url since we are not spliting by pages
        public_url = df_file_metadata['publicurl'] + '#page=' + str(page_num) 

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
                "contentVector": v_contentVector
        }
        )
            
          
   
    return docs

def ingest_docs(docs, client):
    upload_batch_size = 10
    counter = 0
    docs_to_upload = []
    for doc in docs:
        counter += 1
        docs_to_upload.append(doc)
        if counter % upload_batch_size == 0:
            result = client.upload_documents(documents=docs)
            docs_to_upload = []
            print(f' {str(counter)} uploaded')
    #upload the last batch
    if docs_to_upload != []:
        client.upload_documents(documents=docs_to_upload)
    return counter

def process_and_upload_pdf_old(path, num_pages, client, filename, upload_batch_size=10):
    # Create the search client

    # upload_batch_size = 10
    # search_credential = AzureKeyCredential(search_key)
    # client = SearchClient(search_endpoint, search_index_name, search_credential)
    # drafts_client = SearchClient(search_endpoint, drafts_index_name, search_credential)
    # index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_credential)

    # metadata_filepath = 'Files/' + account_name + '/' + directory + '/metadata/' + csv_file_name
    # df_metadata = spark.read.format("csv").option("header","true").option("multiLine", "true").option("quote", "\"").option("escape", "\"").load(metadata_filepath).toPandas()

    docs = []
    # num_pdfs = 0
    counter = 0
    # for path in paths:
      
    # num_pdfs += 1
    # pdf_file_path = '/lakehouse/default/Files/' + account_name + '/' + directory + '/pdfs/' + path.name
    pdf_file_path = path
    pdf_reader = PyPDF2.PdfReader(pdf_file_path)
    # filename = path.split('/')[-1]
    document_id = filename.replace('.pdf','')

    # df_file_metadata = df_metadata[df_metadata['grant_id']==document_id].iloc[0]
    df_file_metadata = {
        'grant_id': document_id,
        'title': filename,
        'publicurl': 'https://www.example.com/' + filename
    }
    text = "" 

    n = num_pages #len(pdf_reader.pages) # TODO: change to MAX Pages
    if len(pdf_reader.pages) < n:
        n = len(pdf_reader.pages)
    for page_num in range(n):

        public_url = df_file_metadata['publicurl'] + '#page=' + str(page_num) 
        page = pdf_reader.pages[page_num]
        text = page.extract_text()         
        
        chunks = chunk_data(text)
        chunk_num = 0
        for chunk in chunks:
            chunk_num += 1
            d = {
                "chunk_id" : filename + '_' + str(page_num).zfill(2) +  '_' + str(chunk_num).zfill(2),
                "document_id": str(df_file_metadata['grant_id']),
                "content": chunk,       
                "title": df_file_metadata['title']
                }

            # d["dateTime"],d["Person"],d["Location"],d["Organization"],d["URL"],d["Email"],d["PersonType"],d["Event"],d["Quantity"] = get_named_entities(cog_services_client,d["content"])
            # d["dateTime"] = "dummy date"
            # d["Person"] = "dummy person"
            # d["Location"] = "dummy location"
            # d["Organization"] = "dummy organization"
            # d["URL"] = "dummy url"
            # d["Email"] = "dummy email"
            # d["PersonType"] = "dummy person type"
            # d["Event"] = "dummy event"
            # d["Quantity"] = "dummy quantity"


            counter += 1

            v_contentVector = generate_embedding(d["content"])

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
                    "contentVector": v_contentVector
            }
            )
            
            if counter % upload_batch_size == 0:
                result = client.upload_documents(documents=docs)
                # result = drafts_client.upload_documents(documents=docs)
                docs = []
                print(f' {str(counter)} uploaded')
                    
    #upload the last batch
    if docs != []:
        client.upload_documents(documents=docs)
        # drafts_client.upload_documents(documents=docs)
    
    return counter


def process_and_upload_pdf_old_old(paths, num_pages):
    # Create the search client

    upload_batch_size = 10
    search_credential = AzureKeyCredential(search_key)
    client = SearchClient(search_endpoint, search_index_name, search_credential)
    # drafts_client = SearchClient(search_endpoint, drafts_index_name, search_credential)
    # index_client = SearchIndexClient(endpoint=search_endpoint, credential=search_credential)

    # metadata_filepath = 'Files/' + account_name + '/' + directory + '/metadata/' + csv_file_name
    # df_metadata = spark.read.format("csv").option("header","true").option("multiLine", "true").option("quote", "\"").option("escape", "\"").load(metadata_filepath).toPandas()

    docs = []
    num_pdfs = 0
    counter = 0
    for path in paths:
      
        num_pdfs += 1
        # pdf_file_path = '/lakehouse/default/Files/' + account_name + '/' + directory + '/pdfs/' + path.name
        pdf_file_path = path
        pdf_reader = PyPDF2.PdfReader(pdf_file_path)
        # filename = path.split('/')[-1]
        document_id = filename.replace('.pdf','')

        # df_file_metadata = df_metadata[df_metadata['grant_id']==document_id].iloc[0]
        df_file_metadata = {
            'grant_id': document_id,
            'title': filename,
            'publicurl': 'https://www.example.com/' + filename
        }
        text = "" 

        n = num_pages #len(pdf_reader.pages) # TODO: change to MAX Pages
        if len(pdf_reader.pages) < n:
            n = len(pdf_reader.pages)
        for page_num in range(n):

            public_url = df_file_metadata['publicurl'] + '#page=' + str(page_num) 
            page = pdf_reader.pages[page_num]
            text = page.extract_text()         
            
            chunks = chunk_data(text)
            chunk_num = 0
            for chunk in chunks:
                chunk_num += 1
                d = {
                    "chunk_id" : filename + '_' + str(page_num).zfill(2) +  '_' + str(chunk_num).zfill(2),
                    "document_id": str(df_file_metadata['grant_id']),
                    "content": chunk,       
                    "title": df_file_metadata['title']
                    }

                # d["dateTime"],d["Person"],d["Location"],d["Organization"],d["URL"],d["Email"],d["PersonType"],d["Event"],d["Quantity"] = get_named_entities(cog_services_client,d["content"])
                # d["dateTime"] = "dummy date"
                # d["Person"] = "dummy person"
                # d["Location"] = "dummy location"
                # d["Organization"] = "dummy organization"
                # d["URL"] = "dummy url"
                # d["Email"] = "dummy email"
                # d["PersonType"] = "dummy person type"
                # d["Event"] = "dummy event"
                # d["Quantity"] = "dummy quantity"


                counter += 1

                v_contentVector = generate_embedding(d["content"])
    
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
                        "contentVector": v_contentVector
                }
                )
                
                if counter % upload_batch_size == 0:
                    result = client.upload_documents(documents=docs)
                    # result = drafts_client.upload_documents(documents=docs)
                    docs = []
                    print(f' {str(counter)} uploaded')
                    
    #upload the last batch
    if docs != []:
        client.upload_documents(documents=docs)
        # drafts_client.upload_documents(documents=docs)
    
    return num_pdfs, counter



# def do_rag(user_question: str, openai_api_base, openai_api_version, openai_api_key, search_endpoint, search_key, search_index_name):
def do_rag(messages: list
            , openai_api_base = openai_api_base
            , openai_api_version = openai_api_version
            , openai_api_key = openai_api_key
            , search_endpoint = search_endpoint
            , search_key = search_key
            , deployment = "gpt-4o"
            , search_index_name = search_index_name
            , max_tokens = 800
            , temperature = 0
            , top_p = 1
            , frequency_penalty = 0
            , presence_penalty = 0  
            , stop = None
            , stream = False
            , system_prompt = "You are AI Assistant."
            , verbose=False):
        
    # import os
    # from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    # openai_api_base = os.getenv("ENDPOINT_URL", "https://openaimma-swedencentral.openai.azure.com/")
    # deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
    # deployment = "gpt-4o"

    # search_endpoint = os.getenv("SEARCH_ENDPOINT", "https://mma-search-demo2.search.windows.net")
    # search_key = os.getenv("SEARCH_KEY", "put your Azure AI Search admin key here")
    # search_index_name = os.getenv("SEARCH_INDEX_NAME", "images-idx")

    # token_provider = get_bearer_token_provider(
    #     DefaultAzureCredential(),
    #     "https://cognitiveservices.azure.com/.default")
        
    # client = AzureOpenAI(
    #     azure_endpoint=endpoint,
    #     azure_ad_token_provider=token_provider,
    #     api_version="2024-05-01-preview",
    # )

    client = AzureOpenAI(
        api_version=openai_api_version,
        azure_endpoint=openai_api_base,
        api_key = openai_api_key
    )
        
    completion = client.chat.completions.create(
        model=deployment,
        messages= messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=None,
        stream=stream,
        extra_body={
            "data_sources": [{
                "type": "azure_search",
                "parameters": {
                    "endpoint": f"{search_endpoint}",
                    "index_name": f"{search_index_name}",
                    "semantic_configuration": "my-semantic-config",
                    "query_type": "vector_semantic_hybrid",
                    "fields_mapping": {
                        "content_fields_separator": "\n",
                        "content_fields": [
                            "content"
                        ],
                        "filepath_field": "sourceurl",
                        "title_field": "title",
                        "url_field": "publicurl",
                        "vector_fields": [
                            "contentVector"
                        ]
                    },
                    "in_scope": True,
                    "role_information": system_prompt,
                    "filter": None,
                    "strictness": 3,
                    "top_n_documents": 5,
                    "authentication": {
                        "type": "api_key",
                        "key": f"{search_key}"
                    },
                    "embedding_dependency": {
                        "type": "deployment_name",
                        "deployment_name": openai_embedding_model
                    }
                }
            }]
        }
    )
    
    # return completion.choices[0].message.content
    if not stream:
        if verbose:
            print(completion.to_json())
        else:
            print(completion.choices[0].message.content)

    return completion



if __name__ == "__main__":
    num_pages = 10
    # search_index_name = "images-idx" # os.getenv("AZURE_SEARCH_INDEX")
    
    create_index(search_index_name, recreate=True, schema = "app/schema.json")
    print(f"Index {search_index_name} created")


    # account_name = get_secrets_from_kv(key_vault_name, "ADLS-ACCOUNT-NAME")
    # path_name = 'Files/' + account_name + '/' + directory + '/pdfs'
    # paths = ["./Obranna__strategie_C_R_2023_final copy.pdf"]
    # paths = ["./Transforming-Content-with-GPT4o.pptx"]
    paths = ["./Obranna__strategie_C_R_2023_final.txt"]
    num_files, num_chunks = process_and_upload_files(paths, num_pages)
    print(f"Processed {num_files} files and {num_chunks} chunks")

    # messages = [
    #     {
    #     "role": "user",
    #     "content": "Kdo je ministr obrany CR?"
    #     }]

    # _ = do_rag(messages, stream=False, verbose=False)
    # print(response.choices[0].message.content)