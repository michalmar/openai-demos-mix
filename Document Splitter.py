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


NEW_AFTER_N_CHARS = 2500
COMBINE_UNDER_N_CHARS = 1000
LOCAL_FOLDER = "."

def get_filename(filepath):
    filename = os.path.basename(filepath)
    return os.path.splitext(filename)
def read_chunks(folder):
    
    book_map = []
    for filename in os.listdir(os.path.join(".",folder, "out")):
        (_filename, _ext) = get_filename(filename)
        print(f"Extracting {filename}...")
        with open(file_path, 'r') as file:
            data = json.load(file)
            book_map.append([0, 0, data["content"]])
        book_map.append(filename)
    return book_map
def create_index(book_index_name, verbose=False):
    
    ### Create Azure Search Vector-based Index
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json','api-key': os.environ['AZURE_SEARCH_KEY']}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    index_payload = {
        "name": book_index_name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": "true", "filterable": "true" },
            {"name": "title","type": "Edm.String","searchable": "true","retrievable": "true"},
            {"name": "chunk","type": "Edm.String","searchable": "true","retrievable": "true"},
            {"name": "chunkVector","type": "Collection(Edm.Single)","searchable": "true","retrievable": "true","dimensions": 1536,"vectorSearchConfiguration": "vectorConfig"},
            {"name": "name", "type": "Edm.String", "searchable": "true", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
            {"name": "location", "type": "Edm.String", "searchable": "false", "retrievable": "true", "sortable": "false", "filterable": "false", "facetable": "false"},
            {"name": "page_num","type": "Edm.Int32","searchable": "false","retrievable": "true"},
            
        ],
        "vectorSearch": {
            "algorithmConfigurations": [
                {
                    "name": "vectorConfig",
                    "kind": "hnsw"
                }
            ]
        },
        "semantic": {
            "configurations": [
                {
                    "name": "my-semantic-config",
                    "prioritizedFields": {
                        "titleField": {
                            "fieldName": "title"
                        },
                        "prioritizedContentFields": [
                            {
                                "fieldName": "chunk"
                            }
                        ],
                        "prioritizedKeywordsFields": []
                    }
                }
            ]
        }
    }

    r = requests.put(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + book_index_name,
                    data=json.dumps(index_payload), headers=headers, params=params)
    print(r.status_code)
    print(r.ok)

    return r.ok
def sanitize_key(key):
    # Remove all characters that are not letters, digits, underscore, dash, or equal sign
    sanitized_key = re.sub(r'[^a-zA-Z0-9_=.-]', '', key)
    return sanitized_key
def ingest_documents(book_index_name, book_pages_map = dict(), verbose=False):
    # Setup the Payloads header
    headers = {'Content-Type': 'application/json','api-key': os.environ['AZURE_SEARCH_KEY']}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    for bookname,bookmap in book_pages_map.items():
        print("Uploading chunks from",bookname)
        for page in tqdm(bookmap):
            try:
                page_num = page[0] + 1
                content = page[2]
                book_url = BASE_CONTAINER_URL + bookname
                upload_payload = {
                    "value": [
                        {
                            "id": sanitize_key(text_to_base64(bookname + str(page_num))),
                            "title": f"{bookname}_page_{str(page_num)}",
                            "chunk": content,
                            "chunkVector": embedder.embed_query(content if content!="" else "-------"),
                            "name": bookname,
                            "location": book_url,
                            "page_num": page_num,
                            "@search.action": "upload"
                        },
                    ]
                }

                r = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + book_index_name + "/docs/index",
                                    data=json.dumps(upload_payload), headers=headers, params=params)
                if r.status_code != 200:
                    print(r.status_code)
                    print(r.text)
            except Exception as e:
                print("Exception:",e)
                print(content)
                continue
def split_documents(books, folder, verbose=False):
    book_pages_map = dict()
    for book in books:
        print("Extracting Text from",book,"...")

        (_filename, _ext) = get_filename(book)

        if (_ext == ".pdf"):
        
            # Capture the start time
            start_time = time.time()
            
            # Parse the PDF
            book_path = os.path.join(folder,book)
            book_map = parse_pdf(file=book_path, form_recognizer=False, verbose=True)
            book_pages_map[book]= book_map
            
            # Capture the end time and Calculate the elapsed time
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(f"Parsing took: {elapsed_time:.6f} seconds")
            print(f"{book} contained {len(book_map)} pages\n")

            # for bookname,bookmap in book_pages_map.items():
            #     print(bookname,"\n","chunk text:",bookmap[random.randint(1, len(bookmap)-1)][2][:80],"...\n")

        
        # TODO if txt has more data -> split is needed
        elif (_ext == ".txt"):
            book_path = os.path.join(folder,book)
            with open(book_path, "rb") as f:
                txt = parse_txt(f)
                book_pages_map[book] = [[0, 0, txt]]
            pass
        else:
            elements, metadata = PartitionFile(_ext, book)
            metdata_text = ''
            for metadata_value in metadata:
                metdata_text += metadata_value + '\n'    
            # statusLog.upsert_document(blob_name, f'{function_name} - partitioning complete', StatusClassification.DEBUG)
            
            title = ''
            # Capture the file title
            try:
                for i, element in enumerate(elements):
                    if title == '' and element.category == 'Title':
                        # capture the first title
                        title = element.text
                        break
            except:
                # if this type of eleemnt does not include title, then process with emty value
                pass
            
            # Chunk the file     
            from unstructured.chunking.title import chunk_by_title
            # chunks = chunk_by_title(elements, multipage_sections=True, new_after_n_chars=NEW_AFTER_N_CHARS, combine_under_n_chars=COMBINE_UNDER_N_CHARS)
            # chunks = chunk_by_title(elements, multipage_sections=True, new_after_n_chars=NEW_AFTER_N_CHARS, combine_under_n_chars=COMBINE_UNDER_N_CHARS, max_characters=MAX_CHARACTERS)   
            chunks = chunk_by_title(elements, multipage_sections=True, new_after_n_chars=NEW_AFTER_N_CHARS, combine_text_under_n_chars=COMBINE_UNDER_N_CHARS)
            # statusLog.upsert_document(blob_name, f'{function_name} - chunking complete. {str(chunks.count)} chunks created', StatusClassification.DEBUG)
                    
            subtitle_name = ''
            section_name = ''
            page_map = []
            # Complete and write chunks
            for i, chunk in enumerate(chunks):      
                if chunk.metadata.page_number == None:
                    page_list = [1]
                else:
                    page_list = [chunk.metadata.page_number] 
                # substitute html if text is a table            
                if chunk.category == 'Table':
                    chunk_text = chunk.metadata.text_as_html
                else:
                    chunk_text = chunk.text
                # add filetype specific metadata as chunk text header
                chunk_text = metdata_text + chunk_text                    
                # utilities.write_chunk(blob_name, blob_uri,
                #                     f"{i}",
                #                     utilities.token_count(chunk.text),
                #                     chunk_text, page_list,
                #                     section_name, title, subtitle_name,
                #                     MediaType.TEXT
                #                     )
                page_map.append((i, 0, chunk_text))
                # write_chunk(blob_path_plus_sas, ".",
                #                     f"{i}",
                #                     99, # utilities.token_count(chunk.text),
                #                     chunk_text, page_list,
                #                     section_name, title, subtitle_name,
                #                     "TEXT" #MediaType.TEXT
                #                     )
            book_pages_map[book] = page_map
        # elif (_ext == ".docx"):
        #     raise Exception("not working due to a missing dependency for 'unstructured' package")
        #     chunk_document(os.path.join(folder,book))
        #     book_pages_map[book] = read_chunks(folder)
        #     pass

   
    return book_pages_map
def PartitionFile(file_extension: str, file_url: str):      
    """ uses the unstructured.io libraries to analyse a document
    Returns:
        elements: A list of available models
    """  
    # Send a GET request to the URL to download the file
    # response = requests.get(file_url)
    # bytes_io = BytesIO(response.content)
    # response.close()   
    bytes_io = file_url
    metadata = [] 
    try:        
        if file_extension == '.csv':
            from unstructured.partition.csv import partition_csv
            elements = partition_csv(filename=bytes_io)               
                     
        elif file_extension == '.doc':
            from unstructured.partition.doc import partition_doc
            elements = partition_doc(filename=bytes_io) 
            
        elif file_extension == '.docx':
            from unstructured.partition.docx import partition_docx
            elements = partition_docx(filename=bytes_io)
            
        elif file_extension == '.eml' or file_extension == '.msg':
            if file_extension == '.msg':
                from unstructured.partition.msg import partition_msg
                elements = partition_msg(filename=bytes_io) 
            else:        
                from unstructured.partition.email import partition_email
                elements = partition_email(filename=bytes_io)
            metadata.append(f'Subject: {elements[0].metadata.subject}')
            metadata.append(f'From: {elements[0].metadata.sent_from[0]}')
            sent_to_str = 'To: '
            for sent_to in elements[0].metadata.sent_to:
                sent_to_str = sent_to_str + " " + sent_to
            metadata.append(sent_to_str)
            
        elif file_extension == '.html' or file_extension == '.htm':  
            from unstructured.partition.html import partition_html
            elements = partition_html(filename=bytes_io) 
            
        elif file_extension == '.md':
            from unstructured.partition.md import partition_md
            elements = partition_md(filename=bytes_io)
                       
        elif file_extension == '.ppt':
            from unstructured.partition.ppt import partition_ppt
            elements = partition_ppt(filename=bytes_io)
            
        elif file_extension == '.pptx':    
            from unstructured.partition.pptx import partition_pptx
            elements = partition_pptx(filename=bytes_io)
            
        elif any(file_extension in x for x in ['.txt', '.json']):
            from unstructured.partition.text import partition_text
            elements = partition_text(filename=bytes_io)
            
        elif file_extension == '.xlsx':
            from unstructured.partition.xlsx import partition_xlsx
            elements = partition_xlsx(filename=bytes_io)
            
        elif file_extension == '.xml':
            from unstructured.partition.xml import partition_xml
            elements = partition_xml(filename=bytes_io)
            
    except Exception as e:
        raise UnstructuredError(f"An error occurred trying to parse the file: {str(e)}") from e
         
    return elements, metadata
def write_chunk(myblob_name, myblob_uri, file_number, chunk_size, chunk_text, page_list, 
                    section_name, title_name, subtitle_name, file_class):
        """ Function to write a json chunk to blob"""
        chunk_output = {
            'file_name': myblob_name,
            'file_uri': myblob_uri,
            'file_class': file_class,
            'processed_datetime': datetime.now().isoformat(),
            'title': title_name,
            'subtitle': subtitle_name,
            'section': section_name,
            'pages': page_list,
            'token_count': chunk_size,
            'content': chunk_text                       
        }
        # Get path and file name minus the root container
        # file_name, file_extension, file_directory = self.get_filename_and_extension(myblob_name)
        (file_name, file_extension) = get_filename(myblob_name)
    
        # blob_service_client = BlobServiceClient(
        #     self.azure_blob_storage_endpoint,
        #     self.azure_blob_storage_key)
        json_str = json.dumps(chunk_output, indent=2, ensure_ascii=False)
        # block_blob_client = blob_service_client.get_blob_client(
            # container=self.azure_blob_content_storage_container,
            # blob=self.build_chunk_filepath(file_directory, file_name, file_extension, file_number))
        # block_blob_client.upload_blob(json_str, overwrite=True)

        # save to disk
        with open(f"out/{file_name}_{file_number}.json", "w") as f:
            f.write(json_str)


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
                                            k=10,
                                            reranker_threshold=0.1, #1
                                            vector_search=True, 
                                            similarity_k=10,
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
            
    # print("Number of chunks:",len(top_docs))

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

#################################################################################
# App elements

st.title("ChatGPT Demo with LangChain (with sources)")

with st.sidebar:
    st.caption("Settings")
    st.session_state.model = st.selectbox("Select a model", ["gpt-35-turbo", "gpt-35-turbo-16k","gpt-4", "gpt-4-turbo"])
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
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
    if uploaded_file is not None:
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
                llm = AzureChatOpenAI(deployment_name=st.session_state.model, temperature=st.session_state.temperature, max_tokens=600, streaming=True, callbacks=[stream_handler])
                
                # check if db is loaded - if so, use the qa_with_sources chain
                if (st.session_state.db is not None):
                    output = ask_gpt_with_sources(llm, prompt, st.session_state.session_id)
                else:
                    # Get response from GPT
                    output = ask_gpt(llm, prompt, st.session_state.session_id)       
                
                stream_handler.container.markdown(output, unsafe_allow_html=True)
    

