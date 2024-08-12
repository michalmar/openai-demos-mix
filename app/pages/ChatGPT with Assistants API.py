import streamlit as st
import os
import json
import time
import random
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI

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
    st.session_state.model = "gpt-4-turbo"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 200

if "assistent_thread" not in st.session_state:
    client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),  
                api_version="2024-02-15-preview",
                azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    st.session_state.client = client

    # Create an assistant
    assistant_code = client.beta.assistants.create(
        name="Math Assist",
        instructions="You are an AI assistant that can write code to help answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model=st.session_state.model #You must replace this value with the deployment name for your model.
    )

    st.session_state.assistant_code = assistant_code

    # # open file for reading
    # with open("./upload/movies.csv", "r") as f:
    #     st.markdown(f.read())


    # Upload a file with an "assistants" purpose
    st.session_state.file = client.files.create(
        file=open("./upload/movies.csv", "rb"),
        purpose='assistants'
    )

    # Create an assistant using the file ID
    assistant_movies = client.beta.assistants.create(
        instructions="You are a movie analyst. When asked a question, you will parse your CSV file to provide the requested analysis.",
        name="Movie Analyst",
        model=st.session_state.model,
        tools=[{"type": "code_interpreter"}],
        file_ids=[st.session_state.file.id]
    )

    st.session_state.assistant_movies = assistant_movies

    # Create a thread
    thread = client.beta.threads.create()
    st.session_state.assistent_thread = thread



#################################################################################
# App elements


st.title("ChatGPT with Assistants API")

with st.sidebar:
    st.caption("Settings")

    st.markdown(f"Thread: `{st.session_state.assistent_thread.id}`")

    # st.session_state.model = st.selectbox("Select a model", ["gpt-35-turbo", "gpt-35-turbo-16k","gpt-4", "gpt-4-turbo"])
    # st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
    # st.session_state.max_tokens = st.slider("Max tokens", 10, 4000, 200, 5)
        
    # st.text_area("Enter your SYSTEM message", key="system_custom_prompt", value=st.session_state.SYSTEM_PROMPT)
    if st.button("Apply & Clear Memory"):
        # save the text from the text_area to SYSTEM_PROMPT
        st.session_state.SYSTEM_PROMPT = st.session_state.system_custom_prompt
        st.session_state.messages = [
                        {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                    ]
    st.caption("Refresh the page to reset to default settings")

    

st.caption(f"powered by Azure OpenAI's {st.session_state.model} model")
with st.expander("What is Assistants API?"):
    st.markdown("Assistants API - new API for creating and managing AI assistants from OpenAI. Based on [Azure Doc Sample](https://learn.microsoft.com/en-us/azure/ai-services/openai/assistants-quickstart)")
    st.caption("Azure OpenAI Assistants (Preview) allows you to create AI assistants tailored to your needs through custom instructions and augmented by advanced tools like code interpreter, retrieval, and custom functions.")


st.caption("Currently, there is only Code Interpreter assitant - more to come...")
st.caption(f"TRY: I need to solve the equation `3x + 11 = 14`. Can you help me?")

for message in st.session_state.messages:
    if message["role"] == "system":
        pass
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("I need to solve the equation `3x + 11 = 14`. Can you help me?"):
    if prompt:
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            client = st.session_state.client
            assistant_code = st.session_state.assistant_code
            assistent_movies = st.session_state.assistant_movies
            thread = st.session_state.assistent_thread

            # Add a user question to the thread
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            # Run the thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant_code.id,
            )

            # Retrieve the status of the run
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )

            status = run.status
            # full_response = "Waiting for the assistant to respond..."
            # Wait till the assistant has responded
            i=0

            # List of some Unicode emojis
            emojis = ["⏳", "⏲️", "⏱️", "⏰","🧭"]

            while status not in ["completed", "cancelled", "expired", "failed"]:
                    # print(f"Waiting for the assistant to respond...(attempt {i+1})")

                    # message_placeholder.text(f"Waiting for the assistant to respond...(attempt {i+1})")
                    message_placeholder.markdown("Waiting for the assistant to respond..." + random.choice(emojis))
                    time.sleep(2)
                    run = client.beta.threads.runs.retrieve(thread_id=thread.id,run_id=run.id)
                    status = run.status
                    i+=1

            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            assistant_messages = [item for item in messages.data if item.role == 'assistant']

            full_response = assistant_messages[0].content[0].text.value

            # final response
            message_placeholder.markdown(full_response)

            # add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
             
    
