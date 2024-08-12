import streamlit as st
import os
import json
# Note: DALL-E 3 requires version 1.0.0 of the openai-python library or later
from openai import AzureOpenAI

from dotenv import load_dotenv
if not load_dotenv("../credentials.env"):
    load_dotenv("credentials.env")

# MODEL = os.environ['AZURE_OPENAI_MODEL_NAME']
# MODEL = "gpt-35-turbo"

SYSTEM_DEFAULT_PROMPT = "You are a helpful assistant."


if "info" not in st.session_state:
    st.session_state.info = None
if "SYSTEM_PROMPT" not in st.session_state:
    st.session_state.SYSTEM_PROMPT = SYSTEM_DEFAULT_PROMPT
if "messages" not in st.session_state:
    st.session_state.messages = [
                    {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                ]
if "model" not in st.session_state:
    st.session_state.model = "gpt-x"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 200



#################################################################################
# App elements


st.title("GPT 4 Vision - GPT-X Demo")

with st.sidebar:
    st.caption("Car status")
    # vehicle_feature_display = st.json(st.session_state.vehicle_features)
    # st.session_state.model = st.selectbox("Select a model", ["gpt-4-turbo"])
    # st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
    st.session_state.max_tokens = st.slider("Max tokens", 10, 4000, 200, 5)
        
    st.text_area("Enter your SYSTEM message", key="system_custom_prompt", value=st.session_state.SYSTEM_PROMPT)
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
    elif message["role"] == "tool":
        pass
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# create text input for user to enter a prompt
prompt = st.text_input("Enter your prompt", value="Describe this picture:")

if prompt.strip() == "":
    prompt = "Describe this picture:"

import base64
import requests
# uplod images and get the base64 encoded string
uploaded_file = st.file_uploader("Upload a file to ground your answers", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    uploaded_file.seek(0)
    ulpoaded_file_bytes = uploaded_file.read()
    base64_encoded_image = base64.b64encode(ulpoaded_file_bytes).decode('ascii')

    # display the image
    st.image(uploaded_file, caption="Uploaded Image", width=300)
    
    base_url = f"{os.environ['AZURE_OPENAI_ENDPOINT']}openai/deployments/{st.session_state.model}"
    headers = {
        "Content-Type": "application/json",
        "api-key": os.environ["AZURE_OPENAI_API_KEY"],
    }
    endpoint = f"{base_url}/chat/completions?api-version=2023-12-01-preview" 
    data = { 
    "messages": [ 
        { "role": "system", "content": st.session_state.SYSTEM_PROMPT }, # Content can be a string, OR 
        { "role": "user", "content": [       # It can be an array containing strings and images. 
            prompt, 
            { "image": base64_encoded_image }      # Images are represented like this. 
            ] } 
        ], 
        "max_tokens": st.session_state.max_tokens  
    } 

    # Make the API call   
    response = requests.post(endpoint, headers=headers, data=json.dumps(data))   

    
    if (response.status_code != 200): 
        st.text(f"Status Code: {response.status_code}")   
        st.json(response.text) 
    else:
        msg = json.loads(response.text)["choices"][0]["message"]
        st.text_area(f'Message:',value=msg["content"], height=200)
        st.json(response.text) 


# # Accept user input
# if prompt := st.chat_input("Issue a command?"):
#     if prompt:
        
#         # Display user message in chat message container
#         with st.chat_message("user"):
#             st.markdown(prompt)
#             st.session_state.messages.append({"role": "user", "content": prompt})
        
#         # Display assistant response in chat message container
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""

#             client = AzureOpenAI(
#                 api_version="2023-09-01-preview",
#                 azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
#                 api_key=os.environ["AZURE_OPENAI_API_KEY"],
#             ) 
            
#             response = client.chat.completions.create(
#                 model = st.session_state.model ,
#                 messages=st.session_state.messages,
#                 # stream=True,
#                 temperature=st.session_state.temperature,
#                 max_tokens=st.session_state.max_tokens,
#                 tools = tools,
#                 tool_choice = "auto",  # auto is default, but we'll be explicit
#             )
#              # Step 1: Sending the conversation and available functions to the model
#             # response = client.chat.completions.create(
#             #     model = os.getenv("OPENAI_API_DEPLOY"), # model = "Azure OpenAI deployment name".
#             #     messages = messages,
#             #     tools = tools,
#             #     tool_choice = "auto",  # auto is default, but we'll be explicit
#             # )

#             response_message = response.choices[0].message    
#             tool_calls = response_message.tool_calls

#             # Fix as per the issue # https://github.com/openai/openai-python/issues/703
#             response_message = json.loads(response.choices[0].message.model_dump_json())
#             if response_message["content"] is None:
#                 response_message["content"] = ""
#             if response_message["function_call"] is None:
#                 del response_message["function_call"]

#             # st.markdown(response.model_dump_json(indent=2))

#             if tool_calls:
#                 # Step 3: Extending conversation with a function reply        
#                 # messages.append(response_message)
#                 st.session_state.messages.append(response_message)

#                 # Step 4: Sending each function's response to the model
#                 for tool_call in tool_calls:
#                     function_name = tool_call.function.name
#                     function_to_call = available_functions[function_name]
#                     if function_name == "set_vehicle_feature_on_off":
#                         action = "switch"
#                     elif function_name == "set_vehicle_feature_up_down":
#                         action = "roll"
#                     function_args = json.loads(tool_call.function.arguments)
#                     function_response = function_to_call(
#                         feature = function_args.get("feature"),
#                         status = function_args.get("status"),
#                         action = action
#                     )
#                     if function_response:
#                         function_response = f"{list(function_response.keys())[0]} is {list(function_response.keys())[0]}"
#                     st.session_state.messages.append(
#                         {
#                             "tool_call_id": tool_call.id,
#                             "role": "tool",
#                             "name": function_name,
#                             "content": function_response
#                         }
#                     )
#                 # Step 5: Sending the updated conversation to the model
#                 second_response = client.chat.completions.create(
#                     model = st.session_state.model,
#                     messages=st.session_state.messages,
#                     # stream=True,
#                     temperature=st.session_state.temperature,
#                     max_tokens=st.session_state.max_tokens,
#                 )
#                 # for part in second_response:
#                 #     full_response += part.choices[0].delta.content or ""
#                 #     message_placeholder.markdown(full_response + "▌")
#                 full_response = second_response.choices[0].message.content
#             else:
#                 # for part in response:
#                 #     full_response += part.choices[0].delta.content or ""
#                 #     message_placeholder.markdown(full_response + "▌")
#                 full_response = response.choices[0].message.content

#             # final response
#             message_placeholder.markdown(full_response)

#             # add assistant response to chat history
#             st.session_state.messages.append({"role": "assistant", "content": full_response})
             
    

