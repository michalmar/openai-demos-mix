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

SYSTEM_DEFAULT_PROMPT = "You are helpful AI assitant."


if "info" not in st.session_state:
    st.session_state.info = None
if "SYSTEM_PROMPT" not in st.session_state:
    st.session_state.SYSTEM_PROMPT = SYSTEM_DEFAULT_PROMPT
if "messages" not in st.session_state:
    st.session_state.messages = [
                    {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                ]
if "model" not in st.session_state:
    # st.session_state.model = "gpt-4-turbo"
    st.session_state.model = "gpt-35-turbo"
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 200


#################################################################################



st.title("ChatGPT fetches data from Database")

with st.sidebar:
        
    st.text_area("Enter your SYSTEM message", key="system_custom_prompt", value=st.session_state.SYSTEM_PROMPT)
    if st.button("Apply & Clear Memory"):
        # save the text from the text_area to SYSTEM_PROMPT
        st.session_state.SYSTEM_PROMPT = st.session_state.system_custom_prompt
        st.session_state.messages = [
                        {"role": "system", "content": st.session_state.SYSTEM_PROMPT},
                    ]
    st.caption("Refresh the page to reset to default settings")

    

st.caption("Ask: Give me details on product 680, please.")
st.caption("Ask: Create a short poem about product 680, please.")
st.caption(f"powered by Azure OpenAI's {st.session_state.model} model")
# st.caption(f"powered by Azure OpenAI's {MODEL} model")


# Example function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


import requests

def get_product_details(product_id):
    function_api_base_url = os.getenv("FUNCTION_API_BASE_URL")
    url = function_api_base_url.replace("{ID}", str(product_id))
    # url = f"{function_api_base_url}{product_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return str(response.json())  # Assuming the API returns JSON data
    else:
        return f"Error: Unable to fetch product details (Status code: {response.status_code})"

def run_conversation(user_input):
    # Step 1: send the conversation and available functions to the model
    messages = [{"role": "user", "content": f"{user_input}"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_product_details",
                "description": "Get the details of product by given ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "product_id": {
                            "type": "string",
                            "description": "The ID of the product to get details for",
                        },
                        # "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=st.session_state.model,
        messages=messages,
        tools=tools,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    # Step 2: check if the model wanted to call a function
    if tool_calls:
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_product_details": get_product_details,
        }  # only one function in this example, but you can have multiple
        messages.append(response_message)  # extend conversation with assistant's reply
        # Step 4: send the info for each function call and function response to the model
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(
                product_id=function_args.get("product_id"),
                # unit=function_args.get("unit"),
            )
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model=st.session_state.model,
            messages=messages,
        )  # get a new response from the model where it can see the function response
        return second_response
    return response
# print(run_conversation())


client = AzureOpenAI(
    api_version="2024-06-01",
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
) 

# product_details = get_product_details(680)
# st.write(product_details)

# Accept user input
user_input_db_query = st.text_input("Enter your message", key="user_input_db_query")
if (user_input_db_query != ''):
    response = run_conversation(user_input_db_query)
    st.caption("Response:")
    st.write(response.choices[0].message.content)
