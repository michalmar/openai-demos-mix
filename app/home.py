import streamlit as st
import os

if "info" not in st.session_state:
    st.session_state.info = None
#################################################################################
# App elements



home_page = st.Page("pages/new_home.py", title="Home", icon=":material/house:", default=True)

# CHAT
chat = st.Page("pages/ChatGPT.py", title="Chat", icon=":material/chat:")

# SETTINGS
settings = st.Page("pages/settings.py", title="Settings", icon=":material/settings:")


# RAG
rag = st.Page("pages/Chat with Data API.py", title="On your data", icon=":material/book:")
rag_custom = st.Page("pages/Chat with Data Custom RAG.py", title="Custom RAG", icon=":material/book:")

# VISION
vision = st.Page("pages/GPT-X.py", title="GPT-4 Vision", icon=":material/warning:")
dalle = st.Page("pages/Dall-e 3.0.py", title="Dall-e 3.0", icon=":material/image:")


# FUNCTIONS

func_db = st.Page("pages/ChatGPT-Functions-DB.py", title="Function Calling - Database   ", icon=":material/extension:")
func_weather = st.Page("pages/ChatGPT-Functions-Weather.py", title="Function calling - Weather", icon=":material/extension:")
func_functions = st.Page("pages/ChatGPT-Functions.py", title="Function Calling - Car", icon=":material/extension:")

crawl_web = st.Page("pages/Crawl and Index Website.py", title="Crawl & Index Website", icon=":material/extension:")



delete_page = st.Page("pages/Dall-e 3.0.py", title="Delete entry", icon=":material/delete:")

pg = st.navigation([home_page, chat, delete_page])

pg = st.navigation(
    {
        "Home": [home_page],
        "Chat": [chat, rag, rag_custom],
        "Vision": [vision, dalle],
        "Tools": [func_functions,crawl_web, settings,],
    }
)



st.set_page_config(page_title="Azure AI Demos", page_icon=":robot_face:")

pg.run()

