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

# VISION
vision = st.Page("pages/GPT-X.py", title="GPT-4 Vision", icon=":material/warning:")
dalle = st.Page("pages/Dall-e 3.0.py", title="Dall-e 3.0", icon=":material/image:")


# FUNCTIONS

func_db = st.Page("pages/ChatGPT-Functions-DB.py", title="Search", icon=":material/extension:")
func_weather = st.Page("pages/ChatGPT-Functions-Weather.py", title="History", icon=":material/extension:")
func_functions = st.Page("pages/ChatGPT-Functions.py", title="Beta", icon=":material/extension:")




delete_page = st.Page("pages/Dall-e 3.0.py", title="Delete entry", icon=":material/delete:")

pg = st.navigation([home_page, chat, delete_page])

pg = st.navigation(
    {
        "Home": [home_page],
        "Chat": [chat, rag],
        "Vision": [vision, dalle],
        "Misc": [func_functions, settings],
    }
)



st.set_page_config(page_title="Data manager", page_icon=":material/edit:")

pg.run()

