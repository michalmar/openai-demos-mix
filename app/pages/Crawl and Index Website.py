import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import re
import urllib3

st.title('Tool - crawl and index website as KB')

# Disable warnings from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


import doc_utils as doc_utils

def get_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        print(f"requesting TEXT {url}")
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')


        # Extract the title
        title = ''
        if soup.title and soup.title.string:
            title = soup.title.string
        else:
            # Try to find the first <h1> tag
            h1_tag = soup.find('h1')
            if h1_tag:
                title = h1_tag.get_text(strip=True)
            else:
                h2_tag = soup.find('h2')
                if h2_tag:
                    title = h2_tag.get_text(strip=True)



        content_div = soup.find('div', class_='contentWrap')
        if content_div:
            return title, content_div.get_text()
        else:
            content_div = soup.find('div', class_='contentBox mainContentBox useTableStyle')
            if content_div:
                return title, content_div.get_text()
            else:
                return None, None
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

def check_if_url_is_valid(url):
    # get main domain from the URL
    
    # remove http:// or https://   
    _base = st.session_state['url'].replace("http://", "").replace("https://", "")
    _base = _base.split("/")[0]
    # TODO: Add more checks
    return url.startswith(f'http://{_base}') or url.startswith(f'https://{_base}')
    

def get_links_from_submenu(url, level = 0):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    try:
        st.write(f"requesting {url}")
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        # st.write(soup)
        _messages = [
                    {"role": "system", "content": """You are an HTML and UX specialist. Your task is to locate page elements in HTML."""},
                    {"role": "user", "content": f"""
                Please locate main menu and left menu in the ## HTML PAGE ## below. Output just a menu item name and corresponding URL in JSON format.
                ## HTML PAGE ##: 
                {soup}
                """},
                   
                ]
        

        res = doc_utils.do_query(messages=_messages, deployment="gpt-4o", temperature=0.2, max_tokens=3000)

        import json
        res_json = doc_utils.extract_json(res)
        #ads

        # st.write(res_json)
        links  = [ ]
        for key in res_json:
            # st.write(f"Key: {key}, Value: {res_json[key]}")
            if isinstance(res_json[key], list):
                for item in res_json[key]:
                    if check_if_url_is_valid(item["url"]):
                        links.append(item["url"])
        return links
    except requests.RequestException as e:
        return f"An error occurred: {e}"

def crawl_submenus(url, depth=0, max_depth=3, visited=None):
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return []
    
    links = get_links_from_submenu(url)
    if (not isinstance(links,list)):
        return []
    unique_links = set(links) - visited
    visited.update(unique_links)
    
    all_links = list(unique_links)

    # st.write(f"{'-' * depth} {url}")
    
    for link in unique_links:
        all_links.extend(crawl_submenus(link, depth + 1, max_depth, visited))
    
    return all_links

    
    return all_links

if 'url' not in st.session_state:
    st.session_state['url'] = None
if 'index_filled' not in st.session_state:
    st.session_state['index_filled'] = False

# Input text field for URL address
if st.session_state.index_filled:
    st.write("Index already filled go to On Your Data.")
else:
    url = st.text_input('Enter URL address', value=st.session_state['url'], key='url')

    if st.button('Submit') and st.session_state['url'] is not None:
        # st.session_state['url'] = url
        st.write(f'Your input: {st.session_state["url"]}')

        # # Parse the web
        # title, text = get_text_from_url(st.session_state['url'])
        # st.header(title)
        # st.write(text)

        with st.spinner("Crawling the website..."):
            with st.expander("Crawled links", expanded=False):
                # Get links from submenu and crawl submenus
                all_links = crawl_submenus(st.session_state['url'], max_depth=3)
        # st.write(all_links)
        st.write(f"Parsing done - found {len(all_links)} links.")

        with st.expander("Links"):
            for link in all_links:
                st.write(link)

        
        import doc_utils as doc_utils
        res = doc_utils.create_index(recreate=True)
        st.success(f"Index {res.name} re-created succesfully.")


        progress_bar = st.progress(0)
        total_links = len(all_links)
        
        paths = []
        for idx, link in enumerate(all_links):
            title, text = get_text_from_url(link)
            if title is not None:
                # st.header(title)
                # save file to disk
                if not os.path.exists("output"):
                    os.makedirs("output")
        
                # remove spaces and special characters
                safe_title = re.sub('\\s+', ' ', title)
                safe_title = safe_title.strip().replace(" ", "_").replace("/", "_").replace(":", "_")
        
                with open(f"output/{safe_title}.txt", "w") as f:
                    f.write(text)
                paths.append(f"output/{safe_title}.txt")

                
            else:
                st.write(f"Error parsing {link}")
        
            # Update progress bar
            progress_bar.progress((idx + 1) / total_links)

        st.success(f"Downloaded {len(paths)} files.")

        with st.spinner(f"Uploading & chunking {'(semantically)' if False else ''}..."):
            num_docs, num_chunks = doc_utils.process_and_upload_files(paths, 10, use_semantic_chunking=False)
            st.success(f"uploaded: {num_docs} docs in {num_chunks} chunks")
        
        st.session_state.index_filled = True
        
        # refresh the page
        st.rerun()

        # # Get links from submenu
        # links = get_links_from_submenu(st.session_state['url'])
        # st.write(links)