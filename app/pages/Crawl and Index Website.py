import streamlit as st
import requests
from bs4 import BeautifulSoup
import os
import re
import urllib3
import random
import time

st.title('Tool - crawl and index website as KB')

# Disable warnings from urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


import doc_utils as doc_utils

all_pages = {}



def get_menu_identifiers(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        print(f"requesting TEXT {url}")
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        _html = soup.body

        # _html = str(_html)[0: len(str(_html)) if len(str(_html)) < 10000 else 10000]
        # # st.write(soup)
        _messages = [
                    {"role": "system", "content": """You are an HTML and UX specialist. Your task is to locate page elements in HTML."""},
                    {"role": "user", "content": f"""
                    Please locate main menu and left menu in the ## HTML PAGE ## below. Output just a div identifier or class name as JSON with keys: main_menu and left_menu and values as respective identifiers or class names.

                    ## HTML PAGE ##: 
                    {_html}
                    """},
                    
                ]
            

        res = doc_utils.do_query(messages=_messages, deployment="gpt-4o", temperature=0.2, max_tokens=3000)

        import json
        res_json = doc_utils.extract_json(res)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return res_json

def get_text_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    try:
        print(f"requesting TEXT {url}")
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')


        # Extract the title
        title = 'unknown'
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



        content_div = soup.find('div', class_='rightSide')
        if content_div:
            return title, content_div.get_text()
        else:
            return None, None
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None, None

def check_if_url_is_valid(url, base):
    # get main domain from the URL
    is_valid = False
    if base in url:
        is_valid = True
    else:
        return False
    
    if url.endswith(".pdf"):
        return False
    if url.endswith(".jpg"):
        return False
    if url.endswith(".png"):
        return False
    if url.endswith(".jpeg"):
        return False
    if url.endswith(".gif"):
        return False
    if url.endswith(".svg"):
        return False
    if url.endswith(".doc"):
        return False
    if url.endswith(".docx"):
        return False
    if url.endswith(".xls"):
        return False
    if url.endswith(".xlsx"):
        return False
    if url.endswith(".ppt"):
        return False
    if url.endswith(".pptx"):
        return False
    if url.endswith(".zip"):
        return False
    

    
    # remove http:// or https://   
    _base = st.session_state['url'].replace("http://", "").replace("https://", "")
    _base = _base.split("/")[0]
    # TODO: Add more checks

    is_valid =  url.startswith(f'http://{_base}') or url.startswith(f'https://{_base}')

    return is_valid
    

def get_links_from_submenu(url, base, menu_identifiers = None, level = 0):
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

    try:
        st.write(f"requesting {url}")
        response = requests.get(url, headers=headers, verify=False)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')

        # get all html content within the body tag
        _html = soup.body
        _base = base

        links = []
        # menu_identifiers contains div classes or ids which where identified by LLM
        if menu_identifiers is not None:
            for k,menu_class_name  in menu_identifiers.items():
                menu_soup = soup.find('div', class_=menu_class_name)
                if menu_soup is None:
                    menu_soup = soup.find('div', id=menu_class_name)
                try:
                    _links = menu_soup.find_all('a', href=True)
                    for link in _links:
                        if check_if_url_is_valid(link['href'], _base):
                            links.append(link['href'])
                            all_pages[link['href']] = _html.text
                except:
                    # not all pages has same menu structure
                    pass
        else:
            # if the menu_identifiers are empty -> get all links in page instead
            try:
                # get all a href links
                _links = soup.find_all('a', href=True)
                
                for link in _links:
                    if check_if_url_is_valid(link['href'], _base):
                        links.append(link['href'])
                        all_pages[link['href']] = _html.text
            except:
                pass

    
        ## Getting links though LLM (not used now since i am getting all the links from identified structure)        
        
        # _html = str(_html)[0: len(str(_html)) if len(str(_html)) < 10000 else 10000]
        # # st.write(soup)
        # _messages = [
        #             {"role": "system", "content": """You are an HTML and UX specialist. Your task is to locate page elements in HTML."""},
        #             {"role": "user", "content": f"""
        #         Please locate main menu and left menu in the ## HTML PAGE ## below. Output just a menu item name and corresponding URL in JSON format.
        #         ## HTML PAGE ##: 
        #         {_html}
        #         """},
                   
        #         ]
        

        # res = doc_utils.do_query(messages=_messages, deployment="gpt-4o", temperature=0.2, max_tokens=3000)

        # import json
        # res_json = doc_utils.extract_json(res)
        # #ads

        

        # # st.write(res_json)
        # links  = [ ]
        # for key in res_json:
        #     # st.write(f"Key: {key}, Value: {res_json[key]}")
        #     if isinstance(res_json[key], list):
        #         for item in res_json[key]:
        #             if check_if_url_is_valid(item["url"]):
        #                 links.append(item["url"])



        return links
    except requests.RequestException as e:
        return f"An error occurred: {e}"

def crawl_submenus(url, base,  menu_identifiers, depth=0, max_depth=3, visited=None):
    if visited is None:
        visited = set()
    
    if depth > max_depth:
        return []
    
    links = get_links_from_submenu(url, base, menu_identifiers)
    if (not isinstance(links,list)):
        return []
    unique_links = set(links) - visited
    visited.update(unique_links)
    
    all_links = list(unique_links)

    # st.write(f"{'-' * depth} {url}")
    
    for link in unique_links:
        all_links.extend(crawl_submenus(link, base, menu_identifiers, depth + 1, max_depth, visited))
    
    return all_links

    
    return all_links

if 'url' not in st.session_state:
    st.session_state['url'] = None
if 'index_filled' not in st.session_state:
    st.session_state['index_filled'] = False
if 'base' not in st.session_state:
    st.session_state['base'] = None

# Input text field for URL address
if st.session_state.index_filled:
    st.write("Index already filled go to On Your Data.")
else:
    url = st.text_input('Enter URL address', value=st.session_state['url'], key='url')
    st.caption("URL is the address of the website you want to crawl. For example: https://www.example.com/some-page/subpage/etc")
    base = st.text_input('Enter base URL address', value=st.session_state['base'], key='base')
    st.caption("Base URL is the domain name of the website. For example: https://www.example.com to crawl only this domain.")

    if st.button('Submit') and st.session_state['url'] is not None:
        # st.session_state['url'] = url
        st.write(f'Your input: {st.session_state["url"]}')

        # # Parse the web
        # title, text = get_text_from_url(st.session_state['url'])
        # st.header(title)
        # st.write(text)

        # extract menu identifiers
        menu_identifiers = get_menu_identifiers(st.session_state['url'])
        st.write(menu_identifiers)
        

        with st.spinner("Crawling the website..."):
            start_time = time.time()
            with st.expander("Crawled links", expanded=False):
                # Get links from submenu and crawl submenus
                all_links = crawl_submenus(st.session_state['url'], st.session_state['base'],menu_identifiers, max_depth=3)
            end_time = time.time()
        # st.write(all_links)
        elapsed_time = end_time - start_time
        st.write(f"Parsing done - found {len(all_links)} links. completed in {elapsed_time:.2f} seconds")

       
        with st.expander("Links"):
            for link in all_links:
                st.write(link)

        
        import doc_utils as doc_utils
        res = doc_utils.create_index(recreate=True)
        st.success(f"Index {res.name} re-created succesfully.")


        progress_bar = st.progress(0)
        total_links = len(all_links)

        if not os.path.exists("output"):
            os.makedirs("output")
        paths = []
        idx = 0
        for title, text in all_pages.items():
            # st.write(f"URL: {title}") # URL
            # st.write(f"Text: {text}")
            
            safe_title = re.sub('\\s+', ' ', title)
            safe_title = safe_title.strip().replace(" ", "_").replace("/", "_").replace(":", "_")
            safe_title = str(random.randint(10000, 99999)) + safe_title # sometimes title is not found
            # check length and if it is too long, truncate
            if len(safe_title) > 60:
                safe_title = safe_title[:60]+str(random.randint(1000, 9999))
    
            with open(f"output/{safe_title}.txt", "w") as f:
                f.write(text)
            paths.append(f"output/{safe_title}.txt")
            progress_bar.progress((idx + 1) / total_links)
            idx += 1

        
        # paths = []
        # for idx, link in enumerate(all_links):
        #     try:
        #         title, text = get_text_from_url(link)
        #         if title is not None:
        #             # st.header(title)
        #             # save file to disk
            
        #             # remove spaces and special characters
        #             safe_title = re.sub('\\s+', ' ', title)
        #             safe_title = safe_title.strip().replace(" ", "_").replace("/", "_").replace(":", "_")
        #             safe_title = str(random.randint(10000, 99999)) + safe_title # sometimes title is not found
        #             # check length and if it is too long, truncate
        #             if len(safe_title) > 60:
        #                 safe_title = safe_title[:60]+str(random.randint(1000, 9999))
            
        #             with open(f"output/{safe_title}.txt", "w") as f:
        #                 f.write(text)
        #             paths.append(f"output/{safe_title}.txt")

                    
        #         else:
        #             st.write(f"Error parsing {link} - no title found")
        #     except Exception as e:
        #         st.write(f"Error parsing {link} - {e}")
        #         pass
        #     # Update progress bar
        #     progress_bar.progress((idx + 1) / total_links)

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