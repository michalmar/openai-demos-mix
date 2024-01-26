import streamlit as st
import requests
from bs4 import BeautifulSoup

st.title('Demo for website chat')

st.write('Comming soon...')

# # url = st.text_input('Enter a web URL')

# # def fix_url(url):
# #     if not url.startswith('http://') and not url.startswith('https://'):
# #         if url.startswith('www.'):
# #             url = 'https://' + url
# #         else:
# #             url = 'https://' + 'www.' + url
# #     return url

# # # check if the url is valid
# # if url == '':
# #     pass
# # else:
# #     st.caption(f'Parse the HTML of a webpage: {url}')
# #     # Get the HTML of the webpage
# #     response = requests.get(fix_url(url))    

# #     # Parse the HTML with BeautifulSoup
# #     soup = BeautifulSoup(response.text, 'html.parser')

# #     # Extract all the text
# #     texts = soup.get_text()

# #     # Display the texts
# #     st.write(texts)

# import requests
# from bs4 import BeautifulSoup

# # The URL of the page you want to scrape
# url = 'https://www.reddit.com/r/Python/'

# # Perform an HTTP GET request to the URL
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
# response = requests.get(url, headers=headers)

# # Check if the request was successful
# if response.status_code == 200:
#     # Create a Beautiful Soup object and specify the parser
#     soup = BeautifulSoup(response.text, 'html.parser')

#     # Find all the headline elements
#     # Reddit headlines are contained within <h3> tags with class '._eYtD2XCVieq6emjKBH3m'
#     # Please note that class names can change, so you might need to inspect the page
#     headlines = soup.find_all('h3', class_='_eYtD2XCVieq6emjKBH3m')

#     # Print each headline text
#     for headline in headlines:
#         st.write(headline.text)
# else:
#     st.write('Failed to retrieve the webpage')