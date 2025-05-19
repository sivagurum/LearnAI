import os

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import json
from typing import List
from openai import OpenAI
import logging

from torch.distributed.rpc.internal import deserialize

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

if api_key and api_key.startswith('sk-proj-') and len(api_key) > 10 :
    logger.info("API Key looks good so far")
else:
    logging.error("API key not available in *.env file !!!")

# MODEL = "gpt-4o-mini"
# openai = OpenAI()

MODEL = "llama3.2"
openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

input_url="https://huggingface.co"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class WebCrowler:
    '''Utility Class'''
    def __init__(self, url):
        self.url = url
        response = requests.get(url, headers=headers)
        self.body = response.content
        soup = BeautifulSoup(self.body, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        if soup.body:
            for irrelevent in soup.body(["script", "style", "img", "input"]):
                irrelevent.decompose()
            self.text = soup.body.get_text(separator="\n", strip=True)
        else:
            self.text = ""

        links = [ link.get('href') for link in soup.find_all('a') ]
        self.links = {link for link in links if link}

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:{self.text}\n\n"

ed = WebCrowler(input_url)
# print(ed.links)
# print(ed.get_contents())

link_system_prompt = ("You are provided with a list of links found on a webpage. \
                      You are able to decide which of the links would be most relevant to include in a brochure about the company, \
                      such as links to an About page, or a company page, or Careers/Jobs page.\n")

link_json_prompt = """
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

link_system_prompt += link_json_prompt

def get_links_users_prompt(website: WebCrowler):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += (f"""please decide which of these are relevant web links for a brochure about the company, respond with full https URL in json format.
                    Do not include Terms of Service, Privacy, email links.
                    And the result should be in {link_json_prompt} format.""")
    user_prompt += "Links (some might be relative links): \n"
    user_prompt += "\n".join(website.links)
    return user_prompt

# print(get_links_users_prompt(ed))

def get_links(url):
    website = WebCrowler(url)
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": link_system_prompt},
            {"role": "user", "content": get_links_users_prompt(website)}
        ],
        response_format={"type": "json_object"}
    )
    result = response.choices[0].message.content
    return json.loads(result)

# print(get_links(input_url))

def get_all_details(url):
    result = "Landing page:\n"
    result += WebCrowler(url).get_contents()
    links = get_links(url)
    print("Found Links:", links)
    for link in links["links"]:
        result += f"\n\n{link['type']}\n"
        result += WebCrowler(link["url"]).get_contents()
    return result

# print(get_all_details(input_url))

system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
and creates a short brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
Include details of company culture, customers and careers/jobs if you have the information."

# Or uncomment the lines below for a more humorous brochure - this demonstrates how easy it is to incorporate 'tone':

# system_prompt = "You are an assistant that analyzes the contents of several relevant pages from a company website \
# and creates a short humorous, entertaining, jokey brochure about the company for prospective customers, investors and recruits. Respond in markdown.\
# Include details of company culture, customers and careers/jobs if you have the information."

def get_brochure_user_prompt(company_name, url):
    user_prompt = f"You are looking at a company called: {company_name}\n"
    user_prompt += f"Here are the contents of its landing page and other relevant pages; use this information to build a short brochure of the company in markdown.\n"
    user_prompt += f"And carefully translate the content into user understandable modern Tamil language in same structure."
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

def create_brochure(company_name, url):
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
    )
    return response.choices[0].message.content

print(create_brochure("HuggingFace", "https://huggingface.co"))