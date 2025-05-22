import os

from genai_utility import LLMManager
from general_utility import WebCrowler

import json
import logging

import gradio as gr

from src.ChatAppGemini.gemini_api_chatbot import model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = LLMManager()

input_url="https://huggingface.co"

# ed = WebCrowler(input_url)
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
    response = llm.ollama_client.chat.completions.create(
        model=llm.OLLAMA_MODEL,
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
    #user_prompt += f"And carefully translate the content into user understandable modern Tamil language in same structure."
    user_prompt += get_all_details(url)
    user_prompt = user_prompt[:5_000] # Truncate if more than 5,000 characters
    return user_prompt

# def create_brochure(company_name, url):
#     response = llm.ollama_client.chat.completions.create(
#         model=llm.OLLAMA_MODEL,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
#           ]
#     )
#     return response.choices[0].message.content

def create_brochure_ollama_stream(company_name, url):
    stream = llm.ollama_client.chat.completions.create(
        model=llm.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": get_brochure_user_prompt(company_name, url)}
          ],
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result


#print(create_brochure("HuggingFace", "https://huggingface.co"))

view = gr.Interface(
    fn=create_brochure_ollama_stream,
    inputs=[gr.Textbox(label="Company Name:"), gr.Textbox(label="URL:")],
    outputs=[gr.Markdown(label="Response: ")],
    flagging_mode="never"
)

view.launch(inbrowser=True)