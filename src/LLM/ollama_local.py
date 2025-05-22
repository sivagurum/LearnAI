import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from openai import OpenAI

# Note: Start Local Ollama before run
# brew services start ollama

MODEL = "llama3.2"
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

input_url="https://www.engadget.com/ai/metas-behemoth-llama-4-model-might-still-be-months-away-221240585.html"

# HEADERS = {"Content-Type": "application/json"}
# messages = [
#     {"role": "user", "content": "Describe some of the business applications of Generative AI"}
# ]

#response = ollama_via_openai.chat.completions.create(
#    model=MODEL,
#    messages=messages
#)

#print(response.choices[0].message.content)

headers = {
 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
}

class Website:

    def __init__(self, url):
        """
        Create this Website object from the given url using the BeautifulSoup library
        """
        self.url = url
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.title = soup.title.string if soup.title else "No title found"
        for irrelevant in soup.body(["script", "style", "img", "input"]):
            irrelevant.decompose()
        self.text = soup.body.get_text(separator="\n", strip=True)

ed = Website(input_url)
#print(ed.title)
#print(ed.text)

system_prompt = "You are an assistant that analyzes the contents of a website \
and provides a short summary, ignoring text that might be navigation related. \
Respond in markdown."

def user_prompt_for(website):
    user_prompt = f"You are looking at a website titled {website.title}"
    user_prompt += "\nThe contents of this website is as follows; \
        please provide a short summary of this website in markdown. \
        If it includes news or announcements, then summarize these too.\n\n"
    user_prompt += website.text
    return user_prompt

#print(user_prompt_for(ed))

def messages_for(website):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(website)}
    ]

#messages_for(ed)

def summarize(url):
    website = Website(url)
    response = ollama_via_openai.chat.completions.create(
        model = MODEL,
        messages = messages_for(website)
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print(summarize(input_url))
