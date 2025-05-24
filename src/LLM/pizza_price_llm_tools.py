import json

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

# openai_api_key = os.getenv('OPENAI_API_KEY')
# if openai_api_key:
#     print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
# else:
#     print("OpenAI API Key not set")

# MODEL = "gpt-4o-mini"
# openai = OpenAI()

# OLLAMA
MODEL = "llama3.2"
openai = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

system_message = """You are in charge of a portion of the help desk of an SivaPizza retail pizza store.
                    Only answer inquiries that are directly within your area of expertise, 
                    from the company's perspective.
                    Do not try to help for personal matters or other than SivaPizza.
                    Do not mention what you can NOT do. Only mention what you can do.
                    Give a short, courteous answers, no more than 1 sentence.
                    Always be accurate and provide only below trained information. If you dont know the answer, say so.
                    
                    Information:
                    - Shop time: Monday to Saturday 10:00 AM to 9:00 PM EST
                    - Shop location: 123, Siva Street, Halifax, NS B3L4P2
                    - Menu: Classic Margherita, Pepperoni Deluxe, BBQ Chicken, Veggie Supreme, and our signature SuperPizza
                """

# def chat(message, history):
#     messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
#     response = openai.chat.completions.create(model=MODEL, messages=messages)
#     print(response)
#     return response.choices[0].message.content

#gr.ChatInterface(fn=chat, type="messages").launch()

price_lists = {"classic margherita": 12, "pepperoni deluxe":10, "bbq chicken":13, "veggie supreme":14, "super pizza":15}

def get_price(item: str):
    return price_lists.get(item.lower(), "UnKnown")

print(get_price("Classic Margherita"))

process_function = {
    "name": "get_price",
    "description": "Get the price of piza. Call this whenever you need to know the pizza price, for example if customer asks 'How much is a pizza price'",
    "parameters": {
        "type": "object",
        "properties": {
            "pizza_name": {
                "type": "string",
                "description": "The pizza that customer wants the prize"
            }
        },
        "required": ["pizza_name"],
        "additionalProperties": False
    }
}

# And this is included in a list of tools
tools = [{"type": "function", "function": process_function}]

def handle_tool_call(message):
    tool_call = message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    pizza_name = arguments.get('pizza_name')
    price = get_price(pizza_name)
    response = {
        "role": "tool",
        "content": json.dumps({"pizza_name": pizza_name, "price": price}),
        "tool_call_id": tool_call.id
    }
    return response, pizza_name



def chat(message, history):
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=tools)

    if response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message

        response, pizza_name = handle_tool_call(message)
        messages.append(message)
        messages.append(response)
        response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

gr.ChatInterface(fn=chat, type="messages").launch()