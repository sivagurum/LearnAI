import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Configure Gemini API key
# You should set the GOOGLE_API_KEY environment variable.
# You can do this in your terminal:
# export GOOGLE_API_KEY="YOUR_API_KEY"
# Or, if you have a .env file:
# from dotenv import load_dotenv
# load_dotenv()  # Load variables from .env file
#If you don't have GOOGLE_API_KEY as an environment variable, you will get an error.

dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)


GOOGLE_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY environment variable is not set. Please set it and restart the app.")
    # Stop execution.  Important for Streamlit!
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize the Gemini chat model
model = genai.GenerativeModel('gemini-2.0-flash')  # Or 'gemini-pro-vision' for multimodal

# Function to convert Streamlit message format to Gemini content format
def _convert_to_gemini_content(messages):
    gemini_content = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        gemini_content.append(genai.Content(role=role, parts=[content]))  # Wrap content in a list
    return gemini_content

# Initialize Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Streamlit UI
st.title("Gemini Chatbot")
st.caption("Powered by Google's Gemini Pro")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("What's on your mind?"):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get Gemini response
    try:
        gemini_messages = _convert_to_gemini_content(st.session_state.messages[:-1]) # convert history
        chat = model.start_chat(history=gemini_messages)
        response = chat.send_message(prompt)
        response_text = response.text
    except Exception as e:
        response_text = f"Sorry, there was an error: {e}"
        st.error(response_text)

    # Add Gemini response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
    with st.chat_message("assistant"):
        st.markdown(response_text)

# Add a button to clear the chat history
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()