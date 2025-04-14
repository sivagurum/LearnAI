# filepath: /Users/sivagurumanickam/My Drive/Macbook/MyProjects/HelloWorld/streamlit_app.py
import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# Retrieve the API key from the environment
api_key = os.getenv("GEMINI_API_KEY")

# Title of the app
st.title("Gemini Chat Application")

# Input field for user message
user_message = st.text_input("Enter your message:", "")

# Button to send the message
if st.button("Send"):
    if user_message.strip():
        # Replace with your Gemini API endpoint
        gemini_api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        payload = {
             "contents": [{"parts": [{"text": user_message}]}]
        }

        params = {"key": api_key}

        # Send the message to the Gemini API
        try:
            response = requests.post(
                gemini_api_url,
                json=payload,
                params=params
            )
            response.raise_for_status() 
            response_data = response.json()

            # Display the response from the Gemini API
            if response.status_code == 200:
                #st.write("Gemini Response:", response_data.get("reply", "No reply"))
                st.write("Gemini Response:", response_data)
            else:
                st.error(f"Error: {response_data.get('error', 'Unknown error')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a message before sending.")