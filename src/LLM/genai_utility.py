import os
import logging

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from anthropic import Anthropic, APIError as AnthropicAPIError
from pydantic import BaseModel, Field
from typing import List, Optional, Generator, Union, Dict
import gradio as gr

# Configure logging at the module level, ONCE
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv(override=True)

class LLMManager:
    def __init__(self):
        print(" *** LLMManager Initiated **")
        self.openai_client = None
        self.gemini_client = None
        self.claude_client = None
        self.ollama_openai_client = None

        # Define default models
        self.OPENAI_MODEL = "gpt-4o-mini"
        self.GEMINI_MODEL = "gemini-1.5-flash"
        self.CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
        self.OLLAMA_MODEL = "llama3.2" # Using .2 if available, adjust to 'llama3' if not

        # Attempt to initialize all clients
        self._initialize_openai_client()
        self._initialize_gemini_client()
        self._initialize_claude_client()
        self._initialize_ollama_openai_client()

    def _initialize_openai_client(self):
        """Initializes the OpenAI client."""
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not openai_api_key:
            logger.warning("OPENAI_API_KEY environment variable not set. OpenAI client will not be initialized.")
            return

        if not (openai_api_key.startswith('sk-') and len(openai_api_key) > 40):
            logger.error(
                f"Error: OPENAI_API_KEY format appears invalid. It should start with 'sk-' and be sufficiently long.")
            return

        try:
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully.")
        except OpenAIError as e:
            logger.error(f"Failed to initialize OpenAI client due to an API error: {e}")
            self.openai_client = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI client initialization: {e}")
            self.openai_client = None

    def _initialize_ollama_openai_client(self):
        """Initializes the Ollama client using an OpenAI-compatible endpoint."""
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")
        try:
            self.ollama_openai_client = OpenAI(base_url=ollama_host, api_key='ollama')
            self.ollama_openai_client.models.list() # Test connectivity
            logger.info(f"Ollama OpenAI-compatible client initialized and connected to {ollama_host} successfully.")
        except OpenAIError as e:
            logger.error(f"Failed to initialize Ollama OpenAI-compatible client due to an API error: {e}")
            logger.error(f"Ensure Ollama server is running on {ollama_host} and accessible.")
            self.ollama_openai_client = None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama OpenAI-compatible client initialization: {e}")
            self.ollama_openai_client = None

    def _initialize_gemini_client(self):
        """Initializes the Gemini client."""
        gemini_api_key = os.getenv("GOOGLE_API_KEY")

        if not gemini_api_key:
            logger.warning("GOOGLE_API_KEY not set in the environment variable. Gemini client will not be initialized.")
            return

        if not (gemini_api_key.startswith('AIza') and len(gemini_api_key) > 30):
            logger.error(
                f"Error: GOOGLE_API_KEY format appears invalid. It should start with 'AIza' and be sufficiently long.")
            return

        try:
            genai.configure(api_key=gemini_api_key)
            self.gemini_client = True
            logger.info("Gemini API configured successfully.")
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Failed to configure Gemini API key due to Google API error: {e}")
            self.gemini_client = None
        except Exception as e:
            logger.error(f"Unexpected error occurred during Gemini API configuration: {e}")
            self.gemini_client = None

    def _initialize_claude_client(self):
        """Initializes the Claude (Anthropic) client."""
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")

        if not claude_api_key:
            logger.warning("ANTHROPIC_API_KEY environment variable not set. Claude client will not be initialized.")
            return

        if not (claude_api_key.startswith('sk-ant-') and len(claude_api_key) > 50):
            logger.error(
                f"Error: ANTHROPIC_API_KEY format appears invalid. "
                f"It should start with 'sk-ant-' and be sufficiently long.")
            return

        try:
            self.claude_client = Anthropic(api_key=claude_api_key)
            logger.info("Claude client initialized successfully.")
        except AnthropicAPIError as e:
            logger.error(f"Failed to initialize Claude due to ClaudeAPI error: {e}")
            self.claude_client = None
        except Exception as e:
            logger.error(f"Unexpected error during Claude API Initialization: {e}")
            self.claude_client = None

    def _prepare_openai_params(self, messages: Union[str, List[Dict[str, str]]], model: str, max_tokens: Optional[int], temperature: Optional[float], response_format: Optional[dict], stream: bool):
        """
        Helper to prepare parameters for OpenAI and OpenAI-compatible API calls.
        Ensures messages is a list of dicts.
        """
        if isinstance(messages, str):
            messages_list = [{"role": "user", "content": messages}]
        else:
            messages_list = messages

        params = {
            "messages": messages_list,
            "model": model,
            "stream": stream
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature
        if response_format is not None:
            params["response_format"] = response_format

        return params

    def _stream_openai_text(self, config_params: Dict) -> Generator[str, None, None]:
        """Internal generator for OpenAI streaming."""
        response_stream = self.openai_client.chat.completions.create(**config_params)
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_openai_text(self, messages: Union[str, List[Dict[str, str]]], model: str = None, max_tokens: Optional[int] = None,
                             temperature: Optional[float] = None, response_format: Optional[dict] = None, stream: bool = False) -> Union[str, Generator[str, None, None], None]:
        """Generates text using the OpenAI client."""
        logger.info("*** Calling generate_openai_text() ***")
        if not self.openai_client:
            logger.error("OpenAI client not initialized. Cannot generate text.")
            return None

        if not messages or (isinstance(messages, str) and len(messages.strip()) == 0):
            logger.error("OpenAI client messages not passed or are empty. Cannot generate text.")
            return None

        used_model = model if model else self.OPENAI_MODEL

        try:
            config_params = self._prepare_openai_params(
                messages=messages,
                model=used_model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                stream=stream # Pass stream to prepare_openai_params
            )

            if stream:
                return self._stream_openai_text(config_params)
            else:
                chat_completion = self.openai_client.chat.completions.create(**config_params)
                return chat_completion.choices[0].message.content
        except OpenAIError as e:
            logger.error(f"Error during OpenAI API call with model '{used_model}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI text generation: {e}")
            return None

    def _stream_ollama_text_openai_compatible(self, config_params: Dict) -> Generator[str, None, None]:
        """Internal generator for Ollama OpenAI-compatible streaming."""
        response_stream = self.ollama_openai_client.chat.completions.create(**config_params)
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def generate_ollama_text_openai_compatible(self, messages: Union[str, List[Dict[str, str]]], model: str = None, max_tokens: Optional[int] = None,
                                    temperature: Optional[float] = None, response_format: Optional[dict] = None, stream: bool = False) -> Union[str, Generator[str, None, None], None]:
        """Generates text using the Ollama client via OpenAI-compatible API."""
        logger.info("*** Calling generate_ollama_text_openai_compatible() ***")
        if not self.ollama_openai_client:
            logger.error("Ollama OpenAI-compatible client not initialized. Cannot generate text.")
            return None

        if not messages or (isinstance(messages, str) and len(messages.strip()) == 0):
            logger.error("Ollama client messages not passed or are empty. Cannot generate text.")
            return None

        used_model = model if model else self.OLLAMA_MODEL

        try:
            config_params = self._prepare_openai_params(
                messages=messages,
                model=used_model,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
                stream=stream # Pass stream to prepare_openai_params
            )
            logger.debug(f"Ollama OpenAI compatible config_params: {config_params}")

            if stream:
                return self._stream_ollama_text_openai_compatible(config_params)
            else:
                chat_completion = self.ollama_openai_client.chat.completions.create(**config_params)
                return chat_completion.choices[0].message.content
        except OpenAIError as e:
            logger.error(f"Error during Ollama OpenAI-compatible API call with model '{used_model}': {e}")
            logger.error(f"Common issues: Ollama server not running, model '{used_model}' not pulled, or incorrect base_url.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Ollama OpenAI-compatible text generation: {e}")
            return None

    def _stream_claude_text(self, params: Dict) -> Generator[str, None, None]:
        """Internal generator for Claude streaming."""
        with self.claude_client.messages.stream(**params) as stream_iterator:
            for chunk in stream_iterator:
                if chunk.type == "content_block_delta" and chunk.delta.text:
                    yield chunk.delta.text

    def generate_claude_text(self, messages: Union[str, List[Dict[str, str]]], model: str = None, max_tokens: Optional[int] = 500,
                             temperature: Optional[float] = 0.7, stream: bool = False) -> Union[str, Generator[str, None, None], None]:
        """Generates text using the Claude (Anthropic) client."""
        logger.info("*** Calling generate_claude_text() ***")
        if not self.claude_client:
            logger.error("Claude client not initialized. Cannot generate text.")
            return None

        if not messages or (isinstance(messages, str) and len(messages.strip()) == 0):
            logger.error("Claude client messages not passed or are empty. Cannot generate text.")
            return None

        used_model = model if model else self.CLAUDE_MODEL

        if isinstance(messages, str):
            messages_list = [{"role": "user", "content": messages}]
        else:
            messages_list = messages

        params = {
            "model": used_model,
            "messages": messages_list,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if temperature is not None:
            params["temperature"] = temperature

        try:
            if stream:
                return self._stream_claude_text(params)
            else:
                message = self.claude_client.messages.create(**params)
                return message.content[0].text
        except AnthropicAPIError as e:
            logger.error(f"Error during Claude API call with model '{used_model}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Claude text generation: {e}")
            return None

    def _stream_gemini_text(self, prompt: str, generation_config: genai.types.GenerationConfig, gemini_model_instance: genai.GenerativeModel) -> Generator[str, None, None]:
        """Internal generator for Gemini streaming."""
        response_stream = gemini_model_instance.generate_content(
            prompt,
            generation_config=generation_config,
            stream=True
        )
        logger.debug(f"response_stream: {response_stream}")
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    def generate_gemini_text(self, prompt: str, model: str = None, max_output_tokens: Optional[int] = None,
                             response_schema: Optional[BaseModel] = None, response_mime_type: Optional[str] = None,
                             temperature: Optional[float] = None, stream: bool = False) -> Union[str, Generator[str, None, None], None]:
        """
        Generates text using the Gemini API.
        Can optionally stream the response.
        """
        logger.info("*** Calling generate_gemini_text() *** ")
        if not self.gemini_client:
            logger.error("Gemini API not configured. Cannot generate text.")
            return None

        if not prompt or (isinstance(prompt, str) and len(prompt.strip()) == 0) == 0:
            logger.error(f"Gemini client prompt not passed or is empty. Cannot generate text. {prompt}")
            return None

        used_model = model if model else self.GEMINI_MODEL
        gemini_model_instance = genai.GenerativeModel(used_model)

        generation_config_params = {}
        if response_schema:
            generation_config_params["response_schema"] = response_schema
        if response_mime_type is not None:
            generation_config_params["response_mime_type"] = response_mime_type
        if temperature is not None:
            generation_config_params["temperature"] = temperature
        if max_output_tokens is not None:
            generation_config_params["max_output_tokens"] = max_output_tokens

        generation_config = genai.types.GenerationConfig(**generation_config_params)

        try:
            if stream:
                return self._stream_gemini_text(prompt, generation_config, gemini_model_instance)
            else:
                response = gemini_model_instance.generate_content(
                    prompt,
                    generation_config=generation_config,
                    stream=False
                )
                logger.debug(f"response: {response}")
                return response.text
        except google_exceptions.GoogleAPIError as e:
            logger.error(f"Error during Gemini API call with model '{used_model}': {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during Gemini text generation: {e}")
            return None

# # Initialize LLMManager globally for Gradio to use
# llm_manager = LLMManager()
#
# def generate_text_with_gradio(client_name: str, is_stream: bool, prompt_text: str):
#     """
#     Function to be exposed via Gradio for generating text.
#     """
#     if not prompt_text or len(prompt_text.strip()) == 0:
#         return "Please provide a prompt."
#
#     result = None
#     if client_name == "gemini":
#         result = llm_manager.generate_gemini_text(prompt=prompt_text, stream=is_stream)
#     elif client_name == "openai":
#         result = llm_manager.generate_openai_text(messages=prompt_text, stream=is_stream)
#     elif client_name == "claude":
#         result = llm_manager.generate_claude_text(messages=prompt_text, stream=is_stream)
#     elif client_name == "ollama":
#         result = llm_manager.generate_ollama_text_openai_compatible(messages=prompt_text, stream=is_stream)
#     else:
#         return "Invalid client selected."
#
#     if result is None:
#         return "Error: Could not generate response. Check logs for details."
#
#     if is_stream:
#         # Gradio will handle iterating over this generator
#         response = ""
#         for chunk in result:
#             response += chunk or ""
#             yield response
#     else:
#         # For non-streaming, return the complete string
#         yield result

# if __name__ == "__main__":
    # manager = LLMManager()
    # print("\n--- Testing generate_gemini_text (Non-Streaming) ---")
    # gemini_non_stream_response = manager.generate_gemini_text(prompt="Tell me a fun fact about giraffes.", max_output_tokens=50, stream=False)
    # if gemini_non_stream_response:
    #     print(f"Gemini Non-Streaming Response: {gemini_non_stream_response}")
    # else:
    #     print("Gemini non-streaming generation failed.")

    # print("\n--- Testing generate_gemini_text (Streaming) ---")
    # gemini_stream_gen = manager.generate_gemini_text(prompt="Write a short haiku about the morning.", max_output_tokens=50, stream=True)
    # if gemini_stream_gen:
    #     print("Gemini Streaming Output:")
    #     for chunk in gemini_stream_gen:
    #         print(chunk, end='', flush=True)
    #     print("\n(Gemini Streaming complete)")
    # else:
    #     print("Gemini streaming test failed.")

    # print("\n--- Testing generate_openai_text (Non-Streaming) ---")
    # openai_non_stream_response = manager.generate_openai_text(messages="What is the capital of Canada?", model="gpt-4o-mini", stream=False)
    # if openai_non_stream_response:
    #     print(f"OpenAI Non-Streaming Response: {openai_non_stream_response}")
    # else:
    #     print("OpenAI non-streaming generation failed.")

    # print("\n--- Testing generate_openai_text (Streaming) ---")
    # openai_stream_gen = manager.generate_openai_text(messages="Tell a very short story about a brave mouse.", model="gpt-4o-mini", max_tokens=100, temperature=0.8, stream=True)
    # if openai_stream_gen:
    #     print("OpenAI Streaming Output:")
    #     for chunk in openai_stream_gen:
    #         print(chunk, end='', flush=True)
    #     print("\n(OpenAI Streaming complete)")
    # else:
    #     print("OpenAI streaming test failed.")

    # print("\n--- Testing generate_ollama_text_openai_compatible (Non-Streaming) ---")
    # ollama_non_stream_response = manager.generate_ollama_text_openai_compatible(messages="Why is the sky blue?", model="llama3.2", stream=False)
    # if ollama_non_stream_response:
    #     print(f"Ollama Non-Streaming Response: {ollama_non_stream_response}")
    # else:
    #     print("Ollama non-streaming generation failed. Ensure Ollama server is running and 'llama3.2' is pulled.")

    # print("\n--- Testing generate_ollama_text_openai_compatible (Streaming) ---")
    # ollama_stream_gen = manager.generate_ollama_text_openai_compatible(messages="Tell me a simple joke.", model="llama3.2", stream=True)
    # if ollama_stream_gen:
    #     print("Ollama Streaming Output:")
    #     for chunk in ollama_stream_gen:
    #         print(chunk, end='', flush=True)
    #     print("\n(Ollama Streaming complete)")
    # else:
    #     print("Ollama streaming test failed. Ensure Ollama server is running and 'llama3.2' is pulled.")

    # print("\n--- Testing generate_claude_text (Non-Streaming) ---")
    # claude_non_stream_response = manager.generate_claude_text(messages="What are the three largest oceans?", model="claude-3-5-sonnet-20240620", stream=False)
    # if claude_non_stream_response:
    #     print(f"Claude Non-Streaming Response: {claude_non_stream_response}")
    # else:
    #     print("Claude non-streaming generation failed.")

    # print("\n--- Testing generate_claude_text (Streaming) ---")
    # claude_stream_gen = manager.generate_claude_text(messages="Write a short paragraph about the benefits of exercise.", model="claude-3-5-sonnet-20240620", stream=True)
    # if claude_stream_gen:
    #     print("Claude Streaming Output:")
    #     for chunk in claude_stream_gen:
    #         print(chunk, end='', flush=True)
    #     print("\n(Claude Streaming complete)")
    # else:
    #     print("Claude streaming test failed.")

    # Gradio Interface Setup
    # print("\n--- Launching Gradio Interface ---")
    # iface = gr.Interface(
    #     fn=generate_text_with_gradio,
    #     inputs=[
    #         gr.Dropdown(choices=["gemini", "openai", "claude", "ollama"], label="Select LLM Client", value="ollama"),
    #         gr.Checkbox(label="Stream Output", value=False),
    #         gr.Textbox(lines=5, label="Prompt", placeholder="Enter your prompt here...")
    #     ],
    #     outputs=[gr.Markdown(label="Generated Text")],
    #     title="LLM Manager Interface",
    #     description="Interact with different LLM clients (Gemini, OpenAI, Claude, Ollama) and choose streaming or non-streaming output.",
    #     flagging_mode="never"
    # )
    # iface.launch()
