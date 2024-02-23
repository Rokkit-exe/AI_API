import os
import dotenv
import asyncio


# chat completion
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# async chat completion
from mistralai.async_client import MistralAsyncClient


dotenv.load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

def chat(message, model="mistral-tiny", role="user"):

    api_key = os.environ["MISTRAL_API_KEY"]
    model=model

    client = MistralClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content=message)
    ]

    # No streaming
    chat_response = client.chat(
        model=model,
        messages=messages,
    )

    return chat_response.choices[0].message.content

def stream_chat(message, model="mistral-tiny"):
    model = "mistral-tiny"

    client = MistralClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content=message)
    ]

    # With streaming
    return client.chat_stream(model=model, messages=messages)

async def async_chat(model, messages):
    api_key = os.environ["MISTRAL_API_KEY"]
    model = "mistral-tiny"

    client = MistralAsyncClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content="What is the best French cheese?")
    ]

    # With async
    async_response = client.chat_stream(model=model, messages=messages)

    async for chunk in async_response: 
        print(chunk.choices[0].delta.content)
    

if __name__ == "__main__":
    chat("mistral-tiny", "What is the best French cheese?")
    #stream_chat("mistral-tiny", "how to declare a function in python?")
    #asyncio.run(async_chat("mistral-tiny", "What is the best French cheese?"))
