from openai import OpenAI
import os
import dotenv
dotenv.load_dotenv()
import requests
import json


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
client = OpenAI()

print(client.api_key)

if False:
    assistant = client.beta.assistants.create(
        name="Math Tutor",
        instructions="You are a personal math tutor. Write and run code to answer math questions.",
        tools=[{"type": "code_interpreter"}],
        model="gpt-3.5-turbo"
    )

if False:
    # create a thread
    thread = client.beta.threads.create()

if False:
    # send a message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="I need to solve the equation `3x + 11 = 14`. Can you help me?"
    )

if False:
    # run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions="Please address the user as Frank."
    )

if False:
    # retrieve the assistant's response
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )

if False:
    # print the assistant's response
    messages = client.beta.threads.messages.list(
        thread_id=thread.id
    )

def request(model, messages):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }

    data = {
        "model": model,
        "messages": messages
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    return response.json()


test_messages = [
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
]

def chat_completion(model, messages, max_tokens=1000):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.content

if __name__ == "__main__":
    print(json.dumps(request(), indent=4))