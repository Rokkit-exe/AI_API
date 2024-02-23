import gradio as gr
import time
from api.mistral_api_requests import chat, stream_chat
from api.openai_api_requests import chat_completion

def wait(text):
    loading = True
    while loading:
        time.sleep(5)
        loading = False
    return "done"

with gr.Blocks() as demo:
    with gr.Tab("Mistral API"):
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                def respond(message, chat_history):
                    bot_message = chat(message)
                    chat_history.append((message, bot_message))
                    return "", chat_history
                
                chatbot = gr.Chatbot(
                    label="Mistral", 
                    show_label=True,
                    avatar_images=["person-5.webp", "mistral-ai-icon-logo.webp"],
                )
                msg = gr.Textbox(
                    placeholder="Type your message here...", 
                    label="User Message", 
                    show_label=True,
                    interactive=True,
                )
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
                clear = gr.ClearButton([msg, chatbot], size="sm") 
            with gr.Column(scale=1) as col2:
                model_dropdown = gr.Dropdown(
                    choices=["mistral-tiny", "mistral-small", "mistral-medium"],
                    label="Model",
                    value="mistral-tiny",
                    show_label=True,
                    interactive=True,
                )
                role_dropdown = gr.Dropdown(
                    choices=["user", "bot"],
                    label="Role",
                    value="user",
                    show_label=True,
                    interactive=True,
                )
    with gr.Tab("OpenAI API"):
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                def respond(model, role, message, chat_history):
                    messages = [
                        {"role": "system", "content": role},
                        {"role": "user", "content": message},
                    ]
                    print(messages)
                    bot_message = chat_completion(model=model, messages=messages, max_tokens=1000)
                    chat_history.append((message, bot_message))
                    return "", chat_history
                
                chatbot = gr.Chatbot(
                    label="GPT", 
                    show_label=True,
                    avatar_images=["person-5.webp", "OpenAI_Logo.webp"],
                )
                msg = gr.Textbox(
                    placeholder="Type your message here...", 
                    label="User Message", 
                    show_label=True,
                    interactive=True,
                )
                clear = gr.ClearButton([msg, chatbot], size="sm")
            with gr.Column(scale=1) as col2:
                model_dropdown = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo-preview"],
                    label="Model",
                    value="gpt-3.5-turbo",
                    show_label=True,
                    interactive=True,
                )
                role_textbox = gr.Textbox(
                    label="Role",
                    value="You are a useful assistant",
                    show_label=True,
                    interactive=True,
                )
                max_tokens_slider = gr.Slider(
                    minimum=10, 
                    maximum=1000,
                    value=500,
                    label="Max Tokens", 
                    show_label=True,
                    interactive=True,
                )
                msg.submit(respond, [model_dropdown, role_dropdown, msg, chatbot], [msg, chatbot])
    with gr.Tab("Local LLM"):
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                chatbot = gr.Chatbot(
                    label="mistra-tiny", 
                    show_label=True,
                    avatar_images=["person-5.webp", "mistral-ai-icon-logo.webp"],
                )
                msg = gr.Textbox(
                    placeholder="Type your message here...", 
                    label="User Message", 
                    show_label=True, 
                )
                clear = gr.ClearButton([msg, chatbot], size="sm") 

                def respond(message, chat_history):
                    bot_message = chat(message)
                    chat_history.append((message, bot_message))
                    return "", chat_history
                
                msg.submit(respond, [msg, chatbot], [msg, chatbot])
            with gr.Column(scale=1) as col2:
                available_models = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
                model_dropdown = gr.Dropdown(
                    choices=available_models, 
                    label="Model", 
                    show_label=True, 
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, 
                    maximum=1.0,
                    value=0.7,
                    label="Temperature", 
                    show_label=True,
                    interactive=True,
                )
                max_tokens_slider = gr.Slider(
                    minimum=10, 
                    maximum=500,
                    value=250,
                    label="Max Tokens", 
                    show_label=True,
                    interactive=True,
                )
                device_dropdown = gr.Dropdown(
                    choices=["cpu", "cuda"], 
                    label="Device", 
                    show_label=True, 
                )
                submit = gr.Button(value="Submit", size="sm")
                submit.click(wait, submit, show_progress="minimal")


demo.launch()

