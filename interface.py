import gradio as gr
import time
from api.mistral_api_requests import chat, stream_chat
from api.openai_api_requests import chat_completion
from utils import install_model, get_available_models
from stable_diffusion import SD_Pipeline
user_image = "images/person-5.webp"
mistral_image = "images/mistral-ai-icon-logo.webp"
openai_image = "images/OpenAI_Logo.webp"

def wait(text):
    loading = True
    while loading:
        time.sleep(5)
        loading = False
    return "done"

with gr.Blocks() as demo:
    with gr.Tab("Mistral API") as mistral_tab:
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                def respond(message, chat_history):
                    bot_message = chat(message)
                    chat_history.append((message, bot_message))
                    return "", chat_history

                chatbot = gr.Chatbot(
                    label="Mistral", 
                    show_label=True,
                    avatar_images=[user_image, mistral_image],
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
                mistral_model_dropdown = gr.Dropdown(
                    choices=["mistral-tiny", "mistral-small", "mistral-medium"],
                    label="Model",
                    value="mistral-tiny",
                    show_label=True,
                    interactive=True,
                )
                mistral_role_dropdown = gr.Dropdown(
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
                    avatar_images=[user_image, openai_image],
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
                msg.submit(respond, [model_dropdown, role_textbox, msg, chatbot], [msg, chatbot])
# local stable diffision TAB
    with gr.Tab("Local Stable Diffision"):
        with gr.Row() as row:
            with gr.Column(scale=1) as col1:
                pipeline = SD_Pipeline()
                load_model_button = gr.Button(value="Load Model", size="sm")
                load_model_button.click(pipeline.load_model, show_progress="minimal")
                text_to_image_interface = gr.Interface(
                    fn=pipeline.generate_image,
                    inputs=gr.Textbox(placeholder="Type your prompt here...", label="Prompt", show_label=True),
                    outputs=gr.Image(label="Generated Image"),
                )
                save_image_button = gr.Button(value="Save Image", size="sm")
                save_image_button.click(pipeline.save_image, show_progress="minimal")


# local LLM TAB
    with gr.Tab("Local LLM"):
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                chatbot = gr.Chatbot(
                    label="local-llm", 
                    show_label=True,
                    avatar_images=[user_image, mistral_image],
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

# install models TAB
    with gr.Tab("Install models"):
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                models = get_available_models()
                model_types = list(models.keys())
                def update_model_dropdown(model_type):
                    print(model_type)
                    model_dropdown.choices = [model["name"] for model in models[model_type]]

                def update_label(model):
                    model_info.value = model
                
                model_types_dropdown = gr.Dropdown(
                    label="Model Type",
                    value=model_types[0],
                    choices=model_types,
                    show_label=True,
                    interactive=True,
                )
                model_dropdown = gr.Dropdown(
                    label="Model",
                    value="",
                    choices=[],
                    show_label=True,
                    interactive=True,
                )
                model_types_dropdown.change(fn=update_model_dropdown, inputs=model_dropdown)
                model_info = gr.Label(
                    label="Model Info",
                    value="",
                    show_label=True,
                )
                model_dropdown.change(fn=update_label, inputs=model_info)
                
            with gr.Column(scale=1) as col2:
                model_name_textbox = gr.Textbox(
                    label="Model Name",
                    value="gpt2",
                    show_label=True,
                    interactive=True,
                )
                model_path_textbox = gr.Textbox(
                    label="Model Path",
                    value="./models",
                    show_label=True,
                    interactive=True,
                )
                install_button = gr.Button(value="Install", size="sm")
                install_button.click(install_model, [model_name_textbox, model_path_textbox], show_progress="minimal")


demo.launch()

