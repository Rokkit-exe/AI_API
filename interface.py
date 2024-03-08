import gradio as gr
import time
from api.mistral_api_requests import chat, stream_chat
from api.openai_api_requests import chat_completion
from utils import install_model, get_installed_models, uninstall_model, MODELS_PATH
from stable_diffusion import SD_Pipeline
from llm import LLM

user_image = "images/person-5.webp"
mistral_image = "images/mistral-ai-icon-logo.webp"
openai_image = "images/OpenAI_Logo.webp"

def gr_install_model(model_name, model_path, progress=gr.Progress()):
    progress(0.1, "Installing model...")
    install_model(model_name, model_path)
    progress(1, "Model installed successfully!")
    return "Model installed successfully!"


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
                llm = LLM("google/gemma-2b-it")
                load_model_button = gr.Button(value="Load Model", size="sm")
                load_model_button.click(llm.load_model, show_progress="minimal")
        gr.Progress(llm.model_loaded)
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
            bot_message = llm.generate(message)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])                
# install models TAB
    with gr.Tab("Install models"):
        with gr.Row() as row:
            with gr.Column(scale=4) as col1:
                model_name_textbox = gr.Textbox(
                    label="Model Name",
                    value="google/gemma-2b-it",
                    placeholder="Model Name",
                    show_label=True,
                    interactive=True,
                )
                model_path_textbox = gr.Textbox(
                    label="Model Path",
                    value=MODELS_PATH,
                    show_label=True,
                    interactive=True,
                )
                text_box = gr.Textbox(
                    label="Progress",
                    value="",
                    interactive=False,
                )
                install_button = gr.Button(value="Install", size="sm")
                install_button.click(gr_install_model, [model_name_textbox, model_path_textbox], text_box )
            with gr.Column(scale=4) as col2:
                models = get_installed_models()
                
                model_dropdown = gr.Dropdown(
                    label="Installed models",
                    value=models[0],
                    choices=models,
                    show_label=True,
                    interactive=True,
                )
                delete_button = gr.Button(value="Delete", size="sm")
                delete_button.click(uninstall_model, [model_dropdown], show_progress="minimal")

demo.launch()

