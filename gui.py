from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QMainWindow, QTabWidget, QComboBox, QLabel, QProgressBar, QScrollArea
from utils import install_model, get_available_models, get_installed_models, uninstall_model
import sys
import os
import json
from PyQt5.QtCore import Qt

user_message = "hello my name is frank and i am a software developer,\nmy goal is to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work."
assistant_message = "hello my name is frank and i am a software developer,\nmy goal is to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work. i am learning how to use the transformers library and i am trying to create a chatbot that can help me with my work."

installed_models = get_installed_models()
available_models = get_available_models()

def inprogress(progress_bar):
    progress_bar.setMaximum(0)

def create_user_message(message):
    user_message = QHBoxLayout()
    user_label = QLabel("User :")
    user_message_label = QLabel(message)
    user_message_label.setStyleSheet("background-color: lightgray; margin: 10px;")
    user_message_label.setWordWrap(True)
    user_message.addWidget(user_label)
    user_message.addWidget(user_message_label)
    user_message.setAlignment(Qt.AlignLeft)
    return user_message

def create_assistant_message(message):
    user_message = QHBoxLayout()
    user_label = QLabel("Assistant :")
    user_message_label = QLabel(message)
    user_message_label.setWordWrap(True)
    user_message_label.setStyleSheet("background-color: darkgray; margin: 10px;")
    user_message.addWidget(user_label)
    user_message.addWidget(user_message_label)
    user_message.setAlignment(Qt.AlignRight)
    return user_message

loaded_model = ""

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(200, 200, 700, 700)
    win.setWindowTitle("AI Chatbot Interface")
    
    tab_widget = QTabWidget()

    mistal_tab = QWidget()
    stable_diffusion_tab = QWidget()
    settings_tab = QWidget()

    # Create a QVBoxLayout for the QWidget
    mistal_tab_layout = QVBoxLayout()
    stable_diffusion_tab_layout = QVBoxLayout()

    install_model_layout = QHBoxLayout()
    load_model_layout = QHBoxLayout()
    model_info_layout = QHBoxLayout()
    settings_tab_layout = QVBoxLayout()
    
    progress_bar = QProgressBar()
    progress_bar.setMinimum(0)
    progress_bar.setMaximum(100)

    loaded_model_label = QLabel(f"loaded model: {loaded_model}")
    refresh_button = QPushButton("refresh")
    refresh_button.clicked.connect(lambda: (get_installed_models(), get_available_models()))

    load_model_label = QLabel("installed models: ")
    load_model_button = QPushButton("load model")
    load_model_button.clicked.connect(lambda: get_installed_models())
    unload_model_button = QPushButton("unload model")
    unload_model_button.clicked.connect(lambda: get_installed_models())

    install_model_label = QLabel("available models: ")
    install_model_button = QPushButton("install model")
    install_model_button.clicked.connect(lambda: (progress_bar.setMaximum(0), install_model(available_models_comboxbox.currentText), progress_bar.setMaximum(100)))
    uninstall_model_button = QPushButton("uninstall model")
    uninstall_model_button.clicked.connect(lambda: uninstall_model(installed_models_comboxbox.currentText))

    installed_models_comboxbox = QComboBox()
    installed_models_comboxbox.addItems(installed_models)

    available_models_comboxbox = QComboBox()
    available_models_comboxbox.addItems(available_models)

    settings_tab_layout.addLayout(model_info_layout)
    settings_tab_layout.addWidget(loaded_model_label)
    settings_tab_layout.addLayout(install_model_layout)
    settings_tab_layout.addLayout(load_model_layout)
    settings_tab_layout.addWidget(progress_bar)


    model_info_layout.addWidget(loaded_model_label)
    model_info_layout.addWidget(refresh_button)

    install_model_layout.addWidget(install_model_label)
    install_model_layout.addWidget(installed_models_comboxbox)
    install_model_layout.addWidget(install_model_button)
    install_model_layout.addWidget(uninstall_model_button)

    load_model_layout.addWidget(load_model_label)
    load_model_layout.addWidget(available_models_comboxbox)
    load_model_layout.addWidget(load_model_button)
    load_model_layout.addWidget(unload_model_button)

    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    # Create a QWidget for the QScrollArea
    scroll_widget = QWidget()
    scroll_area.setWidget(scroll_widget)

    # Create a QVBoxLayout for the scroll_widget
    scroll_layout = QVBoxLayout()
    scroll_widget.setLayout(scroll_layout)

    actions_layout = QHBoxLayout()
    send_button = QPushButton("Send")
    send_button.clicked.connect(lambda: (scroll_layout.addLayout(create_user_message(user_message)), scroll_layout.addLayout(create_assistant_message(assistant_message))))
    clear_button = QPushButton("Clear")
    actions_layout.addWidget(send_button)
    actions_layout.addWidget(clear_button)

    chatbot_label = QLabel("Chatbot")
    chatbot_label.setAlignment(Qt.AlignCenter)
    mistal_tab_layout.addWidget(chatbot_label)
    mistal_tab_layout.addWidget(scroll_area)
    mistal_tab_layout.addLayout(actions_layout)
    

    # Set the QWidget's layout to the QVBoxLayout
    mistal_tab.setLayout(mistal_tab_layout)
    stable_diffusion_tab.setLayout(stable_diffusion_tab_layout)
    settings_tab.setLayout(settings_tab_layout)

    # Add the QWidget to the QTabWidget
    tab_widget.addTab(mistal_tab, "Local LLM")
    tab_widget.addTab(stable_diffusion_tab, "Stable Diffusion")
    tab_widget.addTab(settings_tab, "Settings")

    win.setCentralWidget(tab_widget)

    win.show()
    sys.exit(app.exec_())

window()