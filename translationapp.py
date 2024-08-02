import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import base64
import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define URLs for the model and tokenizer files
model_files = {
    'config.json': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/model/config.json',
    'generation_config.json': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/model/generation_config.json',
    'model.safetensors': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/model/model.safetensors'
}

tokenizer_files = {
    'source.spm': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/tokenizer/source.spm',
    'special_tokens_map.json': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/tokenizer/special_tokens_map.json',
    'target.spm': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/tokenizer/target.spm',
    'tokenizer_config.json': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/tokenizer/tokenizer_config.json',
    'vocab.json': 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/raw/main/tokenizer/vocab.json'
}

# Function to download a file from a URL
def download_file(url, path):
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)
        logger.info(f"Downloaded {path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        st.error(f"Failed to download file: {e}")

# Function to ensure model and tokenizer files are downloaded
def download_files():
    if not os.path.exists('./model'):
        os.makedirs('./model')
    if not os.path.exists('./tokenizer'):
        os.makedirs('./tokenizer')

    for file_name, url in model_files.items():
        download_file(url, f'./model/{file_name}')

    for file_name, url in tokenizer_files.items():
        download_file(url, f'./tokenizer/{file_name}')

# Function to load the model from local path
@st.cache_resource
def load_model():
    download_files()  # Ensure files are downloaded
    try:
        model = MarianMTModel.from_pretrained('./model')
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error("Failed to load the model. Please check the logs for more details.")
        return None

# Function to load the tokenizer from local path
@st.cache_resource
def load_tokenizer():
    download_files()  # Ensure files are downloaded
    try:
        tokenizer = MarianTokenizer.from_pretrained('./tokenizer')
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        st.error("Failed to load the tokenizer. Please check the logs for more details.")
        return None

# Streamlit App
st.title("MarianMT Model Translation")

# Load Model and Tokenizer
model = load_model()
tokenizer = load_tokenizer()
if model and tokenizer:
    st.success("Model and Tokenizer loaded successfully.")
else:
    st.error("Failed to load Model and Tokenizer.")

# Translation interface
st.subheader("Translate Tshiluba to English")

tshiluba_text = st.text_area("Enter Tshiluba text to translate")
if st.button("Translate"):
    if tshiluba_text and model and tokenizer:
        with st.spinner("Translating..."):
            # Tokenize input
            inputs = tokenizer(tshiluba_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)

            # Generate translation
            translated = model.generate(**inputs)

            # Decode the output
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            st.success(f"Translated text: {translated_text}")

            # Convert translated text to speech
            tts = gTTS(translated_text)
            tts.save("translated_audio.mp3")

            # Display audio player
            audio_file = open("translated_audio.mp3", "rb")
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

            # Optionally provide a download link
            def get_binary_file_downloader_html(bin_file, file_label='File'):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
                return href

            st.markdown(get_binary_file_downloader_html("translated_audio.mp3", 'Download translated audio'), unsafe_allow_html=True)
    else:
        st.warning("Please enter some Tshiluba text to translate.")





