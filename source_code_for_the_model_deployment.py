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
model_config_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/model/config.json?raw=true'
model_generation_config_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/model/generation_config.json?raw=true'
model_safetensors_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/model/model.safetensors?raw=true'

tokenizer_source_spm_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/tokenizer/source.spm?raw=true'
tokenizer_special_tokens_map_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/tokenizer/special_tokens_map.json?raw=true'
tokenizer_target_spm_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/tokenizer/target.spm?raw=true'
tokenizer_config_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/tokenizer/tokenizer_config.json?raw=true'
tokenizer_vocab_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/blob/main/tokenizer/vocab.json?raw=true'

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

    # Download model files
    download_file(model_config_url, './model/config.json')
    download_file(model_generation_config_url, './model/generation_config.json')
    download_file(model_safetensors_url, './model/model.safetensors')

    # Download tokenizer files
    download_file(tokenizer_source_spm_url, './tokenizer/source.spm')
    download_file(tokenizer_special_tokens_map_url, './tokenizer/special_tokens_map.json')
    download_file(tokenizer_target_spm_url, './tokenizer/target.spm')
    download_file(tokenizer_config_url, './tokenizer/tokenizer_config.json')
    download_file(tokenizer_vocab_url, './tokenizer/vocab.json')

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

