import os
import zipfile
import gdown
import shutil
import base64
import streamlit as st
from gtts import gTTS
from transformers import MarianMTModel, MarianTokenizer

# Define the URLs to the GitHub repositories
model_repo_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/tree/main/model'
tokenizer_repo_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/tree/main/tokenizer'

# Local directories to save the downloaded files
model_dir = 'model_folder'
tokenizer_dir = 'tokenizer_folder'

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(tokenizer_dir, exist_ok=True)

# Download the files from GitHub
!git clone https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction.git

# Move model and tokenizer files to appropriate directories
shutil.move('TRESORNDALABUZANGU._SportsPrediction/model', model_dir)
shutil.move('TRESORNDALABUZANGU._SportsPrediction/tokenizer', tokenizer_dir)

# Load the model and tokenizer
@st.cache_resource
def load_model(model_path):
    model = MarianMTModel.from_pretrained(model_path)
    return model

@st.cache_resource
def load_tokenizer(tokenizer_path):
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# Streamlit App
st.title("MarianMT Model Translation")

# Load Model and Tokenizer
model = load_model(model_dir)
tokenizer = load_tokenizer(tokenizer_dir)
st.success("Model and Tokenizer loaded successfully from GitHub.")

# Translation interface
st.subheader("Translate Tshiluba to English")

tshiluba_text = st.text_area("Enter Tshiluba text to translate")
if st.button("Translate"):
    if tshiluba_text:
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








