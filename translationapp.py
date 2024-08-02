import streamlit as st
import requests
import zipfile
import io
import os
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import base64

# URLs for model and tokenizer ZIP files
model_zip_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/archive/refs/heads/31e51f7fcfc89a16732c914cdec789b235126557.zip'
tokenizer_zip_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/archive/refs/heads/31e51f7fcfc89a16732c914cdec789b235126557.zip'

# Local paths to save the files
model_zip_path = 'model.zip'
model_extract_path = 'model'
tokenizer_zip_path = 'tokenizer.zip'
tokenizer_extract_path = 'tokenizer'

# Function to download and extract ZIP files
def download_and_extract(zip_url, extract_to):
    if not os.path.exists(extract_to):
        response = requests.get(zip_url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_to)

# Download and extract model
download_and_extract(model_zip_url, model_extract_path)

# Download and extract tokenizer
download_and_extract(tokenizer_zip_url, tokenizer_extract_path)

# Load the model and tokenizer
@st.cache_resource
def load_model(model_path):
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_tokenizer(tokenizer_path):
    try:
        tokenizer = MarianTokenizer.from_pretrained(tokenizer_path)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None

# Streamlit App
st.title("MarianMT Model Translation")

# Load Model and Tokenizer
model = load_model(model_extract_path)
tokenizer = load_tokenizer(tokenizer_extract_path)
if model and tokenizer:
    st.success("Model and Tokenizer loaded successfully.")

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
else:
    st.error("Failed to load model or tokenizer.")

