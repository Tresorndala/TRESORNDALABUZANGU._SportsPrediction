import streamlit as st
from transformers import MarianTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
import base64
import requests
import os

# URLs for model and tokenizer files
model_url = 'https://example.com/path/to/model/model.safetensors'
tokenizer_url = 'https://example.com/path/to/tokenizer/tokenizer.zip'  # Adjust as needed

# Local paths to save the files
model_path = 'model.safetensors'
tokenizer_path = 'tokenizer.zip'

# Download the model file if it doesn't exist
if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, 'wb') as f:
        f.write(response.content)

# Download and extract the tokenizer files if they don't exist
if not os.path.exists(tokenizer_path):
    response = requests.get(tokenizer_url)
    with open(tokenizer_path, 'wb') as f:
        f.write(response.content)

# Assuming tokenizer_path is a zip file, unzip it
import zipfile

with zipfile.ZipFile(tokenizer_path, 'r') as zip_ref:
    zip_ref.extractall('tokenizer')

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
model = load_model(model_path)
tokenizer = load_tokenizer('tokenizer')
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
