import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS
import base64
import os
import requests
import shutil

# URLs to the model and tokenizer folders on GitHub
model_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/tree/main/model'
tokenizer_url = 'https://github.com/Tresorndala/TRESORNDALABUZANGU._SportsPrediction/tree/main/tokenizer'

# Function to download and extract the model
@st.cache_resource
def load_model(model_url):
    if not os.path.exists("./model"):
        os.system(f"git clone {model_url} ./model")
    model = MarianMTModel.from_pretrained("./model")
    return model

# Function to download and extract the tokenizer
@st.cache_resource
def load_tokenizer(tokenizer_url):
    if not os.path.exists("./tokenizer"):
        os.system(f"git clone {tokenizer_url} ./tokenizer")
    tokenizer = MarianTokenizer.from_pretrained("./tokenizer")
    return tokenizer

# Streamlit App
st.title("MarianMT Model Translation")

# Load Model and Tokenizer
model = load_model(model_url)
tokenizer = load_tokenizer(tokenizer_url)
if model and tokenizer:
    st.success("Model and Tokenizer loaded successfully from GitHub.")
else:
    st.error("Failed to load Model and Tokenizer.")

# Translation interface
st.subheader("Translate Tshiluba to English")

tshiluba_text = st.text_area("Enter Tshiluba text to translate")
if st.button("Translate"):
    if tshiluba_text:
        with st.spinner("Translating..."):
            # Tokenize input
            inputs = tokenizer(tshiluba_text, return_tensors="pt", trunc




