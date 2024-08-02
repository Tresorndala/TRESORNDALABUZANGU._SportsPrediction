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
    'special_tokens_map.json': 'https://github.com/Tres




