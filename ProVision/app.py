import google.generativeai as genai
from transformers import BlipProcessor, BlipForConditionalGeneration
import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_KEY")

# Configure Google Gemini API
genai.configure(api_key=api_key)

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Streamlit UI Title
st.set_page_config(page_title="AI Image Captioning", layout="centered")

# Custom CSS with Gradient Background & White Text for Specific Elements
page_bg_style = """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1E1E2F, #3A3A5A, #6A82FB);
        color: white;
    }

    h1 {
        color: #f8f9fa !important;
        text-align: center;
        font-size: 2.5rem !important;
        font-weight: bold;
    }

    div[data-testid="stFileUploader"] label {
        color: white !important;
        font-size: 1rem;
        font-weight: bold;
    }

    div[data-testid="stSuccess"] {
        background: rgba(46, 204, 113, 0.2);
        padding: 10px;
        border-radius: 8px;
        color: white !important;
        font-weight: bold;
    }

    .generated-caption {
        color: white !important;
        font-size: 1rem;
        font-weight: bold;
    }

    button {
        background-color: #6A82FB !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }
    </style>
"""
st.markdown(page_bg_style, unsafe_allow_html=True)

# UI Components
st.title("📸 ProVisionAI- An AI Powered Image Insights and Captioning with Google Gemini")
st.write("Upload an image and generate **creative** social media captions in seconds!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display image
    st.image(image, caption="Uploaded Image", width=None)

    # Generate caption using BLIP model
    with st.spinner("🔍 Generating Insights..."):
        text = "A photograph of"
        inputs = processor(image, text, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)

    st.success("Insights Generated Successfully!")
    st.write(f"**Insights:** {caption}")
    
    # Use Google Gemini API for creative captions
    with st.spinner("✍️ Generating Social Media Captions..."):
        gemini_model = genai.GenerativeModel("gemini-pro")
        prompt = f"Create 3 catchy social media captions for: {caption}. Do not include hashtags."
        
        response = gemini_model.generate_content(prompt)
    
    # Display generated captions with proper numbering
    captions = response.text.split("\n")
    st.write("✍️ **AI-Generated Captions:**")
    
    numbered_captions = [c.strip() for c in captions if c.strip()]
    for idx, line in enumerate(numbered_captions, start=1):
        if line.strip():
            st.markdown(f'<p class="generated-caption"><b>{idx}. {line}</b></p>', unsafe_allow_html=True)
    
    # Regenerate Button
    if st.button("🔄 Regenerate Captions"):
        st.rerun()