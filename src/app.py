import re
import torch
import os
import streamlit as st

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from peft import PeftModel


# =========================================
# Page Config
# =========================================

st.set_page_config(
    page_title="Arabic Text Summarizer",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Arabic Text Summarization")
st.markdown("Summarize Arabic text using Fine-Tuned mT5 + LoRA")


# =========================================
# Load Model
# =========================================

@st.cache_resource
def load_model():

    MODEL_PATH = os.path.join(os.getcwd(), "models")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "google/mt5-small"
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "google/mt5-small",
    #     use_fast=False
    # )

    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/mt5-small"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# =========================================
# Device
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)


# =========================================
# Generate Summary Function
# =========================================

def generate_summary(text):

    text = "summarize Arabic: " + text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,

            max_new_tokens=64,

            min_length=20,

            num_beams=8,

            no_repeat_ngram_size=3,

            repetition_penalty=2.5,

            length_penalty=1.0,

            early_stopping=True
        )

    summary = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    # Remove extra tokens
    summary = re.sub(
        r"<extra_id_\d+>",
        "",
        summary
    )

    summary = summary.strip()

    return summary


# =========================================
# Text Input
# =========================================

user_input = st.text_area(
    "Enter Arabic Text",
    height=250,
    placeholder="اكتب النص العربي هنا..."
)


# =========================================
# Generate Button
# =========================================

if st.button("Generate Summary"):

    if not user_input.strip():

        st.warning("Please enter Arabic text.")

    else:

        with st.spinner("Generating summary..."):

            summary = generate_summary(user_input)

        st.subheader("📄 Summary")

        st.success(summary)