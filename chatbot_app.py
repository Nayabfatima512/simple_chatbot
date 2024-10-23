import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load BlenderBot model and tokenizer
@st.cache_resource
def load_model():
    model_name = "facebook/blenderbot-400M-distill"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Function for interacting with the bot
def chat_with_blenderbot(input_text):
    inputs = tokenizer(input_text, return_tensors="pt")
    reply_ids = model.generate(**inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    return reply

# Streamlit app UI
st.title("BlenderBot Chatbot")
st.write("Chat with the BlenderBot! Type a message and press 'Send'.")

# Chat input and response
user_input = st.text_input("You: ", "")

if st.button("Send"):
    if user_input:
        response = chat_with_blenderbot(user_input)
        st.write(f"Chatbot: {response}")
