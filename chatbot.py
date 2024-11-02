import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_id = "chuanli11/Llama-3.2-3B-Instruct-uncensored"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Streamlit app title
st.title("Text Generation with Llama-3.2")

# User input for the message
user_input = st.text_area("Enter your message:", placeholder="Type here...")

# Max new tokens
max_new_tokens = st.slider("Max new tokens:", min_value=10, max_value=4096, value=100)

# Generate text on button click
if st.button("Generate Text"):
    if user_input:
        # Prepare input for the model
        inputs = tokenizer(user_input, return_tensors="pt").to(device)
        
        # Generate text
        outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
        
        # Decode and display the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.text_area("Generated Text:", value=generated_text, height=300)
    else:
        st.warning("Please enter a message to generate text.")
