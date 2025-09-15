import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

# Check device availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 🔹 Change this depending on what you want:
# Option 1: Use Hugging Face GPT-2 (default, works immediately)
model_path = "gpt2"

# Option 2: Use your local fine-tuned model
# model_path = "./GPT2/gpt2-medquad-finetuned"

def load_model_and_tokenizer(model_path):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer(model_path)

# Predefined responses
predefined_responses = {
    "hi": "Hello! How can I assist you today?",
    "hello": "Hello there! What can I do for you?",
    "thanks": "You're welcome! Anything else I can help with?",
    "thank you": "You're welcome! Feel free to ask any more questions.",
    "sorry": "No problem at all! How can I assist you further?",
    "help": "Sure! Please ask any medical question you have in mind.",
    "what can you do": "I can assist you with medical questions and health advice. What do you need help with today?"
}

def prepare_input(input_text):
    prompt = f"Question: {input_text} Answer:"
    encoded_input = tokenizer.encode(prompt, return_tensors="pt")
    return encoded_input.to(device)

def generate_text(encoded_input):
    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            encoded_input,
            max_length=256,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def answer_query(input_text):
    normalized_text = input_text.strip().lower()
    if normalized_text in predefined_responses:
        return predefined_responses[normalized_text]
    
    encoded_input = prepare_input(input_text)
    try:
        response = generate_text(encoded_input)
        if "Answer:" in response:
            return response.split("Answer:")[1].strip()
        else:
            return response
    except Exception as e:
        return f"Sorry, I couldn't process your query. ({e})"
