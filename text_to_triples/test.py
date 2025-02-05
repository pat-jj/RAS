import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "pat-jj/text2graph-llama-3.2-3b"

def load_model_and_tokenizer():
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set up chat template
    tokenizer.chat_template = tokenizer.chat_template or "llama-3.1"
    
    return model, tokenizer

def generate_triples(model, tokenizer, input_text, max_length=2048):
    # Format the input using chat template
    messages = [{
        "role": "user",
        "content": f"Convert the following text to triples:\n\nText: {input_text}"
    }]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    
    print("\nModel loaded! Enter text to convert to triples (type 'quit' to exit):")
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'quit':
            break
            
        print("\nGenerating triples...")
        response = generate_triples(model, tokenizer, user_input)
        print("\nResponse:", response)

if __name__ == "__main__":
    main()