# This is the file for the structuring step of the framework

"""
Example Input:
"William Gerald Standridge (November 27, 1953 - April 12, 2014) was an American stock car racing driver. He was a competitor in the NASCAR Winston Cup Series and Busch Series."

Output:
(S> William gerald standridge| P> Nationality| O> American),
(S> William gerald standridge| P> Occupation| O> Stock car racing driver),
(S> William gerald standridge| P> Competitor| O> Busch series),
(S> William gerald standridge| P> Competitor| O> Nascar winston cup series),
(S> William gerald standridge| P> Birth date| O> November 27, 1953),
(S> William gerald standridge| P> Death date| O> April 12, 2014)
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

def load_t2t_model(model_path: str = "pat-jj/text2triple-flan-t5"):
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16  # Use bfloat16 for efficiency
    )
    return tokenizer, model


# format triples to the format of [(S, P, O), (S, P, O), ...]
def format_triples(triples):
    return [(triple.split("|")[0].split("> ")[1], triple.split("|")[1].split("> ")[1], triple.split("|")[2].split("> ")[1]) for triple in triples]


def structure(text, tokenizer, model):
    # Tokenize input with proper padding and attention mask
    inputs = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as model
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=8,  # Use beam search
            early_stopping=True,
            length_penalty=0.6,  # Penalize very long outputs
            use_cache=True  # Use KV cache for faster generation
        )
    
    # Decode and return the generated triples
    generated_triples = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Format the triples
    generated_triples = format_triples(generated_triples)
    
    return generated_triples
    