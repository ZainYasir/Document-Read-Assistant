from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_phi2():
    model_id = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.eval()
    return model, tokenizer

def generate_response(query, context, model, tokenizer, max_new_tokens=100):
    stop_token = ""  
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
        + stop_token
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    return text.split(stop_token)[0].split("Answer:")[-1].strip()
