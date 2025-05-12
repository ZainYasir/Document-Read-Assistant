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

def generate_response(query, context, model, tokenizer, max_new_tokens=60):
    sentinel = "[END]"                         # custom non-empty stop marker
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # generate, forcing the sentinel at the end
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        # tell the model to append our sentinel when it hits EOS
        forced_bos_token_id=None,
        forced_eos_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(output[0], skip_special_tokens=False)
    # Now append sentinel and re-decode
    if not text.endswith(sentinel):
        text = text + sentinel

    # split on sentinel
    answer = text.split(sentinel)[0]
    # remove the "Answer:" prefix
    return answer.split("Answer:")[-1].strip()

