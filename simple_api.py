import torch
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer

# Basic setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()

# Load model and tokenizer
checkpoint = "Deci/DeciCoder-1b"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, 
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).to(device)

@app.post("/generate")
async def generate(request: Request):
    request_dict = await request.json()
    prompt = request_dict["prompt"]

    # Convert text to tokens
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # Example tokens: [1, 345, 789, 234, ...]

    # Generate new tokens
    response_tokens = model.generate(
        inputs["input_ids"],
        max_new_tokens=1024,  # Maximum new tokens to generate
    )
    # Example response_tokens: [1, 345, 789, 234, 567, 890, ...]
    #                          └─── input tokens ───┘└── new tokens ──┘

    # Decode only the new tokens (skip input tokens)
    response = tokenizer.decode(
        response_tokens[0][inputs["input_ids"].shape[1]:],  # Only take new tokens
        skip_special_tokens=True  # Remove special tokens like <eos>, <pad>
    )

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)