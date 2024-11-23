import argparse

from fastapi import FastAPI, Request
from fastapi.responses import Response
import torch
import uvicorn

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


# Torch settings
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define stopping behavior
stop_tokens = ["def", "class", "Instruction", "Output"]
stop_token_ids = [589, 823, 9597, 2301]


class StopOnTokens(StoppingCriteria):
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        stop_ids = stop_token_ids
        
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True

        return False


# Load the tokenizer from Hugging Face hub
# Example tokenizer output: {'input_ids': tensor([[1542, 6789, ...]])}
tokenizer = AutoTokenizer.from_pretrained("Deci/DeciCoder-1b")

# Add custom stop tokens to the tokenizer vocabulary
# stop_tokens = ["def", "class", "Instruction", "Output"] 
# This allows the model to recognize when to stop generating
tokenizer.add_special_tokens(
    {"additional_special_tokens": stop_tokens},
    replace_additional_special_tokens=False,
)

# Load the pre-trained language model
# Using bfloat16 for reduced memory usage while maintaining precision
# trust_remote_code=True allows running model-specific code
# Example model output shape: torch.Size([1, sequence_length, vocab_size])
model = AutoModelForCausalLM.from_pretrained(
    "Deci/DeciCoder-1b", torch_dtype=torch.bfloat16, trust_remote_code=True
)

# Move model to GPU if available, otherwise CPU
# device = 'cuda' or 'cpu'
model = model.to(device)


# Run FastAPI
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate LLM Response

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    # RAG will go here
    # ...

    # Tokenize the input prompt and move to GPU/CPU device
    # Example prompt: "Write a function to sort a list"
    # Example inputs: tensor([[1542, 6789, 4321, ...]])
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate response tokens from the model
    # max_new_tokens=1024 limits output length
    # stopping_criteria defines custom stop conditions
    # Example response_tokens: tensor([[1542, 6789, 4321, 8765, 2468, ...]])
    response_tokens = model.generate(
        inputs["input_ids"],
        max_new_tokens=1024,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )

    # Get length of input tokens to extract only the generated response
    # Example input_length: 4 (if prompt was 4 tokens long)
    input_length = inputs["input_ids"].shape[1]

    # Decode response tokens back to text, skipping special tokens
    # Only take tokens after the input length to get just the response
    # Example output: "def sort_list(lst):\n    return sorted(lst)"
    response = tokenizer.decode(
        response_tokens[0][input_length:], skip_special_tokens=True
    )

    return response


if __name__ == "__main__":
    # Start Service - Defaults to localhost on port 8000
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug")
