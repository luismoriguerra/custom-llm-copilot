# Import argparse for command line argument parsing
# Example: python server.py --host localhost --port 8000
import argparse

# Import asynccontextmanager for managing async context
# Used to handle startup/shutdown of async resources
# Example:
#   @asynccontextmanager
#   async def lifespan(app):
#       # startup
#       yield
#       # shutdown
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import Response
import torch
import uvicorn

# Import Milvus Python SDK components
# connections: Used to establish connection to Milvus server
# Example: connections.connect("default", host="localhost", port="19530")
#
# Collection: Main class to interact with Milvus collections (similar to database tables)
# Example: collection = Collection("my_collection")
#         collection.insert(entities)
#         collection.search(query_embeddings)
from pymilvus import (
    connections,
    Collection,
)
# Import SentenceTransformer for generating embeddings from text
# Example: embedder = SentenceTransformer("model_name")
#         embeddings = embedder.encode("text") # Returns tensor([0.1, -0.2, ...])
from sentence_transformers import SentenceTransformer

# Import Hugging Face Transformers components for language model
# AutoModelForCausalLM: Base class for causal language models
# Example: model = AutoModelForCausalLM.from_pretrained("model_name")
#         outputs = model.generate(input_ids) # Returns tensor([[1, 2, ...]])
#
# AutoTokenizer: Converts text to model input tokens
# Example: tokenizer = AutoTokenizer.from_pretrained("model_name") 
#         tokens = tokenizer("text") # Returns {'input_ids': tensor([[1, 2, ...]])}
#
# StoppingCriteria, StoppingCriteriaList: Define when model should stop generating
# Example: stopping_criteria = StoppingCriteriaList([StopOnTokens()])
#         model.generate(stopping_criteria=stopping_criteria)
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


# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

collection_name = "milvus_llm_example"
collection = Collection(collection_name)

# Load tokenizer and models
tokenizer = AutoTokenizer.from_pretrained("Deci/DeciCoder-1b")
tokenizer.add_special_tokens(
    {"additional_special_tokens": stop_tokens},
    replace_additional_special_tokens=False,
)
model = AutoModelForCausalLM.from_pretrained(
    "Deci/DeciCoder-1b", torch_dtype=torch.bfloat16, trust_remote_code=True
)
model = model.to(device)
# Load sentence transformer model for generating embeddings from text
# This model is specifically trained on code-related text
# Example input: "Write a function to sort a list"
# Example output: tensor([-0.1234, 0.5678, ...]) # 768-dimensional vector
embedder = SentenceTransformer(
    "krlvi/sentence-t5-base-nlpl-code_search_net"
)

# Move embedder model to same device (GPU/CPU) as main model
# This ensures consistent processing and optimal performance
embedder = embedder.to(device)


def token_length(text):
    tokens = tokenizer([text], return_tensors="pt")
    return tokens["input_ids"].shape[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager that handles Milvus collection lifecycle.
    
    This ensures the collection is loaded into memory when the server starts and
    properly released when the server shuts down.
    
    Example flow:
        Server startup:
            collection.load() # Loads vectors into memory
            -> Server runs and handles requests
        Server shutdown: 
            collection.release() # Frees memory
    
    Args:
        app: FastAPI application instance
    """
    # Load collection into memory when server starts
    # This improves query performance by keeping vectors in RAM
    collection.load()
    
    # yield control back to FastAPI to run the application
    yield
    
    # When server shuts down, release collection from memory
    # This frees up RAM and ensures clean shutdown
    collection.release()


# Run FastAPI
app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate LLM Response

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    # Make a query
    search_embedding = embedder.encode(prompt)
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    # Search for similar examples in the vector database
    # Returns list of matches sorted by similarity (closest first)
    # Example output: [SearchResult(hits=[Hit(entity=Entity(metadata={'instruction': 'Write a function to sort a list', 'output': 'def sort_list(lst)...'}))])]
    results = collection.search(
        [search_embedding],        # Query vector to search for
        "embeddings",             # Name of vector field to search
        search_params,            # Search parameters like distance metric
        limit=5,                  # Return top 5 closest matches
        output_fields=["metadata"] # Include metadata in results
    )

    # Extract instruction/output pairs from search results
    # Example output: ['Instruction: Write a function to sort a list\nOutput: def sort_list(lst)...\n\n']
    # Extract instruction/output pairs from search results into a list
    # Example results:
    # hits = [Hit(entity=Entity(metadata={'instruction': 'Write a function to sort a list', 
    #                                    'output': 'def sort_list(lst):\n    return sorted(lst)'}))]
    examples = []
    for hits in results:
        for hit in hits:
            # Each hit contains metadata with instruction/output pair
            metadata = hit.entity.metadata
            # Format the example as an instruction/output pair
            # Example: "Instruction: Write a function to sort a list\nOutput: def sort_list(lst):\n    return sorted(lst)\n\n"
            examples.append(
                f"Instruction: {metadata['instruction']}\n"
                f"Output: {metadata['output']}\n\n"
            )

    # Create the system instruction that defines the model's role
    # Example: "You are an expert software engineer who specializes in Python. Write python code..."
    prompt_instruction = (
        "You are an expert software engineer who specializes in Python. "
        "Write python code to fulfill the request from the user.\n\n"
    )

    # Format the user's prompt as an instruction
    # Example: "Instruction: Write a function to calculate fibonacci numbers\nOutput: "
    prompt_user = f"Instruction: {prompt}\nOutput: "

    # Set maximum tokens and calculate initial token count from instruction and user prompt
    max_tokens = 2048
    token_count = token_length(prompt_instruction + prompt_user)

    # Add examples while staying under token limit
    # Example prompt_examples: "Instruction: Write a sort function\nOutput: def sort(lst)...\n\n
    #                          Instruction: Write a binary search\nOutput: def binary_search..."
    prompt_examples = ""
    for example in examples:
        token_count += token_length(example)
        if token_count < max_tokens:
            prompt_examples += example
        else:
            break

    # Combine all parts into final prompt
    # Example full_prompt: "You are an expert...\n\nInstruction: Sort list\nOutput: def sort...\n\n
    #                      Instruction: User prompt\nOutput: "
    full_prompt = f"{prompt_instruction}{prompt_examples}{prompt_user}"

    # Generate response
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    response_tokens = model.generate(
        inputs["input_ids"],
        max_new_tokens=1024,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )
    input_length = inputs["input_ids"].shape[1]
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
