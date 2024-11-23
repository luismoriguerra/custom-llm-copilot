# Import Milvus database related components
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# Import ML/NLP related libraries
from transformers import AutoTokenizer  # For tokenizing text
from datasets import load_dataset  # For loading the python code dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into chunks
from sentence_transformers import SentenceTransformer  # For generating embeddings

# Import utility libraries
from tqdm.auto import tqdm  # For progress bars
from uuid import uuid4  # For generating unique IDs

# Establish connection to Milvus database
connections.connect("default", host="localhost", port="19530")


class PythonCodeIngestion:
    def __init__(self, collection, batch_limit=100):
        self.collection = collection
        self.python_code = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        self.embedder = SentenceTransformer("krlvi/sentence-t5-base-nlpl-code_search_net")
        self.tokenizer = AutoTokenizer.from_pretrained("Deci/DeciCoder-1b")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=lambda text: len(self.tokenizer.encode(text)),
            separators=["\n\n", "\n", " ", ""]
        )
        self.batch_limit = batch_limit

    def process_page(self, page):
        metadata = {
            "instruction": page["instruction"],
            "input": page["input"],
            "output": page["output"]
        }
        prompts = self.text_splitter.split_text(page["prompt"])
        return prompts, [{"chunk": i, "prompt": p, **metadata} for i, p in enumerate(prompts)]

    def batch_upload(self):
        batch_texts, batch_metadatas = [], []
        
        for page in tqdm(self.python_code):
            texts, metadatas = self.process_page(page)
            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(batch_texts) >= self.batch_limit:
                self.collection.insert([
                    [str(uuid4()) for _ in range(len(batch_texts))],
                    self.embedder.encode(batch_texts),
                    batch_metadatas
                ])
                batch_texts, batch_metadatas = [], []

        if batch_texts:
            self.collection.insert([
                [str(uuid4()) for _ in range(len(batch_texts))],
                self.embedder.encode(batch_texts),
                batch_metadatas
            ])
        self.collection.flush()


if __name__ == "__main__":
    # Define collection parameters
    collection_name = "milvus_llm_example"
    dim = 768  # Embedding dimension

    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define collection schema
    fields = [
        FieldSchema(
            name="ids",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=36,
        ),
        FieldSchema(
            name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim
        ),
        FieldSchema(name="metadata", dtype=DataType.JSON),
    ]

    # Create schema with description
    schema = CollectionSchema(
        fields, f"{collection_name} is collection of python code prompts"
    )

    # Create and initialize collection
    print(f"Create collection {collection_name}")
    collection = Collection(collection_name, schema)

    # Get collection reference and print initial count
    collection = Collection(collection_name)
    print(collection.num_entities)

    # Initialize ingestion class and upload data
    python_code_ingestion = PythonCodeIngestion(collection)
    python_code_ingestion.batch_upload()
    print(collection.num_entities)

    # Define search index parameters
    search_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    # Create search index and load into memory
    collection.create_index("embeddings", search_index)
    collection.load()

    # Simplified search example
    query = "Construct a neural network model in Python to classify the MNIST data set correctly."
    results = collection.search(
        [python_code_ingestion.embedder.encode(query)],
        "embeddings",
        {"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["metadata"]
    )
    
    for hit in results[0]:
        print(f"Distance: {hit.distance}")
        print(f"Instruction: {hit.entity.metadata['instruction']}\n")
