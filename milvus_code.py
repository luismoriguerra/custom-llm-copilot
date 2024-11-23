from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

# For tokenizing text and measuring token lengths
from transformers import AutoTokenizer
# For loading Python code instruction dataset
from datasets import load_dataset
# For splitting code into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter
# For generating embeddings from text
from sentence_transformers import SentenceTransformer

# For progress bars during processing
from tqdm.auto import tqdm
# For generating unique IDs
from uuid import uuid4


# Connect to Milvus
connections.connect("default", host="localhost", port="19530")


class PythonCodeIngestion:
    def __init__(
        self,
        collection,
        python_code=None,
        embedder=None,
        tokenizer=None,
        text_splitter=None,
        batch_limit=100,
    ):
        # Store reference to Milvus collection for inserting vectors
        self.collection = collection

        # Load Python code dataset if not provided
        # Example output: Dataset({
        #    features: ['instruction', 'input', 'output', 'prompt'],
        #    num_rows: 18000
        # })
        self.python_code = python_code or load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca",
            split="train",
        )

        # Initialize sentence embedder model for encoding text into vectors
        # Example output: <SentenceTransformer object at 0x...>
        # Generates 768-dimensional embeddings
        self.embedder = embedder or SentenceTransformer(
            "krlvi/sentence-t5-base-nlpl-code_search_net"
        )

        # Initialize tokenizer for measuring text length in tokens
        # Example output: PreTrainedTokenizer object
        # Used by text splitter to chunk by token count
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            "Deci/DeciCoder-1b"
        )

        # Configure text splitter for breaking code into chunks
        # Example input: "def foo():\n    x = 1\n    y = 2\n    return x + y"
        # Example output: ["def foo():", "    x = 1\n    y = 2", "    return x + y"]
        self.text_splitter = (
            text_splitter
            or RecursiveCharacterTextSplitter(
                chunk_size=400,  # Target size of each chunk in tokens
                chunk_overlap=20,  # Number of overlapping tokens between chunks
                length_function=self.token_length,  # Function to count tokens
                separators=["\n\n", "\n", " ", ""],  # Where to split text
            )
        )

        # Maximum number of chunks to process in one batch
        self.batch_limit = batch_limit

    def token_length(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_metadata(self, page):
        return {
            "instruction": page["instruction"],
            "input": page["input"],
            "output": page["output"],
        }

    def split_texts_and_metadatas(self, page):
        """Split a page's prompt text into chunks and create metadata for each chunk.
        
        Args:
            page: Dictionary containing 'prompt', 'instruction', 'input', and 'output' fields
            
        Returns:
            tuple: (prompts, metadatas) where:
                prompts: List of text chunks from splitting the prompt
                metadatas: List of metadata dicts for each chunk
                
        Example:
            Input page = {
                'prompt': 'def add(a,b):\n    return a + b\n\ndef subtract(a,b):', 
                'instruction': 'Write math functions',
                'input': '',
                'output': 'Here are some math functions'
            }
            
            Returns:
            prompts = ['def add(a,b):', '    return a + b', 'def subtract(a,b):']
            metadatas = [
                {'chunk': 0, 'prompt': 'def add(a,b):', 'instruction': '...', ...},
                {'chunk': 1, 'prompt': '    return a + b', 'instruction': '...', ...},
                {'chunk': 2, 'prompt': 'def subtract(a,b):', 'instruction': '...', ...}
            ]
        """
        # Get base metadata that applies to all chunks
        basic_metadata = self.get_metadata(page)
        
        # Split the prompt text into chunks using the text splitter
        prompts = self.text_splitter.split_text(page["prompt"])
        
        # Create metadata dict for each chunk, including chunk index and prompt text
        metadatas = [
            {"chunk": j, "prompt": prompt, **basic_metadata}
            for j, prompt in enumerate(prompts)
        ]
        # Example output:
        # [
        #     {
        #         "chunk": 0,
        #         "prompt": "def add(a,b):", 
        #         "instruction": "Write math functions",
        #         "input": "",
        #         "output": "Here are some math functions"
        #     },
        #     {
        #         "chunk": 1,
        #         "prompt": "    return a + b",
        #         "instruction": "Write math functions", 
        #         "input": "",
        #         "output": "Here are some math functions"
        #     }
        # ]
        
        return prompts, metadatas

    def upload_batch(self, texts, metadatas):
        """Upload a batch of texts and metadata to Milvus collection.
        
        Args:
            texts: List of text strings to embed and upload
                Example: ["def add(a,b):", "    return a + b"]
            metadatas: List of metadata dicts for each text
                Example: [
                    {"chunk": 0, "prompt": "def add(a,b):", "instruction": "..."},
                    {"chunk": 1, "prompt": "    return a + b", "instruction": "..."}
                ]
        """
        # Generate unique IDs for each text
        # Example: ["123e4567-e89b-12d3-a456-426614174000", ...]
        # Generate a list of unique UUIDs, one for each text in the input
        # uuid4() creates a random UUID like "123e4567-e89b-12d3-a456-426614174000"
        # We convert each UUID to string format for Milvus compatibility
        # The list comprehension creates as many UUIDs as there are texts
        ids = [str(uuid4()) for _ in range(len(texts))]
        
        # Generate embeddings for each text using the embedder model
        # Example shape: (num_texts, embedding_dim)
        # Example: [[0.1, 0.2, ...], [0.3, 0.4, ...]]
        embeddings = self.embedder.encode(texts)
        
        # Insert the IDs, embeddings and metadata into Milvus collection
        # Format: [list of IDs, list of embeddings, list of metadata dicts]
        self.collection.insert([ids, embeddings, metadatas])

    def batch_upload(self):
        """Process and upload the Python code dataset in batches.
        
        Iterates through dataset pages, splits texts into chunks with metadata,
        and uploads in batches to avoid memory issues.
        """
        # Initialize empty lists to accumulate batch data
        batch_texts = []  # Will contain code chunks
        batch_metadatas = []  # Will contain metadata for each chunk
        
        # Iterate through dataset with progress bar
        # Example page: {
        #   'instruction': 'Write math functions',
        #   'input': '',
        #   'output': 'Here are some math functions...'
        # }
        for page in tqdm(self.python_code):
            # Split page into chunks with metadata
            # Example texts: ["def add(a,b):", "    return a + b"]
            # Example metadatas: [{"chunk": 0, "prompt": "def add..."}, ...]
            texts, metadatas = self.split_texts_and_metadatas(page)

            # Add to current batch
            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            # When batch reaches limit, upload and reset
            if len(batch_texts) >= self.batch_limit:
                # Example batch_limit: 100 chunks per batch
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        # Upload any remaining chunks in final partial batch
        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)

        # Ensure all data is written to disk
        self.collection.flush()


if __name__ == "__main__":
    collection_name = "milvus_llm_example"
    dim = 768

    # Create collection if it doesn't exist
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    # Define the schema fields for the Milvus collection
    fields = [
        # Primary key field for unique document IDs
        # Example ID: "doc_123e4567-e89b-12d3-a456-426614174000" 
        FieldSchema(
            name="ids",
            dtype=DataType.VARCHAR,  # String data type
            is_primary=True,         # Set as primary key
            auto_id=False,           # IDs provided manually
            max_length=36,           # Max UUID string length
        ),
        
        # Vector field to store embeddings
        # Example embedding: [0.123, -0.456, 0.789, ..., 0.321] 
        FieldSchema(
            name="embeddings", 
            dtype=DataType.FLOAT_VECTOR,  # Vector of floats
            dim=dim                       # Vector dimension (768)
        ),
        
        # JSON field for storing metadata about each document
        # Example metadata: {
        #   "instruction": "Write a sorting function",
        #   "chunk": 0,
        #   "source": "python_dataset"
        # }
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON  # JSON document
        ),
    ]

    schema = CollectionSchema(
        fields, f"{collection_name} is collection of python code prompts"
    )

    print(f"Create collection {collection_name}")
    collection = Collection(collection_name, schema)

    # Connect to collection and show size
    collection = Collection(collection_name)
    print(collection.num_entities)

    # Ingest data and show stats now that data is ingested
    python_code_ingestion = PythonCodeIngestion(collection)
    python_code_ingestion.batch_upload()
    print(collection.num_entities)

    # Build search index
    # Define the search index configuration
    search_index = {
        # IVF_FLAT: Inverted file with flat quantization
        # Divides vectors into clusters for faster search
        "index_type": "IVF_FLAT",  
        
        # L2 distance metric for comparing vectors
        # Smaller L2 distance = more similar vectors
        # Example: L2([1,1], [2,2]) = sqrt((1-2)^2 + (1-2)^2) = 1.414
        "metric_type": "L2",
        
        # Index parameters
        "params": {
            "nlist": 128,  # Number of clusters to divide vectors into
                          # More clusters = faster search but less accurate
                          # Fewer clusters = slower but more accurate
        },
    }

    # Create the index on the embeddings field
    # Example index structure:
    # Cluster 1: [doc1_vector, doc4_vector, ...] 
    # Cluster 2: [doc2_vector, doc5_vector, ...]
    # ...
    # Cluster 128: [doc3_vector, doc6_vector, ...]
    collection.create_index("embeddings", search_index)

    # Before conducting a search, you need to load the data into memory.
    collection.load()

    # Make a query
    query = (
        "Construct a neural network model in Python to classify "
        "the MNIST data set correctly."
    )
    search_embedding = python_code_ingestion.embedder.encode(query)
    # Configure search parameters for querying the vector index
    search_params = {
        # L2 distance metric for comparing vectors
        # Example: L2([1,1], [2,2]) = sqrt((1-2)^2 + (1-2)^2) = 1.414
        # Smaller distance = more similar vectors
        "metric_type": "L2",

        # Additional search parameters
        "params": {
            # Number of clusters to search during query
            # Higher nprobe = more clusters searched = better recall but slower
            # Lower nprobe = fewer clusters = faster but may miss results
            # Example with nprobe=10:
            # If query vector is closest to cluster 5, will search:
            # - Cluster 5 (closest)
            # - 9 next closest clusters
            "nprobe": 10,
        },
    }
    # Search for similar vectors in the collection
    # Parameters:
    # - [search_embedding]: List of query vectors to search for
    # - "embeddings": Name of the vector field to search in
    # - search_params: Search configuration (metric, nprobe etc)
    # - limit=3: Return top 3 most similar results
    # - output_fields: Additional fields to return with results
    results = collection.search(
        [search_embedding],
        "embeddings", 
        search_params,
        limit=3,
        output_fields=["metadata"],
    )

    # Process search results
    # Results are grouped by query vector, so iterate through hits for each query
    # Example output:
    # Distance: 0.7067 (smaller = more similar)
    # Instruction: "Create a neural network in Python to identify hand-written digits..."
    for hits in results:  # For each query vector's results
        for hit in hits:  # For each match found
            # Print L2 distance score (smaller = more similar)
            print(f"Distance: {hit.distance:.4f}")
            # Print the instruction text stored in metadata
            print(f"Instruction: {hit.entity.metadata['instruction']}")
    # 0.7066953182220459
    # Create a neural network in Python to identify
    # hand-written digits from the MNIST dataset.
    # 0.7366453409194946
    # Create a question-answering system using Python
    # and Natural Language Processing.
    # 0.7389795184135437
    # Write a Python program to create a neural network model that can
    # classify handwritten digits (0-9) with at least 95% accuracy.
