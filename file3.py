if __name__ == "__main__":
    collection_name = "milvus_llm_example"
    dim = 768
 
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
 
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
 
    schema = CollectionSchema(
        fields, f"{collection_name} is collection of python code prompts"
    )
 
    print(f"Create collection {collection_name}")
    collection = Collection(collection_name, schema)
 
    collection = Collection(collection_name)
    print(collection.num_entities)
 
    python_code_ingestion = PythonCodeIngestion(collection)
    python_code_ingestion.batch_upload()
    print(collection.num_entities)
 
    search_index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("embeddings", search_index)