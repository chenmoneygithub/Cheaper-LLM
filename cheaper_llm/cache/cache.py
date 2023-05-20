import redis
from redis.commands.search.field import TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_NAME = "prompt_cache" # Vector Index Name

class PromptCache:
    def __init__(self, cache_file, query_threshold=0.9):
        self.redis_client = redis.Redis(host="localhost", port=6379)
        self.redis_pipe = self.redis_client.pipeline()
        self._create_index(vector_dimensions=384)
        self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.counter = 0
        self.query_threshold = query_threshold
        
    def _create_index(vector_dimensions: int):
        try:
            # check to see if index exists
            r.ft(INDEX_NAME).info()
            print("Index already exists!")
        except:
            # schema
            schema = (
                TagField("tag"),
                VectorField(
                    "vector", # Vector Field Name
                    "FLAT", { # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32", # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions, # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE", # Vector Search Distance Metric
                    }
                ),
            )

            # index Definition
            definition = IndexDefinition(
                prefix=[DOC_PREFIX], 
                index_type=IndexType.HASH,
            )

            # create Index
            r.ft(INDEX_NAME).create_index(
                fields=schema, 
                definition=definition,
            )
                
    def put(prompt, response):
        embeddings = np.array(self.encoder.encode(prompt))
        self.counter += 1
        self.redis_pipe.hset(f"doc:{self.counter}", mapping = {
            "vector": embedding.tobytes(),
            "content": response,
            "tag": "cheaper_llm"
        })
        res = pipe.execute()
        
    def read(prompt):
        query_embedding = np.array(model.encode(query_text))
        query = Query("(@tag:{ cheaper_llm })=>[KNN 2 @vector $vec as score]")
            .sort_by("score")
            .return_fields("id", "score")
            .paging(0, 1)
            .dialect(2)

        # Find all vectors within 0.8 of the query vector
        query_params = {
            "radius": 1 - self.query_threshold,
            "vec": query_embedding.tobytes(),
        }
        cache_response = self.redis_client.ft(INDEX_NAME).search(query, query_params).docs
        if len(cache_response) == 0:
            return None
        return cache_response[0]["content"]

        
        

        
        
            
            
