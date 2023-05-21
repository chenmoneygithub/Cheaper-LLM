import numpy as np
import redis
from redis.commands.search.field import TagField
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition
from redis.commands.search.indexDefinition import IndexType
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer

INDEX_NAME = "prompt_cache"  # Vector Index Name
DOC_PREFIX = "doc:"


class PromptCache:
    def __init__(self, cache_file=None, query_threshold=0.9, reset=False):
        self.redis_client = redis.Redis(host="localhost", port=6379)
        self.redis_pipe = self.redis_client.pipeline()
        if reset:
            self.reset()
        self._create_index(vector_dimensions=384)
        self.encoder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.counter = 0
        self.query_threshold = query_threshold

    def _create_index(self, vector_dimensions):
        try:
            # check to see if index exists
            self.redis_client.ft(INDEX_NAME).info()
            print("Index already exists!")
        except:
            # schema
            schema = (
                TagField("tag"),
                VectorField(
                    "vector",  # Vector Field Name
                    "FLAT",
                    {  # Vector Index Type: FLAT or HNSW
                        "TYPE": "FLOAT32",  # FLOAT32 or FLOAT64
                        "DIM": vector_dimensions,  # Number of Vector Dimensions
                        "DISTANCE_METRIC": "COSINE",  # Vector Search Distance Metric
                    },
                ),
            )

            # index Definition
            definition = IndexDefinition(
                prefix=[DOC_PREFIX],
                index_type=IndexType.HASH,
            )

            # create Index
            self.redis_client.ft(INDEX_NAME).create_index(
                fields=schema,
                definition=definition,
            )

    def put(self, prompt, response, model):
        embeddings = np.array(self.encoder.encode(prompt))
        self.counter += 1
        self.redis_pipe.hset(
            f"doc:{self.counter}",
            mapping={
                "vector": embeddings.tobytes(),
                "content": response,
                "model": model,
                "tag": "cheaper",
            },
        )
        self.redis_pipe.execute()

    def get(self, prompt):
        query_embedding = np.array(self.encoder.encode(prompt))
        query = (
            Query("(@tag:{ cheaper })=>[KNN 2 @vector $vec as score]")
            .sort_by("score")
            .return_fields("content", "tag", "score", "model")
            .paging(0, 2)
            .dialect(2)
        )

        query_params = {
            "radius": 1 - self.query_threshold,
            "vec": query_embedding.tobytes(),
        }
        cache_response = (
            self.redis_client.ft(INDEX_NAME).search(query, query_params).docs
        )
        if len(cache_response) == 0:
            return None
        return {
            "content": cache_response[0]["content"],
            "model": cache_response[0]["model"],
            "score": cache_response[0]["score"],
        }

    def reset(self):
        self.redis_client.ft(INDEX_NAME).dropindex(delete_documents=True)
