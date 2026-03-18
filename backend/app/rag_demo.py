"""分片"""
from typing import List

def split_text(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as f:
        content = f.read()
    return [chunk for chunk in content.split('##') if chunk.strip()]

chunks = split_text("./data/demo.md")

# for i, chunk in enumerate(chunks):
#     print(f"[{i}], {chunk}\n")


"""向量化"""
from os import name
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
def embedd_chunk(chunk: str) -> List[float]:
    return embedding_model.encode(chunk).tolist()

embeddings = [embedd_chunk(chunk) for chunk in chunks]

"""添加到向量数据库"""
import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection(name="default")
def add_chunks_to_collection(chunks: List[str], embeddings: List[List[float]]) -> None:
    ids = [f"id_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids= ids
    )
add_chunks_to_collection(chunks, embeddings)

"""召回"""
def retrieve(query: str, top_k: int) -> List[str]:
    query_embedding = embedd_chunk(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results["documents"][0]

"""重排"""
from sentence_transformers import CrossEncoder

def rerank(query: str, chunks: List[str], top_k: init) -> List[str]:
    cross_encoder = CrossEncoder("shibing624/text2vec-base-chinese")
    scores = cross_encoder.predict([(query, chunk) for chunk in chunks])
    sorted_chunks = [chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)]
    return sorted_chunks[:top_k]


