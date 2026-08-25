# tests/faiss_check.py

from ai_service.vectorDB.faiss_util import load_index, embed_query, search # type: ignore

index, id_map = load_index(
    "knowledge-base/FAISS"
)

query = "What is Article 17?"

query_vector = embed_query(query)

results = search(
    index,
    id_map,
    query_vector,
    top_k=5
)

for r in results:
    print("=" * 80)
    print(r["chunk_id"])
    print(r["score"])
    print(r["text"][:300])