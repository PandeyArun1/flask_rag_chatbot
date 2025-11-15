import chromadb
from sentence_transformers import SentenceTransformer
model=SentenceTransformer('all-MiniLM-L6-v2')

chroma_client=chromadb.Client()
collection=chroma_client.get_or_create_collection(name="documents")


def store_in_chunks(chunks):
    try:
        existing=collection.get()
        existing_texts=set(existing['documents']) if existing and 'documents' in existing else set()
    except Exception as e:
        print("Warning: Could not fetch existing documents:", e)
        existing_texts = set()

    ids=[]
    texts=[]
    embeddings=[]
    metadatas=[]

    chunk_id=1
    for chunk in chunks:
        text=chunk['chunks']
        filename=chunk['filename']

        if text in existing_texts:
            print(f"skipping duplicate chunk from {filename}")
            continue
    
        embedding=model.encode(text).tolist()
        ids.append(f"chunk_{chunk_id}")
        texts.append(text)
        embeddings.append(embedding)
        metadatas.append({"filename":filename})
        chunk_id+=1

    if texts:
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Stored {len(texts)} new chunks in ChromaDB.")
    else:
        print("No new chunks to add (all were duplicates).")




def retrieve_relevant_chunks(query, top_k=3, strong_match_threshold=0.40):
    print(f"\n💬 [INFO] Received query: {query}")

    if not query or not query.strip():
        return []

    try:
        query_embedding = model.encode(query).tolist()
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return []

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k * 5,
        include=["documents", "metadatas", "distances"]
    )

    if not results or not results["documents"][0]:
        return []

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    combined = list(zip(docs, metas, distances))
    combined_sorted = sorted(combined, key=lambda x: x[2])

    best_doc, best_meta, best_distance = combined_sorted[0]
    best_filename = best_meta.get("filename", "Unknown")

    print("📏 Distances returned:", distances)

    # ⭐ RULE 1: If strong match → return ONLY ONE
    if best_distance < strong_match_threshold:
        print("🎯 Strong match — Returning ONLY best chunk")
        return [{
            "filename": best_filename,
            "distance": best_distance,
            "content": best_doc
        }]

    # ⭐ RULE 2: Filter out chunks with unrelated filenames
    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in combined_sorted
        if meta.get("filename") == best_filename
        or dist < best_distance + 0.40    # Similar content window
    ]

    # If filtered becomes empty, fall back to top-k
    if not filtered:
        filtered = combined_sorted[:top_k]

    # ⭐ RULE 3: Return top-k from cleaned list
    final = []
    for doc, meta, dist in filtered[:top_k]:
        final.append({
            "filename": meta.get("filename", "Unknown"),
            "distance": dist,
            "content": doc
        })

    return final





