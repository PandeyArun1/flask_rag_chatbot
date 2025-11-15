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




def retrieve_relevant_chunks(query, top_k=3):
    print(f"\n💬 [INFO] Received query: {query}")

    if not query or not query.strip():
        print("⚠️ [WARN] Empty query received. Returning empty list.")
        
        return []
    

    # Step 1 — Encode query
    try:
        query_embedding = model.encode(query).tolist()
        print("🧠 [DEBUG] Query embedding created successfully.")
    except Exception as e:
        print(f"❌ [ERROR] Failed to create query embedding: {e}")
        return []

    # Step 2 — Check if any data exists
    count = collection.count()
    print(f"📦 [DEBUG] Total chunks currently in ChromaDB: {count}")

    if count == 0:
        print("⚠️ [WARN] No data found in ChromaDB. Did you call /process_files first?")
        return []

    # Step 3 — Perform similarity search
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 5,  # get more & then filter
            include=['documents', 'metadatas', 'distances']
        )
        print("🔍 [DEBUG] Query executed successfully.")
    except Exception as e:
        print(f"❌ [ERROR] Chroma query failed: {e}")
        return []

    # Step 4 — Process results
    retrieved_chunks = []

    if not results or not results.get('documents') or len(results['documents'][0]) == 0:
        print("⚠️ [WARN] No results returned from ChromaDB.")
        return []

    docs = results['documents'][0]
    metas = results['metadatas'][0]
    distances = results['distances'][0]

    print("📏 Distances returned:", distances)

    # Combine results
    combined = list(zip(docs, metas, distances))

    # Sort by similarity (lower distance = more similar)
    combined_sorted = sorted(combined, key=lambda x: x[2])

    # Take top_k best matches
    best_chunks = combined_sorted[:top_k]

    for doc, meta, dist in best_chunks:
        retrieved_chunks.append({
            "filename": meta.get("filename", "Unknown"),
            "distance": dist,
            "content": doc
        })

    print(f"✅ [INFO] Final chunks returned: {len(retrieved_chunks)}")

    return retrieved_chunks
