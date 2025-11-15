def splits_into_chunks(text,chunk_size=500, overlap=50):
    print(f"\n [INFO] Splitting text into chunks of size {chunk_size} with overlap {overlap}...")
    chunks=[]
    start =0 
    while start < len(text):
        end=start + chunk_size
        chunk=text[start:end]
        chunks.append(chunk.strip())
        start+=chunk_size -overlap
    return chunks





