from sentence_transformers import SentenceTransformer

print("Downloading model (this is ~80MB and might take a minute)...")
# This line triggers the download on the first run, and loads from cache after that
model = SentenceTransformer('all-MiniLM-L6-v2')

# Test if it can actually process text
embeddings = model.encode(["Checking if the model works"])

print(f"Success! Model downloaded. Output shape: {embeddings.shape}")