from sentence_transformers import SentenceTransformer
import numpy as np

# Documents corpus
documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

# Query
query = 'tell me about bumrah'

# Load encoder
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Embed documents and query (normalize so dot product == cosine similarity)
doc_embeddings = model.encode(documents, normalize_embeddings=True)
query_embedding = model.encode([query], normalize_embeddings=True)[0]

# Cosine similarity using normalized embeddings
scores = np.dot(doc_embeddings, query_embedding)

# Best match
best_idx = int(np.argmax(scores))
best_score = float(scores[best_idx])

print(query)
print(documents[best_idx])
print("similarity score is:", best_score)

# Also show embedding dimension
print(f"Embedding dimension: {len(doc_embeddings[0])}")