"""
End-to-end native RAG pipeline for PDF.
Loads chunks, embeds them, retrieves context, and answers questions.
Author: Akansha Verma
"""

import os
import cohere
import numpy as np
from openai import OpenAI
from config import OPENAI_API_KEY, COHERE_KEY
from chunking_data import get_chunks

# ======================================================
# Step 1: Setting Up and Loading Data
# ======================================================
client = OpenAI(api_key=OPENAI_API_KEY)
cohere_client = cohere.Client(api_key=COHERE_KEY)
os.makedirs("./data", exist_ok=True)
print("Imports OK")

# Get chunks with proper overlap to preserve list integrity
pdf_docs = get_chunks(chunk_size=1200, chunk_overlap=200)
chunk_texts = [doc.page_content for doc in pdf_docs]

# ======================================================
# Step 2: Generate Embeddings
# ======================================================

def get_embeddings_in_batches(texts, batch_size=100):
    print(f"\nGenerating embeddings for {len(texts)} chunks using text-embedding-3-small...")
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(batch_embeddings)
        
    return all_embeddings

chunk_embeddings = get_embeddings_in_batches(chunk_texts)

# Convert lists to numpy arrays for matrix operations
chunk_matrix = np.array(chunk_embeddings)
normed_chunk_matrix = chunk_matrix / np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
assert chunk_matrix.shape[0] == len(chunk_texts)
# ======================================================
# Step 3: Implement Vector Search
# ======================================================

def get_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def retrieve_top_k(query_str, k=8):
    """Vectorized cosine similarity search."""
    q_emb = np.array(get_embedding(query_str))
    q_norm = q_emb / np.linalg.norm(q_emb)
    
    scores = np.dot(normed_chunk_matrix, q_norm)
    top_indices = np.argsort(scores)[::-1][:k]
    
    return [(scores[i], chunk_texts[i]) for i in top_indices]


# ======================================================
# Step 4: Build RAG Query Function
# ======================================================

def rag_query(question, k=8):
    results = retrieve_top_k(question, k=k)
    
    top_scores = [score for score, _ in results]
    top_chunks = [text for _, text in results]
    
    print(f"\n[Search Trace] Retrieved top {k} chunks.")
    print(f"[Search Trace] Top similarity score: {top_scores[0]:.3f} | Score #8: {top_scores[-1]:.3f}")
    
    context_string = "\n\n---\n\n".join(top_chunks)
    
    prompt = f"""Use ONLY the context below to answer the question.
Read through ALL context chunks carefully before responding.
If the context does not contain enough information to answer the question, say "I don't know."

Context:
{context_string}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content, top_chunks, top_scores

# --- VALIDATE RAG ---
print("\nRunning RAG pipeline query...")

test_query = "What are the four ethical principles for Trustworthy AI?"
final_answer, retrieved_evidence, scores = rag_query(question=test_query, k=8)

print(f"\nQ: {test_query}")
print(f"A:\n{final_answer}\n")

# --- CHECKPOINT: MULTI-QUESTION EVALUATION ---

test_questions = [
    "What are the 7 key requirements for Trustworthy AI?",
    "What are the three components of Trustworthy AI?",
    "How is the weather today?"  # Guardrail test
]

print("=" * 70)
print("RUNNING RAG CHECKPOINT EVALUATION")
print("=" * 70)

for idx, q in enumerate(test_questions, start=1):
    answer, retrieved_chunks, scores = rag_query(question=q, k=8)
    
    print(f"\n--- Question {idx}: {q} ---")
    print(f"Top Similarity Score: {scores[0]:.4f}")
    print(f"Grounded Answer:\n{answer}")
    print("-" * 70)

# ======================================================
# Step 5: ADD COHERE RERANKING (TWO-STAGE PIPELINE)
# ======================================================

def rerank_with_cohere(query_str, candidate_results, top_n=5):
    """Stage 2: Re-score top candidate chunks using Cohere Rerank API."""
    candidate_texts = [text for _, text in candidate_results]
    response = cohere_client.rerank(
        model="rerank-v4.0-pro",
        query=query_str,
        documents=candidate_texts,
        top_n=min(top_n, len(candidate_texts))
    )

    reranked = []
    for item in response.results:
        original_score, original_text = candidate_results[item.index]
        reranked.append((item.relevance_score, original_text, item.index))

    return reranked


def rag_query_with_cohere(question, initial_k=20, final_top_n=5):
    """Two-stage RAG query: vector retrieval -> Cohere reranking -> LLM generation."""
    candidate_results = retrieve_top_k(question, k=initial_k)
    reranked_results = rerank_with_cohere(question, candidate_results, top_n=final_top_n)

    print(f"\n[Cohere Rerank Trace] Retained top {len(reranked_results)} chunks after reranking:")
    for rank, (score, text, original_rank) in enumerate(reranked_results, start=1):
        print(f"  Rank #{rank} (Cohere Score: {score:.4f} | Original Vector Rank: {original_rank + 1})")
        print(f"  Snippet: {text[:100].strip().replace(chr(10), ' ')}...\n")

    top_chunks = [text for _, text, _ in reranked_results]
    context_string = "\n\n---\n\n".join(top_chunks)

    prompt = f"""Use ONLY the context below to answer the question.
Read through ALL context chunks carefully before responding.
If the context does not contain enough information to answer the question, say "I don't know."

Context:
{context_string}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    rerank_scores = [score for score, _, _ in reranked_results]
    return response.choices[0].message.content, top_chunks, rerank_scores

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING STEP 5: COHERE RERANKING EVALUATION")
    print("=" * 70)

    rerank_answer, rerank_chunks, rerank_scores = rag_query_with_cohere(
        question=test_query,
        initial_k=20,
        final_top_n=5
    )

    print(f"\nQ: {test_query}")
    print(f"A:\n{rerank_answer}\n")

    for idx, q in enumerate(test_questions, start=1):
        answer, retrieved_chunks, scores = rag_query_with_cohere(
            question=q,
            initial_k=20,
            final_top_n=5
        )

        print(f"\n--- Step 5 Question {idx}: {q} ---")
        print(f"Top Reranked Score: {scores[0]:.4f}")
        print(f"Grounded Answer:\n{answer}")
        print("-" * 70)
