# Week 4 Assignment: Embeddings
# Source data: company_info_sample.txt

def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


file_content = read_file("company_info_sample.txt")

# Manually break the data into sample parts for the project
sample_parts = [
    {
        "part_id": 1,
        "topic": "Company Overview",
        "text": "Foshan Zhuoyi Textile Co., Ltd. is a textile company located in the Pearl River Delta, near Hong Kong and Macao. The company was established in 2001 and focuses on denim fabric manufacturing and related textile services."
    },
    {
        "part_id": 2,
        "topic": "Core Business",
        "text": "The company specializes in denim fabric production. It provides textile-related services such as fabric development, weaving, dyeing, and production support."
    },
    {
        "part_id": 3,
        "topic": "Quality and Service",
        "text": "The company focuses on quality control, technology development, and customer service. It uses testing equipment and production experience to support fabric development and production needs."
    },
    {
        "part_id": 4,
        "topic": "Sustainability",
        "text": "The company values social responsibility and environmental protection. It supports recycling and regeneration in textile production and aims to promote sustainable development."
    },
    {
        "part_id": 5,
        "topic": "Customer Inquiry Examples",
        "text": "International customers may ask questions about fabric types, denim products, customization, sustainability, production services, and contact information."
    }
]

print("Source data loaded successfully.")
print("\n--- Sample Parts ---")

for part in sample_parts:
    print(f"\nPart {part['part_id']}: {part['topic']}")
    print(part["text"])

# Create embeddings and store vectors in a pandas DataFrame
import spacy
import pandas as pd

nlp = spacy.load("en_core_web_md")

embedding_rows = []

for part in sample_parts:
    doc = nlp(part["text"])
    vector = doc.vector

    embedding_rows.append({
        "part_id": part["part_id"],
        "topic": part["topic"],
        "text": part["text"],
        "embedding_dimension": len(vector),
        "embedding_vector": vector.tolist()
    })

df = pd.DataFrame(embedding_rows)

print("\n--- Embeddings DataFrame ---")
print(df[["part_id", "topic", "embedding_dimension"]])

# Run cosine similarity comparisons
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


comparison_samples = [
    {
        "query": "Do you support recycling, regeneration, environmental protection, and sustainable development?",
        "expected_topic": "Sustainability"
    },
    {
        "query": "What fabric development, weaving, and dyeing services do you provide?",
        "expected_topic": "Core Business"
    },
    {
        "query": "Do you support quality control and textile product development?",
        "expected_topic": "Quality and Service"
    },
    {
        "query": "The basketball team won the game after scoring many points.",
        "expected_topic": "Out of Scope"
    },
    {
        "query": "The patient visited a doctor for a medical appointment.",
        "expected_topic": "Out of Scope"
    }
]

print("\n--- Cosine Similarity Comparisons ---")

for sample in comparison_samples:
    query_doc = nlp(sample["query"])
    query_vector = query_doc.vector

    scores = []

    for row in embedding_rows:
        score = cosine_similarity(query_vector, np.array(row["embedding_vector"]))
        scores.append({
            "topic": row["topic"],
            "score": score
        })

    best_match = max(scores, key=lambda x: x["score"])

    print(f"\nQuery: {sample['query']}")
    print(f"Expected topic: {sample['expected_topic']}")
    print(f"Best match: {best_match['topic']}")
    print(f"Cosine similarity score: {best_match['score']:.4f}")

# Second model: TF-IDF vectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

part_texts = [part["text"] for part in sample_parts]
part_topics = [part["topic"] for part in sample_parts]
query_texts = [sample["query"] for sample in comparison_samples]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(part_texts + query_texts)

part_vectors = tfidf_matrix[:len(part_texts)]
query_vectors = tfidf_matrix[len(part_texts):]

print("\n--- TF-IDF Cosine Similarity Comparisons ---")

tfidf_results = []

for i, sample in enumerate(comparison_samples):
    similarities = sklearn_cosine_similarity(query_vectors[i], part_vectors)[0]
    best_index = similarities.argmax()
    best_topic = part_topics[best_index]
    best_score = similarities[best_index]

    tfidf_results.append({
        "query": sample["query"],
        "expected_topic": sample["expected_topic"],
        "best_match": best_topic,
        "score": best_score
    })

    print(f"\nQuery: {sample['query']}")
    print(f"Expected topic: {sample['expected_topic']}")
    print(f"Best match: {best_topic}")
    print(f"TF-IDF cosine similarity score: {best_score:.4f}")

#Conclusion
print("\n--- Final Model Choice ---")
print("For this project, TF-IDF works better than the spaCy embedding model for the small sample dataset.")
print("The TF-IDF model gives higher scores for textile-related questions and much lower scores for out-of-scope questions.")
print("This fits the use case because the AI assistant needs to separate relevant textile customer questions from unrelated questions.")