# Week 4 Assignment: RAG

import pandas as pd


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


file_content = read_file("company_info_sample.txt")

# Store the project information in a dataframe for later retrieval
rag_data = [
    {
        "chunk_id": 1,
        "topic": "Company Overview",
        "text": "Foshan Zhuoyi Textile Co., Ltd. is a textile company located in the Pearl River Delta, near Hong Kong and Macao. The company was established in 2001 and focuses on denim fabric manufacturing and related textile services."
    },
    {
        "chunk_id": 2,
        "topic": "Core Business",
        "text": "The company specializes in denim fabric production. It provides textile-related services such as fabric development, weaving, dyeing, and production support."
    },
    {
        "chunk_id": 3,
        "topic": "Quality and Service",
        "text": "The company focuses on quality control, technology development, and customer service. It uses testing equipment and production experience to support fabric development and production needs."
    },
    {
        "chunk_id": 4,
        "topic": "Sustainability",
        "text": "The company values social responsibility and environmental protection. It supports recycling and regeneration in textile production and aims to promote sustainable development."
    },
    {
        "chunk_id": 5,
        "topic": "Customer Inquiry Examples",
        "text": "International customers may ask questions about fabric types, denim products, customization, sustainability, production services, and contact information."
    }
]

df = pd.DataFrame(rag_data)

print("Source data loaded successfully.")
print("\n--- RAG DataFrame ---")
print(df[["chunk_id", "topic"]])

# Retrieval method: TF-IDF + cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

chunk_texts = df["text"].tolist()

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)

query = "Do you offer sustainable denim fabric and recycling support?"
query_vector = vectorizer.transform([query])

similarities = cosine_similarity(query_vector, chunk_vectors)[0]
best_index = similarities.argmax()

best_topic = df.iloc[best_index]["topic"]
best_text = df.iloc[best_index]["text"]
best_score = similarities[best_index]

print("\n--- Retrieval Result ---")
print(f"Query: {query}")
print(f"Best topic: {best_topic}")
print(f"Cosine similarity score: {best_score:.4f}")
print("Retrieved text:")
print(best_text)

# Generate a concise answer based on the retrieved text
def generate_answer(query, retrieved_text):
    answer = (
        f"Based on the retrieved company information, the answer to your question is: "
        f"{retrieved_text}"
    )
    return answer


final_answer = generate_answer(query, best_text)

print("\n--- Final Answer ---")
print(final_answer)

print("\n--- RAG Pipeline Summary ---")
print("1. The company information was stored in a pandas DataFrame.")
print("2. TF-IDF and cosine similarity were used to retrieve the most relevant chunk.")
print("3. The retrieved chunk was used to generate a concise grounded answer.")
print("4. The final answer can be traced back to the original source data.")