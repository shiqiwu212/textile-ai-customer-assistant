import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_FILE = "company_info_sample.txt"
SIMILARITY_THRESHOLD = 0.12


def load_knowledge_base(file_path):
    """
    Read the company knowledge base text file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def split_into_sections(text):
    """
    Split the knowledge base into sections.

    The expected format is:
    Section Title:
    Section content...
    """
    pattern = r"\n?([A-Za-z0-9&\-\/ ]+):\n"
    parts = re.split(pattern, text)

    sections = []

    # If the file starts directly with a title, re.split returns an empty first item.
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip()

        if title and content:
            sections.append({
                "section": title,
                "text": content
            })

    return sections


def build_dataframe(sections):
    """
    Convert sections into a pandas DataFrame.
    """
    data = []

    for index, item in enumerate(sections, start=1):
        data.append({
            "chunk_id": index,
            "section": item["section"],
            "text": item["text"]
        })

    return pd.DataFrame(data)


def retrieve_best_match(query, df):
    """
    Retrieve the most relevant section using TF-IDF and cosine similarity.
    """
    documents = df["text"].tolist()
    section_titles = df["section"].tolist()

    searchable_text = [
        section_titles[i] + " " + documents[i]
        for i in range(len(documents))
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(searchable_text)

    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]

    best_match = df.iloc[best_index]

    return best_match, best_score


def generate_answer(query, best_match, score):
    """
    Generate a simple grounded answer based on the retrieved section.
    """
    section = best_match["section"]
    context = best_match["text"]

    if score < SIMILARITY_THRESHOLD:
        return (
            "This question seems outside the current textile company knowledge base. "
            "I can only answer questions related to textile products, company services, "
            "customization, sustainability, production, pricing, shipping, or contact information."
        )

    answer = (
        f"Based on the company information, this question is most related to the "
        f"'{section}' section. {context}"
    )

    return answer


def run_interactive_assistant(df):
    """
    Run an interactive terminal demo.
    """
    print("\n==============================================")
    print("Textile Company AI Customer Inquiry Assistant")
    print("==============================================")
    print("Type a customer question and press Enter.")
    print("Type 'exit' to stop the program.\n")

    while True:
        query = input("Customer Question: ").strip()

        if query.lower() in ["exit", "quit", "q"]:
            print("\nThank you. Demo ended.")
            break

        if query == "":
            print("Please enter a question.\n")
            continue

        best_match, score = retrieve_best_match(query, df)
        final_answer = generate_answer(query, best_match, score)

        print("\n--- Best Matched Section ---")
        print(best_match["section"])

        print("\n--- Similarity Score ---")
        print(round(score, 4))

        print("\n--- Retrieved Context ---")
        print(best_match["text"])

        print("\n--- Final Answer ---")
        print(final_answer)

        print("\n----------------------------------------------\n")


def main():
    knowledge_text = load_knowledge_base(KNOWLEDGE_FILE)
    sections = split_into_sections(knowledge_text)
    df = build_dataframe(sections)

    print(f"\nLoaded {len(df)} knowledge sections from {KNOWLEDGE_FILE}.")

    run_interactive_assistant(df)


if __name__ == "__main__":
    main()