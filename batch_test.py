import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KNOWLEDGE_FILE = "company_info_sample.txt"
QUESTIONS_FILE = "sample_questions.csv"
OUTPUT_FILE = "batch_test_results.csv"
SIMILARITY_THRESHOLD = 0.12


def load_knowledge_base(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the file: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    return text


def split_into_sections(text):
    pattern = r"\n?([A-Za-z0-9&\-\/ ]+):\n"
    parts = re.split(pattern, text)

    sections = []

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1].strip()

        if title and content:
            sections.append({
                "section": title,
                "text": content
            })

    return sections


def build_knowledge_dataframe(sections):
    rows = []

    for index, item in enumerate(sections, start=1):
        rows.append({
            "chunk_id": index,
            "section": item["section"],
            "text": item["text"]
        })

    return pd.DataFrame(rows)


def build_vectorizer(df):
    searchable_text = [
        row["section"] + " " + row["text"]
        for _, row in df.iterrows()
    ]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(searchable_text)

    return vectorizer, tfidf_matrix


def retrieve_best_match(query, df, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

    best_index = similarity_scores.argmax()
    best_score = similarity_scores[best_index]
    best_match = df.iloc[best_index]

    return best_match, best_score


def generate_simple_answer(best_match, score):
    if score < SIMILARITY_THRESHOLD:
        return (
            "This question seems outside the current textile company knowledge base. "
            "The assistant should not make up an answer."
        )

    return (
        f"Based on the company information, this question is most related to "
        f"the '{best_match['section']}' section. {best_match['text']}"
    )


def normalize_text(value):
    return str(value).strip().lower()


def main():
    knowledge_text = load_knowledge_base(KNOWLEDGE_FILE)
    sections = split_into_sections(knowledge_text)
    knowledge_df = build_knowledge_dataframe(sections)

    questions_df = pd.read_csv(QUESTIONS_FILE)

    vectorizer, tfidf_matrix = build_vectorizer(knowledge_df)

    results = []

    for _, row in questions_df.iterrows():
        question_id = row["question_id"]
        category = row["category"]
        question = row["question"]
        expected_section = row["expected_section"]
        is_in_scope = row["is_in_scope"]

        best_match, score = retrieve_best_match(
            question,
            knowledge_df,
            vectorizer,
            tfidf_matrix
        )

        predicted_section = best_match["section"]
        final_answer = generate_simple_answer(best_match, score)

        expected_clean = normalize_text(expected_section)
        predicted_clean = normalize_text(predicted_section)

        section_match = expected_clean == predicted_clean

        if score < SIMILARITY_THRESHOLD:
            predicted_scope = "No"
        else:
            predicted_scope = "Yes"

        scope_match = normalize_text(is_in_scope) == normalize_text(predicted_scope)

        results.append({
            "question_id": question_id,
            "category": category,
            "question": question,
            "expected_section": expected_section,
            "predicted_section": predicted_section,
            "similarity_score": round(score, 4),
            "expected_scope": is_in_scope,
            "predicted_scope": predicted_scope,
            "section_match": section_match,
            "scope_match": scope_match,
            "final_answer": final_answer
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    section_accuracy = results_df["section_match"].mean()
    scope_accuracy = results_df["scope_match"].mean()

    print("\nBatch testing completed.")
    print(f"Knowledge sections loaded: {len(knowledge_df)}")
    print(f"Questions tested: {len(results_df)}")
    print(f"Exact section match accuracy: {section_accuracy:.2%}")
    print(f"Scope detection accuracy: {scope_accuracy:.2%}")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()