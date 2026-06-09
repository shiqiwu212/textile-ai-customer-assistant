# Week 4 Assignment: Chunking
# Source data: company_info_sample.txt
def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


file_content = read_file("company_info_sample.txt")

print("Source data loaded successfully.")
print("\n--- Source Data Preview ---")
print(file_content)

# Method 1: Fixed-size chunking
def fixed_size_chunking(text, chunk_size):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


fixed_chunks = fixed_size_chunking(file_content, 50)

print("\n--- Method 1: Fixed-size Chunks ---")
for i, chunk in enumerate(fixed_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

# Method 2: Section-based chunking
def section_based_chunking(text):
    sections = text.strip().split("\n\n")
    chunks = []

    for section in sections:
        clean_section = section.strip()
        if clean_section:
            chunks.append(clean_section)

    return chunks


section_chunks = section_based_chunking(file_content)

print("\n--- Method 2: Section-based Chunks ---")
for i, chunk in enumerate(section_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

# Method 3: Sentence-level chunking
def sentence_level_chunking(text, sentences_per_chunk):
    text = text.replace("\n", " ")
    sentences = text.split(". ")
    chunks = []

    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ". ".join(sentences[i:i + sentences_per_chunk])
        if not chunk.endswith("."):
            chunk += "."
        chunks.append(chunk)

    return chunks


sentence_chunks = sentence_level_chunking(file_content, 2)

print("\n--- Method 3: Sentence-level Chunks ---")
for i, chunk in enumerate(sentence_chunks, 1):
    print(f"\nChunk {i}:")
    print(chunk)

# Chosen strategy explanation
print("\n--- Chosen Chunking Strategy ---")
print("I choose section-based chunking as the final strategy.")
print("This method works best for this data because the company information is already organized by topics.")
print("Each section, such as Company Overview, Core Business, Quality and Service, and Sustainability, can stay as one complete chunk.")
print("This makes it easier for an AI assistant to retrieve the right information when answering customer questions.")

# Create embeddings for chosen chunks using spaCy
import spacy

nlp = spacy.load("en_core_web_md")

print("\n--- Embeddings for Chosen Section-based Chunks ---")

embedded_chunks = []

for i, chunk in enumerate(section_chunks, 1):
    doc = nlp(chunk)
    vector = doc.vector

    embedded_chunks.append({
        "chunk_id": i,
        "text": chunk,
        "embedding": vector
    })

    print(f"\nChunk {i}:")
    print(chunk)
    print(f"Embedding dimension: {len(vector)}")
    print(f"First 10 embedding values: {vector[:10]}")

# Save chunked and embedded results to a JSON file
import json

output_data = []

for item in embedded_chunks:
    output_data.append({
        "chunk_id": item["chunk_id"],
        "text": item["text"],
        "embedding": item["embedding"].tolist()
    })

with open("chunked_embedded_output.json", "w", encoding="utf-8") as file:
    json.dump(output_data, file, indent=2)

print("\nSaved chunked and embedded results to chunked_embedded_output.json")