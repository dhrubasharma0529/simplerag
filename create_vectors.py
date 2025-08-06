from pinecone import Pinecone
from dotenv import load_dotenv
from google import genai
import os
import fitz # PyMuPDF
load_dotenv()
Pinecone_client = Pinecone(api_key =os.getenv("PINECONE_API"))
vector_index = Pinecone_client.Index("student-kb")
google_client = genai.Client(api_key = os.getenv("GEMINI_API"))
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text 
def embed_text (text):
    response = google_client.models.embed_content(
        model = "gemini-embedding-001",
        contents = text,
        config={
            "output_dimensionality" : 768
        }
    )

    vector = response.embeddings[0].values
    return vector

def upsert_vectors_to_pinecone(document_texts):
    upsert_data = []
    for idx,text in enumerate(document_texts):
        vector = embed_text(text)

        vector_id = f"doc-{idx}"
        metadata = {
            "text":text
        }
        upsert_data.append((vector_id,vector,metadata))
    vector_index.upsert(upsert_data)
if __name__ == "__main__":
    pdf_dir = "documents"
    document_dirs = os.listdir(pdf_dir)
    document_texts = []
    for file_path in document_dirs:
        text = extract_text_from_pdf(os.path.join(pdf_dir,file_path))
        document_texts.append(text)
    upsert_vectors_to_pinecone(document_texts)
    print("All documents processed and vectors created.")