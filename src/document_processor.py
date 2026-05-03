from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import os

def extract_text_from_pdf(pdf_file):
    """Extract text from an uploaded PDF file."""
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    print(f"Extracted {len(text)} characters from {len(reader.pages)} pages")
    return text


def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    """Split text into overlapping chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    print(f"Split into {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


if __name__ == "__main__":
    # Test with a sample PDF
    import sys
    if len(sys.argv) > 1:
        text = extract_text_from_pdf(sys.argv[1])
        chunks = split_text_into_chunks(text)
        print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")
    else:
        print("Usage: python document_processor.py <path_to_pdf>")