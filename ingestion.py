#imports the PDF loader from the langchain_community library
from langchain_community.document_loaders import PyPDFLoader
#imports the text-splitting functionality
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

#splits the loaded documents into smaller chunks for easier processing
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

    chunks = text_splitter.split_documents(documents)
    #to create embeddings.
    return chunks


if __name__ == "__main__":
    pdf_path = "data/sample_company_policy.pdf"

    documents = load_pdf(pdf_path)

    chunks = split_documents(documents)

    print(f"Total pages loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content)
        print("Metadata:", chunk.metadata)
