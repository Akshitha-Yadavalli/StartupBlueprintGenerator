import os

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


DATA_PATH = "data"

VECTOR_DB = "vectorstore"


def create_vector_database():

    loader = PyPDFDirectoryLoader(DATA_PATH)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(docs, embeddings)

    db.save_local(VECTOR_DB)

    print("Vector Database Created Successfully!")


def retrieve_context(query):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        VECTOR_DB,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = db.similarity_search(query, k=4)

    context = ""

    for doc in docs:
        context += doc.page_content + "\n\n"

    return context


if __name__ == "__main__":
    create_vector_database()