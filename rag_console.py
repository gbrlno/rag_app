import argparse
import sys

from pprint import pprint

import chromadb

from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

path_chroma="chroma"
path_documents_base="documents"


def load_documents(path_documents):
    document_loader = PyPDFDirectoryLoader(path_documents)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_vector_collection(collection_name_current):
    ollama_ef = OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name="nomic-embed-text:latest",
    )

    chroma_client = chromadb.PersistentClient(path=path_chroma)
    return chroma_client.get_or_create_collection(
        name=collection_name_current,
        embedding_function=ollama_ef,
        metadata={"hnsw:space": "cosine"},
    )

def add_to_vector_collection(collection_name_current, chunks):
    collection = get_vector_collection(collection_name_current)
    documents, metadatas, ids = [], [], []

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        documents.append(chunk.page_content)
        metadatas.append(chunk.metadata)
        ids.append(chunk_id)
    
    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

def main():
    #use python console_rag.py -p to make it process files in documents
    parser = argparse.ArgumentParser(description="Rag Console App")

    #add arguments
    parser.add_argument("-p", "--process_docs", action="store_true", help="process files in documents folder")
    parser.add_argument('--set_collection', type=str, help="subfolder of documents to be processed", default="small_world")
    
    #parse arguments
    args = parser.parse_args()

    print(f"args.process_docs: {args.process_docs}")
    print(f"args.set_collection: {args.set_collection}")

    collection_name_current = args.set_collection
    path_documents = path_documents_base+"/"+collection_name_current
    
    print(f"collection_name_current: {collection_name_current}")
    print(f"path_documents: {path_documents}")

    if args.process_docs:
        #process files in documents
        documents = load_documents(path_documents)
        chunks = split_documents(documents)
        #pprint(chunks, indent=4)
        add_to_vector_collection(collection_name_current, chunks)

        sys.exit()

if __name__ == "__main__":
    main()