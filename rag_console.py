import argparse
import sys

from pprint import pprint

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
        pprint(chunks, indent=4)
        sys.exit()

if __name__ == "__main__":
    main()