import argparse
import sys

from pprint import pprint

import chromadb
import ollama

from sentence_transformers import CrossEncoder

from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

path_chroma="chroma"
path_documents_base="documents"

system_prompt = """
You are an AI assistant tasked with providing answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use concise language.
2. Organize your answer into paragraphs.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""

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

def query_collection(collection_name_current, prompt, n_results = 10):
    collection = get_vector_collection(collection_name_current)
    results = collection.query(query_texts=[prompt], n_results=n_results)
    return results

def generate_context(documents):
    context = ""
    
    for document in documents:
        context += document

    return context

def call_llm(context, prompt):
    response = ollama.chat(
        model="llama3.2:latest",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break

def re_rank_cross_encoders(prompt, documents):
    #Re-ranks documents using a cross-encoder model for more accurate relevance scoring
    relevant_text = ""
    relevant_text_ids = []

    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    ranks = encoder_model.rank(prompt, documents, top_k=5)
    for rank in ranks:
        relevant_text += documents[rank["corpus_id"]]
        relevant_text_ids.append(rank["corpus_id"])

    return relevant_text, relevant_text_ids

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
    
    while True:
        user_query = input("Human: ")
        if user_query == "done":
            return
        
        results = query_collection(collection_name_current, user_query)
        #context = generate_context(results.get("documents")[0])
        list_docus = results.get("documents")[0]
        context, relevant_text_ids = re_rank_cross_encoders(user_query, list_docus)
        
        print("AI: ", end="")
        for chunk in call_llm(context=context, prompt=user_query):
            print(chunk, end="")
        print()

if __name__ == "__main__":
    main()