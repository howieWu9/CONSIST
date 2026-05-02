"""

====================================================

    - JSON 
    - 
    - 
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List

from tqdm import tqdm
from langchain_core.documents import Document

from utils.embeddings import CustomEmbedding
from utils.vector_store import MyVectorStore
import configs
from utils.neo4j_operator import sanitize_label
import tiktoken


def split_text(text: str,
               overlap: int = 16,
               max_chunk_size: int = 128,
               min_chunk_size: int = 100,
               padding: str = " ...",
               model_name: str = 'gpt-3.5-turbo') -> List[str]:
    """
    

    Args:
        text (str): 
        overlap (int): token
        max_chunk_size (int): token
        min_chunk_size (int): token
        padding (str): 
        model_name (str): tokenization

    Returns:
        List[str]: 
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)

    step_size = max_chunk_size - overlap
    pos = 0
    chunks = []

    while pos < len(tokens):
        end_pos = pos + max_chunk_size

        if end_pos >= len(tokens):
            chunk = tokens[pos:len(tokens)]
            if len(chunk) < min_chunk_size and chunks:
                chunks[-1].extend(chunk)
            else:
                chunks.append(chunk)
            break
        else:
            chunk = tokens[pos:end_pos]
            chunks.append(chunk)
            pos += step_size

    texts = [encoding.decode(chunk) for chunk in chunks]

    padded_texts = []
    num_chunks = len(texts)

    if num_chunks <= 1:
        return texts

    for i, chunk_text in enumerate(texts):
        if i == 0:
            padded_chunk = chunk_text + padding
        elif i == num_chunks - 1:
            padded_chunk = padding + chunk_text
        else:
            padded_chunk = padding + chunk_text + padding
        padded_texts.append(padded_chunk)
    return padded_texts


def process_sample(sample: dict, vector_store: MyVectorStore, lock: Lock) -> None:
    """

    Args:
        sample (dict):  question  context
        vector_store (MyVectorStore): 
        lock (Lock): 
    """
    query = sample["question"]

    # 
    for article in sample["context"]:
        documents = []
        title = article[0].strip()

        chunks = []
        for para in article[1]:
             chunks.extend(split_text(para))

        for chunk in chunks:
            # 
            if len(chunk.strip()) > 15:
                #  page_content | 
                page_content = f"{title} | {chunk.strip()}"
                meta = {
                    "title": title,
                    "sentence": chunk.strip(),
                }
                documents.append(Document(page_content=page_content, metadata=meta))

        if not documents:
            continue

        try:
            # 
            embeddings = vector_store.embedding_documents(documents)

            # 
            with lock:
                vector_store.add_embedding_to_collection(sanitize_label(query), documents, embeddings)
                vector_store.add_embedding_to_collection(sanitize_label(title), documents, embeddings)
                vector_store.add_embedding_to_collection(configs.DATASET_CONFIG["domain"], documents, embeddings)
        except Exception as e:
            raise e



def documents_to_vector_store(max_workers: int | None = None) -> None:
    """

    Args:
        max_workers (int | None):  os.cpu_count()
    """

    #  Embedding 
    embeddings = CustomEmbedding(
        api_key=configs.EMBEDDING_CONFIG["api_key"],
        base_url=configs.EMBEDDING_CONFIG["base_url"],
        model_name=configs.EMBEDDING_CONFIG["model_name"],
    )

    # 
    os.makedirs(configs.DATASET_CONFIG["doc_vector_store_path"], exist_ok=True)

    # 
    vector_store = MyVectorStore(configs.DATASET_CONFIG["doc_vector_store_path"], embeddings)

    # 
    try:
        with open(configs.DATASET_CONFIG["dataset_file"], encoding="utf-8") as f:
            samples = json.load(f)
    except Exception as e:
        print(f"Failed to load {configs.DATASET_CONFIG['document_file']}: {e}")
        return

    # 
    lock = Lock()

    #  CPU 
    if max_workers is None:
        max_workers = os.cpu_count() or 2

    # 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(lambda s: process_sample(s, vector_store, lock), samples),
                total=len(samples),
                desc="Processing",
            )
        )
    vector_store.force_save_all()
    print(vector_store.get_performance_stats())


# -------------------------  -------------------------
if __name__ == "__main__":
    documents_to_vector_store(max_workers=20)
