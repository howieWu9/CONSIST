import json
import threading
import random
import hashlib
import logging
from typing import List, Optional, Tuple
import time
import numpy as np
import torch
import redis
import uuid
import configs
from configs import logger, REDIS_CONFIG
from core.llm_functions import score_triples
import requests
def hash_text(input_string: str) -> str:
    """
    Generate a 15-character MD5 hash for a given string.

    Args:
        input_string (str): Input string.

    Returns:
        str: 15-character hexadecimal hash value.
    """
    return hashlib.md5(input_string.encode()).hexdigest()[-15:]

class PathRanker:
    """
    Unified path scorer supporting multiple ranking strategies:
    - embedding: semantic similarity
    - llm: large language model
    - hybrid: embedding pre-filter + LLM rerank
    - trained_ranker: custom trained ranker model
    - ranker_llm: trained ranker pre-filter + LLM rerank
    """

    _global_ranker_lock = threading.Lock()

    def __init__(
        self,
        llm_client: Optional[object] = None,
        embedding_model: Optional[object] = None,
        strategy: str = "hybrid"
    ):
        """
        Args:
            llm_client: LLM client (required for LLM strategies)
            embedding_model: Embedding model (required for embedding/hybrid strategies)
            strategy: Ranking strategy ['embedding', 'llm', 'hybrid', 'trained_ranker', 'ranker_llm']
        """
        self.strategy = strategy
        self.llm = llm_client
        self.embedding = embedding_model
        self.trained_score_model = None
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.redis_client.ping()
        self._ranker_lock = threading.Lock()
        # Strategy validation
        if self.strategy == "llm" and not self.llm:
            raise ValueError("LLM strategy requires 'llm_client'.")
        if self.strategy in {"embedding", "hybrid"} and not self.embedding:
            raise ValueError("Embedding strategy requires 'embedding_model'.")

    def _get_top_k_by_scores(
        self, scores: List[float], paths: List['Path'], k: int
    ) -> Tuple[List[float], List['Path']]:
        """
        Return top-k scored paths and their scores.
        """
        combined = list(zip(scores, paths))
        combined_sorted = sorted(combined, key=lambda x: x[0], reverse=True)
        top_k = combined_sorted[:k]
        top_k_scores = [item[0] for item in top_k]
        top_k_paths = [item[1] for item in top_k]
        return top_k_scores, top_k_paths

    def score_and_rank_paths(self, paths: List['Path'], query: str, strategy=None) -> List['Path']:
        """
        Unified scoring entry point.

        Args:
            paths (List[Path]): Candidate paths.
            query (str): Query string.

        Returns:
            List[Path]: Ranked paths.
        """

        if strategy is None:
            strategy = self.strategy

        if strategy == "embedding":
            scores = self._score_by_embedding(paths, query)
        elif strategy == "llm":
            scores = self._score_by_llm(paths, query)
        elif strategy == "trained_ranker":
            scores = self._score_by_ranker(paths, query)
        elif strategy == "hybrid":
            if len(paths) > 100:
                scores = self._score_by_embedding(paths, query)
                top_k_scores, top_k_paths = self._get_top_k_by_scores(scores, paths, 100)
                scores = self._score_by_llm(top_k_paths, query)
                paths = top_k_paths
            else:
                scores = self._score_by_llm(paths, query)
        elif strategy == "ranker_llm":
            if len(paths) > 30:
                scores = self._score_by_ranker(paths, query)
                top_k_scores, top_k_paths = self._get_top_k_by_scores(scores, paths, 30)
                scores = self._score_by_llm(top_k_paths, query)
                paths = top_k_paths
            else:
                scores = self._score_by_llm(paths, query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Update path scores
        for p, s in zip(paths, scores):
            while len(p.scores) >= len(p.relations) and len(p.scores) > 0: # 
                p.scores.pop()
            p.scores.append(s)


        return sorted(paths, key=lambda x: x.score, reverse=True)



    def _score_by_ranker(self, paths: list, query: str, timeout: int = 1200):
        """
         Redis Stream 
         Redis 
        """
        if len(paths) > 3000:
            logger.error(f"Ranking by ranker: {len(paths)} paths, random sampling 3000 paths.")
            paths = random.sample(paths, 3000)

        # 1. ID
        correlation_id = str(uuid.uuid4())
        logger.debug(f"[{correlation_id[:8]}] : '{query}', : {len(paths)}")

        # 2. 
        triple_paths = [
            [
                (
                    r['begin']['mention'],
                    r['r'],
                    r['end']['mention'],
                    r['text'] if r.get('text') else r['title']
                )
                for r in p.to_dict()['relations']
            ]
            for p in paths
        ]
        request_payload = {'query': query, 'paths': triple_paths}
        stream_message = {
            'correlation_id': correlation_id.encode('utf-8'),
            'payload': json.dumps(request_payload).encode('utf-8')
        }

        # ---  ---
        # 3.  Pipeline 
        pipe = self.redis_client.pipeline()
        pipe.time()  #  Redis 
        pipe.xadd('request_stream', stream_message)
        pipeline_results = pipe.execute()

        # 4.  Pipeline ID
        redis_time = pipeline_results[0]  # time() 
        #  stream ID '-'
        # 
        start_id = f"{redis_time[0] * 1000 + redis_time[1] // 1000}-0"

        logger.debug(f"[{correlation_id[:8]}]  ID '{start_id}'  'response_stream'")

        # 5. ID
        start_time = time.time()

        while time.time() - start_time < timeout:
            #  ()
            remaining_time_ms = int((timeout - (time.time() - start_time)) * 1000)
            if remaining_time_ms <= 0:
                break

            try:
                responses = self.redis_client.xread(
                    {'response_stream': start_id},
                    count=10,  # 
                    block=remaining_time_ms
                )
            except redis.exceptions.TimeoutError:
                continue  # block 

            if not responses:
                continue

            for stream_name, messages in responses:
                for message_id, data in messages:
                    # ID
                    start_id = message_id

                    # 
                    response_correlation_id = data.get('correlation_id') or data.get(b'correlation_id')
                    response_result = data.get('result') or data.get(b'result')
                    if isinstance(response_correlation_id, bytes):
                        response_correlation_id = response_correlation_id.decode('utf-8')
                    if isinstance(response_result, bytes):
                        response_result = response_result.decode('utf-8')

                    if response_correlation_id == correlation_id:
                        logger.debug(f"[{correlation_id[:8]}] !")
                        if response_result is None:
                            raise KeyError("response_stream message missing result")
                        result = json.loads(response_result)

                        if "error" in result:
                            raise Exception(f": {result['error']}")

                        return result["result"]

        raise TimeoutError(f"[{correlation_id[:8]}]  ({timeout})")
    def close(self):
        self.redis_client.close()
        print(" Redis ")


    def _score_by_embedding(self, paths: List['Path'], query: str) -> List[float]:
        """
        Pure semantic similarity strategy.
        """
        if len(paths) > 2000:
            logger.error(f"Ranking by embedding: {len(paths)} paths, random sampling 1000 paths.")
            paths = random.sample(paths, 2000)
        elif len(paths) > 1000:
            logger.warning(f"Ranking by embedding: {len(paths)} paths.")
        else:
            logger.debug(f"Ranking by embedding: {len(paths)} paths.")

        similarities = self._get_from_cache(paths)
        if similarities is None:
            format_strings = [path.format_string for path in paths]
            path_embeds = self.embedding.embed_documents(format_strings)
            query_embed = self.embedding.embed_query(query)
            similarities = [self._cos_sim(query_embed, pe) for pe in path_embeds]
            self._set_to_cache(paths, similarities)
        return similarities

    def _set_to_cache(self, paths: List['Path'], similarities: List[float]) -> None:
        """
        Cache similarities for the given paths.
        """
        cached_key = hash_text(str([p.to_dict() for p in paths]))
        try:
            self.redis_client.set(cached_key, json.dumps(similarities))
        except Exception as e:
            logger.warning(f"Failed to set cache: {e}")

    def _get_from_cache(self, paths: List['Path']) -> Optional[List[float]]:
        """
        Retrieve similarities from cache.
        """
        cached_key = hash_text(str([p.to_dict() for p in paths]))
        try:
            similarities = self.redis_client.get(cached_key)
            if similarities is not None:
                return json.loads(similarities)
        except Exception as e:
            logger.warning(f"Failed to get cache: {e}")
        return None

    def _score_by_llm(self, paths: List['Path'], query: str) -> List[float]:
        """
        Pure LLM scoring strategy.
        """
        if len(paths) > 50:
            paths
        scored = score_triples(
            llm=self.llm,
            query=query,
            paths=paths,
            batch_size=configs.RETRIEVER_CONFIG['rank_batch_size']
        )
        return [s['score'] for s in scored]

    @staticmethod
    def _cos_sim(vec1: List[float], vec2: List[float]) -> float:
        """
        Cosine similarity calculation.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
