import json
import traceback
from typing import List, Dict, Any, Optional
from utils.embeddings import CustomEmbedding
from utils.neo4j_operator import Neo4jOperations, sanitize_label
from utils.vector_store import MyVectorStore
from utils.my_llm import LLMClient, ClientConfig
from core.llm_functions import (
    topic_entity_extraction,
    eval_sufficiency_with_llm,
    complete_relations
)
import hashlib
import redis
from configs import *
from core.path_ranker import PathRanker
from collections import defaultdict
# from memory_profiler import profile
from line_profiler_pycharm import profile
import copy
from tabulate import tabulate
from typing import List, Set
from copy import deepcopy #  deepcopy

class GraphRetriever:
    def __init__(self):
        """"""
        self.init_config()

    def init_config(self):
        # 
        self.graph_client = Neo4jOperations(
            uri=NEO4J_CONFIG["uri"],
            auth=NEO4J_CONFIG["auth"]
        )

        # 
        if PIPLINE_CONFIG['expand_COG']:
            self.co_occur_graph = Neo4jOperations(**OCCURRENCE_GRAPH)

        #  redis
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(
            endpoints=LLM_CONFIG['endpoints'],
            client_cfg=ClientConfig(
                cache=self.redis_client,
                temperature=LLM_RUNTIME_CONFIG.get("temperature", 0.3),
            )
        )

        # 
        self.embedding_model = CustomEmbedding(
            api_key=EMBEDDING_CONFIG['api_key'],
            base_url=EMBEDDING_CONFIG['base_url'],
            model_name=EMBEDDING_CONFIG['model_name']
        )

        # 
        self.entities_vector_db = MyVectorStore(
            DATASET_CONFIG['entities_vector_store_path'],
            embedding=self.embedding_model
        )

        self.doc_vector_db = MyVectorStore(
            DATASET_CONFIG['doc_vector_store_path'],
            embedding=self.embedding_model
        )

        self.ranker = PathRanker(
            llm_client=self.llm,
            embedding_model=self.embedding_model,
            strategy=RETRIEVER_CONFIG.get("rank_strategy", "embedding"))

        # 
        self.sufficiency_check = RETRIEVER_CONFIG.get("sufficiency_check", False)
        self.min_similarity = RETRIEVER_CONFIG.get("min_similarity", 0.6)

    def retrieve(self, query: str, domain: str, max_depth: int = 3, max_width: int = 3) -> List[Dict[str, Any]]:
        """
        
        Args:
            query: 
            max_depth: 
        Returns:
            
        """
        logger.info(f": {query}")
        try:
            start_entities = self.preprocess_query(query, domain)
            logger.info(f"start_entities: {[e['mention'] for e in start_entities]}")
            if not start_entities:
                logger.error("")
                return []

            # beam search 
            results = self._beam_search(query, domain, start_entities, max_depth=max_depth, max_width=max_width)

            # 
            if PIPLINE_CONFIG["TCR"]:
                results = self.triple_context_restoration(results, query)

            temp_triples = list(set([
                f"<{triple['begin']['mention']} | {triple['r']} | {triple['end']['mention']}>"
                for path in results for triple in path.get('relations', [])
            ]))
            TRIPLE_COUNT['latent'] += len([r for r in temp_triples if r.split('|')[1].strip().startswith('_')])
            TRIPLE_COUNT['total'] += len(temp_triples)
            # print(TRIPLE_COUNT)

            return sorted(results, key=lambda x: x['score'], reverse=True)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f": {tb}")
            return []

    def get_source(self, ent):
        if ent.get('source_list'):
            return list(set(ent['source_list'] + [ent['source']]))
        elif ent.get('source'):
            return [ent['source']]
        else:
            return []

    def triple_context_restoration(self, paths, query, k=1, retrieve_from_source=True):
        cache_key = f"BASE_GRAPHRAG|TCR:{str(paths)}"
        cached = self.redis_client.get(cache_key)
        if cached:
            return json.loads(cached)
        for path in paths:
            context_sentences = []
            for relation in path['relations']:
                format_rel = f"{relation['begin']['mention']}, {relation['r'].replace('_', ' ')}, {relation['end']['mention']}"
                sources = self.get_source(relation['begin']) + self.get_source(relation['end'])
                if len(sources) != 0:
                    docs = self.doc_vector_db.query_from_collections([sanitize_label(s) for s in sources], query, k)
                else:
                    docs = self.doc_vector_db.query_collection(DATASET_CONFIG['domain'], format_rel)
                context_sentences.extend(
                    [doc.page_content for doc in docs[:k]]
                )
            path['context_sentences'] = context_sentences
        # paths[0]['context_sentences'] += sent_related_query

        self.redis_client.set(cache_key, json.dumps(paths))
        return paths

    def preprocess_query(self, query: str, domain: str) -> List[Dict]:
        """
        
        """
        try:
            query = query.strip()
            # 1. 
            topic_entities = topic_entity_extraction(self.llm, query)

            contents = [
                f"{e['mention']} | {e.get('description', '')}"
                for e in topic_entities
            ]

            final_metadata = []
            # 2. 
            for content in contents:
                # 
                cache_key = f"vector_query_cache:{domain}:{content}"

                try:
                    #  Redis 
                    cached_result = self.redis_client.get(cache_key)
                    if cached_result:
                        # 
                        logger.debug(f": {cache_key[:100]}...")
                        final_metadata.append(json.loads(cached_result))
                        continue  #  content

                except Exception as e:
                    logger.debug(f"Redis : {str(e)}")

                # ---  ---
                logger.debug(f": {cache_key[:100]}...")

                # 
                result = self.entities_vector_db.query_collection(
                    domain,
                    content
                )

                # top-1 
                metadata = result[0].metadata
                final_metadata.append(metadata)

                #  Redis 
                try:
                    self.redis_client.set(cache_key, json.dumps(metadata))  # 1
                    logger.info(f": {cache_key[:100]}...")
                except Exception as e:
                    logger.error(f"Redis : {str(e)}")

            return final_metadata

        except Exception as e:
            logger.error(f": {str(e)}")
            return []
    def shortest_path(self, query, answer, domain):
        """
        
        :param query:
        :param answer:
        :return: 
        """
        start_entities = self.preprocess_query(query, domain=domain)
        query = query.strip()
        query_domain = sanitize_label(query)
        end_entity = self._link_to_graph(answer, query_domain)

        # 
        paths = [
            self.graph_client.find_shortest_path(
                head_id=ent['id'],
                tail_id=end_entity['id'],
                head_label=query_domain,
                tail_label=query_domain
            ) for ent in start_entities
        ]

        # 
        non_empty_paths = [path for path in paths if path]

        # 
        return [{'relations': path} for path in non_empty_paths]

    def _link_to_graph(self, entity, question_domain):
        query_text = f"{entity}"
        results = self.entities_vector_db.query_collection(
            question_domain,  # 
            query_text
        )
        return results[0].metadata

    def output_log(self, final_depth, current_paths, query, level='info'):
        # 
        logger_func = getattr(logger, level, logger.info)  #  info/debug/warning 
        path_num = len(current_paths)

        if not current_paths:
            logger_func("")
            return
        # 
        rows = [
            (p.format_string, str([round(s, 3) for s in p.scores]), round(p.score, 3))
            for p in current_paths
        ]
        headers = ("Path", "Scores", "Score")
        output = tabulate(
            rows,
            headers=headers,
            tablefmt="pretty",
            colalign=("left", "left", "left")
        )
        logger_func(f"\nDepth: {final_depth}, Path Num: {path_num}")
        logger_func(f"\nQuery: {query}\nCurrent paths:\n{output}\n{'=' * 40}")

    @profile
    def _beam_search(
            self,
            query: str,
            domain: str,
            initial_entities: List[Dict],
            max_depth: int,
            max_width: int,
            prune_zero_score: bool = False  # 0
    ) -> List[Dict]:
        """"""
        current_paths = [Path(e) for e in initial_entities]

        for depth in range(max_depth):
            logger.debug(f"Executing Beam Search Depth: {depth}")
            # 1. 
            finished_paths = [p for p in current_paths if p.is_finished]
            to_expand_paths = [p for p in current_paths if not p.is_finished]

            # 2. 
            cooccur_expanded_paths = []
            fact_expanded_paths = []

            for path in to_expand_paths:
                logger.debug(f"expand from knowledge graph: {domain}")
                kg_expanded_paths = self.expand_from_knowledge_graph(path, domain)
                fact_expanded_paths.extend(kg_expanded_paths)
                if PIPLINE_CONFIG['expand_COG']:
                    logger.debug(f"expand from co-occurrence graph: domain: {domain} | id: {path.current_node['id']}")
                    kg_new_node_ids = {p.current_node['id'] for p in kg_expanded_paths}
                    cooccur_expanded_paths.extend(self.expand_from_cooccurrence_graph(path, domain, exclude_ids=kg_new_node_ids))
                    logger.debug(f"Current co-occurrence paths: {len(cooccur_expanded_paths)}")

                if not fact_expanded_paths and not cooccur_expanded_paths:
                    path.is_finished = True
                    finished_paths.append(path)

            all_paths = []

            # 3. 
            if fact_expanded_paths:
                logger.debug(f"scored expanded fact paths: {len(fact_expanded_paths)}")
                new_paths = list({p.format_string: p for p in fact_expanded_paths}.values())
                scored_fact_expanded_paths = self.ranker.score_and_rank_paths(new_paths, query)
                all_paths += scored_fact_expanded_paths

                # 
                path_saved = [(query, path) for query, path in
                                       zip([query] * len(scored_fact_expanded_paths), [p.to_dict() for p in scored_fact_expanded_paths])]
                for p in path_saved:
                    write_to_file(json.dumps(p))


            if PIPLINE_CONFIG['expand_COG'] and cooccur_expanded_paths:
                logger.debug(f"scored expanded co-occur paths: {len(cooccur_expanded_paths)}")
                new_paths = list({p.format_string: p for p in cooccur_expanded_paths}.values())
                scored_cooccur_expanded_paths = self.ranker.score_and_rank_paths(new_paths, query)
                cooccur_paths = scored_cooccur_expanded_paths[:40]
                logger.debug(f"covering co-occurrence to rel: {len(cooccur_paths)}")
                paths = [p for p in self.cover_co_occurrence_to_rel(cooccur_paths, query=query, remove_none=False) if "none" not in p.format_string]
                all_paths += paths

            if PIPLINE_CONFIG['two_stage'] and RETRIEVER_CONFIG['rank_strategy'] == "trained_ranker":
                all_paths = [p for p in self.ranker.score_and_rank_paths(all_paths, query, strategy='llm')]

            # 
            current_paths = sorted(all_paths, key=lambda p: p.score, reverse=True)[:max_width]

            # 5. 
            self.output_log(depth + 1, current_paths, query, level='debug')

            # 6. 
            if self.sufficiency_check and self._eval_sufficiency(current_paths, query):
                logger.debug("")
                break

        # 8. 
        if current_paths:
            final_depth = len(current_paths[0])
        else:
            final_depth = 0

        self.output_log(final_depth, current_paths, query)
        STATISTIC['n_of_path'] += 1
        STATISTIC['average_depth'] = ((STATISTIC['n_of_path'] - 1) * STATISTIC['average_depth'] + final_depth) / \
                                     STATISTIC['n_of_path']
        STATISTIC['path_lens'].append(final_depth)

        return [p.to_dict() for p in current_paths]



    def _expand_domain_edges(self, path: 'Path', domain) -> List['Path']:
        """"""
        try:
            triples = self.graph_client.get_connected_edges_and_nodes(
                path.current_node['id'],
                label=domain
            )
            expanded_paths = [
                path.copy().add_node(triple['target_node'], triple['edge'])
                for triple in triples
            ]
            return expanded_paths
        except Exception as e:
            logger.error(f": {str(e)}")
            raise e

    def _expand_cooccurrence_edges(self, path: 'Path', exclude_ids: set, domain=None) -> List['Path']:
        """"""
        try:
            co_occur_triples = []
            for ent, doc in self.co_occur_graph.query_cooccurrence_entities_with_docs_degree_limit(path.current_node['id'], domain):
                if ent['id'] not in exclude_ids:
                    co_occur_triples.append({
                        'begin': path.current_node,
                        'end': ent,
                        'r': "co-occurrence",
                        'title': doc['title'],
                        "text": doc['text']
                    })

            expanded_paths = [
                path.copy().add_node(triple['end'], triple) for triple in co_occur_triples
            ]
            return expanded_paths
        except Exception as e:
            logger.error(f": {str(e)}")
            return []

    def expand_from_knowledge_graph(self, path: 'Path', domain) -> List['Path']:
        """
        
        
        """
        # 
        visited_ids = {n['id'] for n in path.nodes}

        # 1. 
        new_paths = self._expand_domain_edges(path, domain)

        # 2. 
        valid_paths = [p for p in new_paths if p.current_node['id'] not in visited_ids]

        return valid_paths

    def expand_from_cooccurrence_graph(self, path: 'Path', domain, exclude_ids: Set[str]) -> List['Path']:
        """
        
        

        Args:
            path (Path): 
            domain: 
            exclude_ids (Set[str]): ID
                                     
        """
        # 
        visited_ids = {n['id'] for n in path.nodes}

        # 1. 
        #  `exclude_ids` 
        new_paths = self._expand_cooccurrence_edges(path, exclude_ids, domain)

        # 2. 
        #  exclude_ids visited_ids 
        valid_paths = [p for p in new_paths if p.current_node['id'] not in visited_ids]

        return valid_paths

    def cover_co_occurrence_to_rel(self, paths: List['Path'], query="", remove_none=False) -> List['Path']:
        """
         'co-occurrence'  LLM  ()
        :param paths: 
        :return: 
        """
        CO_OCCURRENCE_RELATION = 'co-occurrence'

        # 1.  co-occurrence 
        cooccur_triples = [
            triple
            for path in paths
            for triple in path.relations
            if triple['r'] == CO_OCCURRENCE_RELATION
        ]
        if not cooccur_triples:
            return paths

        # 2.  triple 
        doc2triples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for triple in cooccur_triples:
            #  'title' 
            if 'title' not in triple:
                raise ValueError(" 'title' ")
            doc2triples[triple['title']].append(triple)

        # 3. LLM
        entity_pair_to_relations = defaultdict(list)
        for title, triples_in_doc in doc2triples.items():
            text = triples_in_doc[0]['text']
            entity_pairs = [
                [triple['begin']['mention'], triple['end']['mention']]
                for triple in triples_in_doc
            ]
            try:
                completed_triples = complete_relations(self.llm, text, entity_pairs, query)
            except Exception as e:
                raise RuntimeError(f"LLM : {title}") from e

            for completed_triple in completed_triples:
                key = (completed_triple['head'], completed_triple['tail'])
                entity_pair_to_relations[key].extend(completed_triple['relations'])

        final_paths = []
        for path in paths:
            # 
            # 
            wip_paths = [deepcopy(path)]

            # 
            for r_index, relation in enumerate(path.relations):
                if relation['r'] != CO_OCCURRENCE_RELATION:
                    # 
                    continue

                #  (wip_paths) 
                key = (relation['begin']['mention'], relation['end']['mention'])
                predicted_rels = entity_pair_to_relations.get(key)

                # 
                if not predicted_rels:
                    if remove_none:
                        # none wip_paths 
                        wip_paths = []
                    else:
                        #  "none"
                        for p in wip_paths:
                            p.relations[r_index]['r'] = "_none"
                    #  'none'
                    break

                # /
                predicted_rels = list(dict.fromkeys(entity_pair_to_relations.get(key)))

                expanded_paths = []
                for p in wip_paths:  # 
                    for i, rel in enumerate(predicted_rels):
                        # 
                        if i == 0:
                            p.relations[r_index]['r'] = "_" + rel
                            expanded_paths.append(p)
                        else:
                            new_path_version = deepcopy(p)
                            new_path_version.relations[r_index]['r'] = "_" + rel
                            expanded_paths.append(new_path_version)

                # 
                wip_paths = expanded_paths

            #  path 
            # 
            final_paths.extend(wip_paths)

        if len(final_paths) != len(paths):
            s1 = [p.format_string for p in final_paths]
            s2 = [p.format_string for p in paths]
            s1, s2

        return final_paths


    def _eval_sufficiency(
            self,
            paths: List['Path'],
            query: str
    ):
        """
        
        :param paths:
        :param query:
        :return:
        """
        res = eval_sufficiency_with_llm(llm=self.llm, query=query, paths=[p.to_dict() for p in paths])
        return res
class Path:
    """
        

        
        - nodes: 
        - relations: 
        - score: 
        - _diversity: 

        
        - add_node: 
        - copy: 
        - to_dict: 
        """

    def __init__(self, start_node: Dict):
        self.nodes = [start_node]
        self.relations = []
        self._diversity = None


        self.is_finished = False
        self.scores = []

    @property
    def score(self) -> float:
        valid_scores = [score for score in self.scores if score != 0.0]
        if valid_scores:
            self._score = sum(valid_scores) / (len(valid_scores) + 1e-8) #+ 0.01 * getattr(self, 'diversity', 0)
        else:
            self._score = 0.0
        return round(self._score, 5)

    @property
    def current_node(self) -> Dict:
        """"""
        return self.nodes[-1]

    @property
    def diversity(self) -> float:
        """"""
        # if self._diversity is None:
        type_counts = {}
        for n in self.nodes:
            t = n.get('type', 'Unknown')
            type_counts[t] = type_counts.get(t, 0) + 1
        self._diversity = 1 - sum((v / len(self.nodes)) ** 2 for v in type_counts.values())
        return self._diversity

    def add_node(self, node: Dict, relation: Optional[Dict] = None) -> 'Path':
        """
                

                :param node: 
                :param relation: 
                :return: Path
                """
        self.nodes.append(node)
        if relation:
            self.relations.append(relation)
        return self

    def pop_node(self) -> 'Path':
        self.nodes.pop()
        self.relations.pop()
        return self

    def copy(self) -> 'Path':
        return copy.deepcopy(self)

    def to_dict(self) -> Dict:

        return {
            "nodes": self.nodes,
            "relations": self.relations,
            "score": round(self.score, 5),
            "diversity": round(self.diversity, 5),
            "format_string": self.format_string,
            'scores': self.scores
        }

    def __len__(self):
        return len(self.nodes)

    @property
    def format_string(self, path_format="link"):
        if path_format == "link":
            format_string = f"{self.nodes[0]['mention']}"
            for relation, node in zip(self.relations, self.nodes[1:]):
                if relation['end']['id'] == node['id']:
                    format_string += f" - {relation['r']} -> {node['mention']}"
                else:
                    format_string += f" <- {relation['r']} - {node['mention']}"

            return format_string
        elif path_format == "triplets":
            triplet_str = []
            for relation, node in zip(self.relations, self.nodes[1:]):
                triplet_str.append(f"({relation['begin']['mention']}, {relation['r']}, {relation['end']['mention']})")
            format_string = '\n'.join(triplet_str)
            return format_string



class TextRetriever:
    def __init__(self):
        """"""
        self.init_config()

    def init_config(self):
        #  redis
        self.redis_client = redis.Redis(**REDIS_CONFIG)
        self.llm = LLMClient(
            endpoints=LLM_CONFIG['endpoints'],
            cache=self.redis_client
        )

        # 
        self.embedding_model = CustomEmbedding(
            api_key=EMBEDDING_CONFIG['api_key'],
            base_url=EMBEDDING_CONFIG['base_url'],
            model_name=EMBEDDING_CONFIG['model_name']
        )

        self.doc_vector_db = MyVectorStore(
            DATASET_CONFIG['doc_vector_store_path'],
            embedding=self.embedding_model
        )

    def retrieve(self,query, domain):
        docs = self.doc_vector_db.query_collection(domain, query, k=4)
        return [d.page_content for d in docs]
