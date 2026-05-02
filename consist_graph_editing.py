from __future__ import annotations

import copy
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from configs import TOKEN_COUNT
from prompts.consist_prompts import answer_support_prompt, pair_conflict_prompt, ras_sufficiency_prompt, seed_selection_prompt


EdgeKey = Tuple[str, str, str]


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def edge_key(relation: Dict[str, Any]) -> EdgeKey:
    begin = relation.get("begin", {})
    end = relation.get("end", {})
    head = begin.get("id") or begin.get("mention")
    tail = end.get("id") or end.get("mention")
    return (_norm(head), _norm(relation.get("r")), _norm(tail))


def serialize_edge(relation: Dict[str, Any]) -> str:
    begin = relation.get("begin", {})
    end = relation.get("end", {})
    return f"<{begin.get('mention', '')} | {relation.get('r', '')} | {end.get('mention', '')}>"


def serialize_edges(edges: Iterable[Dict[str, Any]]) -> str:
    return "\n".join(serialize_edge(edge["relation"]) for edge in edges)


def _record_tokens(counter: Dict[str, Any]) -> None:
    TOKEN_COUNT["input"] += int(counter.get("input_token", 0) or 0)
    TOKEN_COUNT["output"] += int(counter.get("output_token", 0) or 0)


def _edge_set_key(edges: Iterable[Dict[str, Any]]) -> Tuple[EdgeKey, ...]:
    return tuple(sorted(edge["key"] for edge in edges))


def _node_key(node: Dict[str, Any]) -> str:
    return _norm(node.get("id") or node.get("mention") or node.get("name"))


@dataclass
class CONSISTConfig:
    method: str = "consist"
    seed_count: int = 3
    lambda_c: float = 0.3
    lambda_e: float = 0.05
    lambda_comp: float = 0.1
    rollout_h: int = 2
    k_add: int = 3
    k_del: int = 2
    use_gumbel: bool = True
    gumbel_tau: float = 1.0
    support_threshold: float = 0.8
    use_ras_sufficiency_judge: bool = True
    stop_on_no_retrieval: bool = True
    use_llm_seed_selector: bool = False
    random_pruning_ratio: float = 0.0
    record_efficiency: bool = True
    use_llm_pair_judge: bool = False
    use_llm_support_judge: bool = False
    eps: float = 1e-6
    rng_seed: int = 13

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CONSISTConfig":
        llm_config = config.get("llm", {}) or {}
        return cls(
            method=config.get("method", "consist"),
            seed_count=int(config.get("seed_count", 3)),
            lambda_c=float(config.get("lambda_c", 0.3)),
            lambda_e=float(config.get("lambda_e", 0.05)),
            lambda_comp=float(config.get("lambda_comp", 0.1)),
            rollout_h=int(config.get("rollout_h", 2)),
            k_add=int(config.get("k_add", 3)),
            k_del=int(config.get("k_del", 2)),
            use_gumbel=bool(config.get("use_gumbel", True)),
            gumbel_tau=float(config.get("gumbel_tau", 1.0)),
            support_threshold=float(config.get("support_threshold", 0.8)),
            use_ras_sufficiency_judge=bool(config.get("use_ras_sufficiency_judge", True)),
            stop_on_no_retrieval=bool(config.get("stop_on_no_retrieval", True)),
            use_llm_seed_selector=bool(config.get("use_llm_seed_selector", False)),
            random_pruning_ratio=float(config.get("random_pruning_ratio", 0.0) or 0.0),
            record_efficiency=bool(config.get("record_efficiency", True)),
            use_llm_pair_judge=bool(llm_config.get("use_llm_pair_judge", False)),
            use_llm_support_judge=bool(llm_config.get("use_llm_support_judge", False)),
            rng_seed=int(config.get("rng_seed", 13)),
        )


@dataclass
class CONSISTRunStats:
    graph_size: int = 0
    conflict_pairs: int = 0
    latency_seconds: float = 0.0
    candidate_edges: int = 0
    edit_steps: int = 0
    selected_add_edges: int = 0
    selected_del_edges: int = 0
    utility: float = 0.0
    support: float = 0.0
    comp_seed: int = 0
    utility_breakdown: Dict[str, Any] = field(default_factory=dict)
    seed_selection_method: str = ""
    seed_entities: List[str] = field(default_factory=list)
    frontier_nodes: List[str] = field(default_factory=list)
    stopped_by_sufficiency: bool = False
    stop_reason: str = "max_steps"
    sufficiency_label: str = ""
    seeds_connected: bool = False
    seed_components: List[List[str]] = field(default_factory=list)
    edit_trace: List[Dict[str, Any]] = field(default_factory=list)


class CONSISTGraphEditor:
    """Minimal evidence-subgraph editor used by consist and ablation modes.

    The editor operates on retrieved BaseGraphRAG paths, not on the persistent Neo4j
    database. This keeps baseline BaseGraphRAG reproducible while allowing controlled
    add/delete pruning before final answer generation.
    """

    def __init__(self, config: CONSISTConfig, llm: Optional[Any] = None):
        self.config = config
        self.llm = llm
        self.rng = random.Random(config.rng_seed)
        self._support_cache: Dict[Tuple[str, Tuple[EdgeKey, ...]], float] = {}
        self._conflict_cache: Dict[Tuple[str, Tuple[EdgeKey, ...]], float] = {}
        self._ras_cache: Dict[Tuple[str, Tuple[EdgeKey, ...]], Tuple[bool, str, str]] = {}

    def run(self, paths: List[Dict[str, Any]], query: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start = time.time()
        self._support_cache = {}
        self._conflict_cache = {}
        self._ras_cache = {}
        if not paths:
            return [], CONSISTRunStats(
                latency_seconds=time.time() - start,
                stop_reason="no_paths",
            ).__dict__

        all_edges = self._extract_edges(paths)
        seed_keys, seed_selection_method = self._initial_seed_keys(paths, all_edges, query)
        seed_entities = self._seed_entities(seed_keys)
        current_keys = set(seed_keys)
        edit_steps = 0
        selected_add_edges = 0
        selected_del_edges = 0
        edit_trace: List[Dict[str, Any]] = []
        stop_reason = "max_steps"
        sufficiency_label = ""
        stopped_by_sufficiency = False
        edge_by_key = {edge["key"]: edge for edge in all_edges}

        max_steps = max(1, self.config.rollout_h)
        for step_index in range(max_steps):
            stop_now, label, reason = self._should_stop(
                self._edges_for_keys(all_edges, current_keys),
                query,
            )
            if stop_now:
                stop_reason = reason
                sufficiency_label = label
                stopped_by_sufficiency = True
                break

            before_edges = self._edges_for_keys(all_edges, current_keys)
            before_breakdown = self.utility_breakdown(before_edges, query, seed_keys)
            before_utility = before_breakdown["utility"]
            add_edits, del_edits = self._select_edits(current_keys, all_edges, seed_keys, query)
            if not add_edits and not del_edits:
                stop_reason = "no_valid_edits"
                break

            proposed_keys = set(current_keys)
            for edit in add_edits:
                proposed_keys.add(edit)
            for edit in del_edits:
                proposed_keys.discard(edit)

            after_edges = self._edges_for_keys(all_edges, proposed_keys)
            after_breakdown = self.utility_breakdown(after_edges, query, seed_keys)
            after_utility = after_breakdown["utility"]
            accepted = after_utility > before_utility + self.config.eps
            if accepted:
                current_keys = proposed_keys
                selected_add_edges += len(add_edits)
                selected_del_edges += len(del_edits)
                edit_steps += 1
            else:
                stop_reason = "non_improving_edit"

            edit_trace.append({
                "step": step_index,
                "accepted": accepted,
                "before_utility": before_utility,
                "after_utility": after_utility,
                "before_breakdown": before_breakdown,
                "after_breakdown": after_breakdown,
                "added_edges": [
                    serialize_edge(edge_by_key[key]["relation"])
                    for key in add_edits
                    if key in edge_by_key
                ],
                "deleted_edges": [
                    serialize_edge(edge_by_key[key]["relation"])
                    for key in del_edits
                    if key in edge_by_key
                ],
            })
            if not accepted:
                break
            stop_now, label, reason = self._should_stop(
                self._edges_for_keys(all_edges, current_keys),
                query,
            )
            if stop_now:
                stop_reason = reason
                sufficiency_label = label
                stopped_by_sufficiency = True
                break

        current_keys = self._apply_random_pruning(current_keys)
        final_edges = [edge for edge in all_edges if edge["key"] in current_keys]
        frontier_nodes = self._frontier_nodes(final_edges)
        conflict_pairs = self.build_conflict_pairs(final_edges, query)
        support = self.answer_support(final_edges, query)
        comp_seed = self.comp_seed(final_edges, seed_keys)
        component_summary = self.seed_component_summary(final_edges, seed_keys)
        utility = self.utility(final_edges, query, seed_keys)
        utility_breakdown = self.utility_breakdown(final_edges, query, seed_keys)
        edited_paths = self._paths_from_keys(paths, current_keys)

        stats = CONSISTRunStats(
            graph_size=len(final_edges),
            conflict_pairs=len(conflict_pairs),
            latency_seconds=time.time() - start,
            candidate_edges=len(all_edges),
            edit_steps=edit_steps,
            selected_add_edges=selected_add_edges,
            selected_del_edges=selected_del_edges,
            utility=utility,
            support=support,
            comp_seed=comp_seed,
            utility_breakdown=utility_breakdown,
            seed_selection_method=seed_selection_method,
            seed_entities=seed_entities,
            frontier_nodes=frontier_nodes,
            stopped_by_sufficiency=stopped_by_sufficiency,
            stop_reason=stop_reason,
            sufficiency_label=sufficiency_label,
            seeds_connected=component_summary["seeds_connected"],
            seed_components=component_summary["seed_components"],
            edit_trace=edit_trace,
        )
        return edited_paths, stats.__dict__

    def _extract_edges(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        edges: Dict[EdgeKey, Dict[str, Any]] = {}
        for path_index, path in enumerate(paths):
            path_score = float(path.get("score", 0.0) or 0.0)
            scores = path.get("scores") or []
            for edge_index, relation in enumerate(path.get("relations", [])):
                key = edge_key(relation)
                edge_score = float(scores[edge_index]) if edge_index < len(scores) else path_score
                if key not in edges or edge_score > edges[key]["score"]:
                    edges[key] = {
                        "key": key,
                        "relation": relation,
                        "path_index": path_index,
                        "edge_index": edge_index,
                        "score": edge_score,
                    }
        return sorted(edges.values(), key=lambda item: item["score"], reverse=True)

    def _initial_seed_keys(self, paths: List[Dict[str, Any]], edges: List[Dict[str, Any]], query: str) -> Tuple[Set[EdgeKey], str]:
        if self.config.use_llm_seed_selector and self.llm is not None and edges:
            try:
                candidate_nodes = self._candidate_seed_nodes(edges)
                selected_nodes = self._select_seed_nodes_with_llm(candidate_nodes, query)
                seed_keys = self._seed_keys_from_selected_nodes(edges, selected_nodes)
                if seed_keys:
                    return seed_keys, "llm_seed_selector"
            except Exception:
                pass

        seed_keys: Set[EdgeKey] = set()
        seed_path_count = max(1, self.config.seed_count)
        for path in paths[:seed_path_count]:
            for relation in path.get("relations", []):
                seed_keys.add(edge_key(relation))
        if not seed_keys:
            seed_keys.update(edge["key"] for edge in edges[:seed_path_count])
        return seed_keys, "top_paths_fallback"

    def _candidate_seed_nodes(self, edges: List[Dict[str, Any]], max_candidates: int = 30) -> List[Dict[str, Any]]:
        candidates: Dict[str, Dict[str, Any]] = {}
        for edge in edges:
            relation = edge["relation"]
            score = float(edge.get("score", 0.0) or 0.0)
            for role in ("begin", "end"):
                node = relation.get(role, {}) or {}
                key = _node_key(node)
                if not key:
                    continue
                existing = candidates.get(key)
                if existing is None or score > existing["score"]:
                    candidates[key] = {
                        "node_key": key,
                        "mention": node.get("mention") or node.get("name") or node.get("id") or "",
                        "description": node.get("description", ""),
                        "score": score,
                    }
        ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
        for index, candidate in enumerate(ranked[:max_candidates], start=1):
            candidate["seed_id"] = f"S{index}"
        return ranked[:max_candidates]

    def _select_seed_nodes_with_llm(self, candidates: List[Dict[str, Any]], query: str) -> Set[str]:
        if not candidates:
            return set()
        lines = [
            f"{candidate['seed_id']} | {candidate['mention']} | {candidate.get('description', '')}"
            for candidate in candidates
        ]
        prompt = seed_selection_prompt(query, "\n".join(lines), self.config.seed_count)
        token_counter = {"input_token": 0, "output_token": 0}
        raw = self.llm.call_llm(
            prompt,
            request_overrides={"temperature": 0},
            token_counter=token_counter,
        )
        _record_tokens(token_counter)
        valid_ids = {candidate["seed_id"]: candidate["node_key"] for candidate in candidates}
        selected_ids = re.findall(r"\bS\d+\b", str(raw or ""))
        selected_nodes: List[str] = []
        for seed_id in selected_ids:
            node_key = valid_ids.get(seed_id)
            if node_key and node_key not in selected_nodes:
                selected_nodes.append(node_key)
        return set(selected_nodes[: self.config.seed_count])

    def _seed_keys_from_selected_nodes(self, edges: List[Dict[str, Any]], selected_nodes: Set[str]) -> Set[EdgeKey]:
        seed_keys: Set[EdgeKey] = set()
        if not selected_nodes:
            return seed_keys
        covered_nodes: Set[str] = set()
        for edge in edges:
            head, _, tail = edge["key"]
            touched = {head, tail} & selected_nodes
            new_touched = touched - covered_nodes
            if new_touched:
                seed_keys.add(edge["key"])
                covered_nodes.update(new_touched)
            if len(covered_nodes) >= min(self.config.seed_count, len(selected_nodes)):
                break
        return seed_keys

    def _seed_entities(self, seed_keys: Set[EdgeKey]) -> List[str]:
        return sorted({node for key in seed_keys for node in (key[0], key[2]) if node})

    def _frontier_nodes(self, edges: List[Dict[str, Any]]) -> List[str]:
        heads = {edge["key"][0] for edge in edges}
        tails = {edge["key"][2] for edge in edges}
        frontier = tails - heads
        if not frontier:
            frontier = tails
        return sorted(node for node in frontier if node)

    def _select_edits(
        self,
        current_keys: Set[EdgeKey],
        all_edges: List[Dict[str, Any]],
        seed_keys: Set[EdgeKey],
        query: str,
    ) -> Tuple[List[EdgeKey], List[EdgeKey]]:
        add_candidates = [edge["key"] for edge in all_edges if edge["key"] not in current_keys]
        del_candidates = [] if self.config.method == "add_only" else [edge["key"] for edge in all_edges if edge["key"] in current_keys]
        add_scored = self._rank_edit_candidates(add_candidates, current_keys, all_edges, seed_keys, query, edit_type="add")
        del_scored = self._rank_edit_candidates(del_candidates, current_keys, all_edges, seed_keys, query, edit_type="del")
        add_selected = self._select_topk(add_scored, self.config.k_add)
        del_selected = self._select_topk(del_scored, self.config.k_del)
        return add_selected, del_selected

    def _select_topk(self, scored: List[Tuple[EdgeKey, float]], k: int) -> List[EdgeKey]:
        finite = [(key, score) for key, score in scored if score > float("-inf")]
        if k <= 0 or not finite:
            return []
        if not self.config.use_gumbel:
            return [key for key, _ in finite[:k]]
        tau = max(self.config.gumbel_tau, self.config.eps)
        noisy = [
            (key, (score + self._sample_gumbel()) / tau)
            for key, score in finite
        ]
        noisy.sort(key=lambda item: item[1], reverse=True)
        return [key for key, _ in noisy[:k]]

    def _sample_gumbel(self) -> float:
        u = min(max(self.rng.random(), self.config.eps), 1.0 - self.config.eps)
        return -math.log(-math.log(u))

    def _rank_edit_candidates(
        self,
        candidates: List[EdgeKey],
        current_keys: Set[EdgeKey],
        all_edges: List[Dict[str, Any]],
        seed_keys: Set[EdgeKey],
        query: str,
        edit_type: str,
    ) -> List[Tuple[EdgeKey, float]]:
        current_edges = self._edges_for_keys(all_edges, current_keys)
        d_t = self.utility(current_edges, query, seed_keys)
        scored: List[Tuple[EdgeKey, float]] = []
        for key in candidates:
            next_keys = set(current_keys)
            if edit_type == "add":
                next_keys.add(key)
            else:
                next_keys.discard(key)
            next_edges = self._edges_for_keys(all_edges, next_keys)
            u_now = self.utility(next_edges, query, seed_keys)
            if self.config.method == "w_o_nash":
                score = u_now
            else:
                fut_keys = self._rollout(next_keys, all_edges, seed_keys, query)
                fut_edges = self._edges_for_keys(all_edges, fut_keys)
                u_fut = self.utility(fut_edges, query, seed_keys)
                now_delta = u_now - d_t
                fut_delta = u_fut - d_t
                if now_delta <= self.config.eps or fut_delta <= self.config.eps:
                    score = float("-inf")
                else:
                    score = math.log(now_delta) + math.log(fut_delta)
            scored.append((key, score))
        return sorted(scored, key=lambda item: item[1], reverse=True)

    def _rollout(
        self,
        keys: Set[EdgeKey],
        all_edges: List[Dict[str, Any]],
        seed_keys: Set[EdgeKey],
        query: str,
    ) -> Set[EdgeKey]:
        rollout_keys = set(keys)
        for _ in range(max(0, self.config.rollout_h - 1)):
            add_candidates = [edge["key"] for edge in all_edges if edge["key"] not in rollout_keys]
            if not add_candidates:
                break
            scored = []
            for key in add_candidates:
                trial_keys = set(rollout_keys)
                trial_keys.add(key)
                trial_edges = self._edges_for_keys(all_edges, trial_keys)
                scored.append((key, self.utility(trial_edges, query, seed_keys)))
            scored.sort(key=lambda item: item[1], reverse=True)
            current_u = self.utility(self._edges_for_keys(all_edges, rollout_keys), query, seed_keys)
            if not scored or scored[0][1] <= current_u:
                break
            rollout_keys.add(scored[0][0])
        return rollout_keys

    def utility(self, edges: List[Dict[str, Any]], query: str, seed_keys: Set[EdgeKey]) -> float:
        return self.utility_breakdown(edges, query, seed_keys)["utility"]

    def utility_breakdown(self, edges: List[Dict[str, Any]], query: str, seed_keys: Set[EdgeKey]) -> Dict[str, Any]:
        support = self.answer_support(edges, query)
        conflict = 0.0 if self.config.method == "w_o_conflict" else self.conflict_score(edges, query)
        edge_penalty = self.config.lambda_e * len(edges)
        component_count = self.comp_seed(edges, seed_keys)
        conflict_penalty = self.config.lambda_c * conflict
        comp_penalty = self.config.lambda_comp * component_count
        return {
            "support": support,
            "conflict_score": conflict,
            "conflict_penalty": conflict_penalty,
            "edge_count": len(edges),
            "edge_penalty": edge_penalty,
            "comp_seed": component_count,
            "comp_penalty": comp_penalty,
            "utility": support - conflict_penalty - edge_penalty - comp_penalty,
        }

    def _should_stop(self, edges: List[Dict[str, Any]], query: str) -> Tuple[bool, str, str]:
        if not edges:
            return False, "", "empty_graph"
        cache_key = (query, _edge_set_key(edges))
        if cache_key in self._ras_cache:
            return self._ras_cache[cache_key]

        if self.config.use_ras_sufficiency_judge and self.llm is not None:
            prompt = ras_sufficiency_prompt(query, serialize_edges(edges))
            try:
                token_counter = {"input_token": 0, "output_token": 0}
                raw = self.llm.call_llm(
                    prompt,
                    request_overrides={"temperature": 0},
                    token_counter=token_counter,
                )
                _record_tokens(token_counter)
                label = self._parse_ras_label(raw)
                if label == "SUFFICIENT":
                    decision = (True, label, "ras_sufficient")
                elif label == "NO_RETRIEVAL" and self.config.stop_on_no_retrieval:
                    decision = (True, label, "ras_no_retrieval")
                else:
                    decision = (False, label, "ras_need_more")
                self._ras_cache[cache_key] = decision
                return decision
            except Exception:
                pass

        support = self.answer_support(edges, query)
        if support >= self.config.support_threshold:
            decision = (True, "THRESHOLD", "support_threshold")
        else:
            decision = (False, "THRESHOLD", "below_support_threshold")
        self._ras_cache[cache_key] = decision
        return decision

    def answer_support(self, edges: List[Dict[str, Any]], query: str) -> float:
        if not edges:
            return 0.0
        cache_key = (query, _edge_set_key(edges))
        if cache_key in self._support_cache:
            return self._support_cache[cache_key]
        if self.config.use_llm_support_judge and self.llm is not None:
            prompt = answer_support_prompt(query, serialize_edges(edges))
            try:
                token_counter = {"input_token": 0, "output_token": 0}
                raw = self.llm.call_llm(
                    prompt,
                    request_overrides={"temperature": 0},
                    token_counter=token_counter,
                )
                _record_tokens(token_counter)
                score = self._parse_score(raw, default=0.0)
                self._support_cache[cache_key] = score
                return score
            except Exception:
                pass
        scores = [float(edge.get("score", 0.0) or 0.0) for edge in edges]
        score = max(scores)
        self._support_cache[cache_key] = score
        return score

    def build_conflict_pairs(self, edges: List[Dict[str, Any]], query: str) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for i, edge_i in enumerate(edges):
            rel_i = edge_i["relation"]
            head_i = _norm(rel_i.get("begin", {}).get("id") or rel_i.get("begin", {}).get("mention"))
            relation_i = _norm(rel_i.get("r"))
            tail_i = _norm(rel_i.get("end", {}).get("id") or rel_i.get("end", {}).get("mention"))
            for edge_j in edges[i + 1:]:
                rel_j = edge_j["relation"]
                same_head = head_i == _norm(rel_j.get("begin", {}).get("id") or rel_j.get("begin", {}).get("mention"))
                same_slot = relation_i == _norm(rel_j.get("r"))
                different_tail = tail_i != _norm(rel_j.get("end", {}).get("id") or rel_j.get("end", {}).get("mention"))
                if same_head and same_slot and different_tail:
                    pairs.append((edge_i, edge_j))
        return pairs

    def conflict_score(self, edges: List[Dict[str, Any]], query: str) -> float:
        cache_key = (query, _edge_set_key(edges))
        if cache_key in self._conflict_cache:
            return self._conflict_cache[cache_key]
        pairs = self.build_conflict_pairs(edges, query)
        if not pairs:
            self._conflict_cache[cache_key] = 0.0
            return 0.0
        if not self.config.use_llm_pair_judge or self.llm is None:
            score = float(len(pairs))
            self._conflict_cache[cache_key] = score
            return score
        score = 0.0
        for edge_i, edge_j in pairs:
            prompt = pair_conflict_prompt(query, serialize_edge(edge_i["relation"]), serialize_edge(edge_j["relation"]))
            try:
                token_counter = {"input_token": 0, "output_token": 0}
                raw = self.llm.call_llm(
                    prompt,
                    request_overrides={"temperature": 0},
                    token_counter=token_counter,
                )
                _record_tokens(token_counter)
                score += self._parse_score(raw, default=1.0)
            except Exception:
                score += 1.0
        self._conflict_cache[cache_key] = score
        return score

    def _parse_score(self, raw: Any, default: float) -> float:
        match = re.search(r"-?\d+(?:\.\d+)?", str(raw))
        if not match:
            return default
        return max(0.0, min(1.0, float(match.group(0))))

    def _parse_ras_label(self, raw: Any) -> str:
        text = str(raw or "").upper()
        match = re.search(r"\[(NO_RETRIEVAL|SUFFICIENT|SUBQ)\]", text)
        if match:
            return match.group(1)
        if "NO_RETRIEVAL" in text:
            return "NO_RETRIEVAL"
        if "SUFFICIENT" in text:
            return "SUFFICIENT"
        if "SUBQ" in text:
            return "SUBQ"
        return "UNKNOWN"

    def comp_seed(self, edges: List[Dict[str, Any]], seed_keys: Set[EdgeKey]) -> int:
        return self.seed_component_summary(edges, seed_keys)["component_count"]

    def seed_component_summary(self, edges: List[Dict[str, Any]], seed_keys: Set[EdgeKey]) -> Dict[str, Any]:
        parent: Dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: str, b: str) -> None:
            parent[find(a)] = find(b)

        seed_nodes = {node for key in seed_keys for node in (key[0], key[2])}
        for edge in edges:
            head, _, tail = edge["key"]
            union(head, tail)
        grouped: Dict[str, List[str]] = {}
        for node in seed_nodes:
            if node in parent:
                grouped.setdefault(find(node), []).append(node)
            else:
                grouped.setdefault(node, []).append(node)
        seed_components = [sorted(nodes) for nodes in grouped.values()]
        return {
            "component_count": len(seed_components),
            "seeds_connected": len(seed_components) <= 1,
            "seed_components": seed_components,
        }

    def _apply_random_pruning(self, keys: Set[EdgeKey]) -> Set[EdgeKey]:
        if self.config.method != "random_pruning" or self.config.random_pruning_ratio <= 0:
            return keys
        keep_keys = list(keys)
        prune_count = int(len(keep_keys) * self.config.random_pruning_ratio)
        if prune_count <= 0:
            return keys
        to_remove = set(self.rng.sample(keep_keys, min(prune_count, len(keep_keys))))
        return {key for key in keys if key not in to_remove}

    def _edges_for_keys(self, all_edges: List[Dict[str, Any]], keys: Set[EdgeKey]) -> List[Dict[str, Any]]:
        return [edge for edge in all_edges if edge["key"] in keys]

    def _paths_from_keys(self, paths: List[Dict[str, Any]], keys: Set[EdgeKey]) -> List[Dict[str, Any]]:
        selected_paths: List[Dict[str, Any]] = []
        for path in paths:
            relations = [relation for relation in path.get("relations", []) if edge_key(relation) in keys]
            if not relations:
                continue
            edited_path = copy.deepcopy(path)
            edited_path["relations"] = relations
            edited_path["score"] = float(path.get("score", 0.0) or 0.0)
            edited_path["format_string"] = " | ".join(serialize_edge(relation) for relation in relations)
            selected_paths.append(edited_path)
        return selected_paths
