from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from tqdm import tqdm

import configs
from core.generator import MyGenerator
from core.consist_graph_editing import CONSISTGraphEditor, CONSISTConfig
from core.retriever import GraphRetriever
from evaluation.hotpot_evaluate_v1 import eval as hotpot_eval
from evaluation.hotpot_evaluate_v1 import exact_match_score
from utils.neo4j_operator import sanitize_label


CONSIST_MODES = {"consist", "w_o_conflict", "w_o_nash", "add_only", "random_pruning"}


def _canonical_dataset_name() -> str:
    configured = configs.DATASET_CONFIG["dataset_name"]
    return configs.DATASET_RESULT_ALIASES.get(configured, configured).lower()


def _effective_method_and_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(config)
    mode = configs.EXPERIMENT_MODE
    if mode.startswith("random_pruning_"):
        ratio_text = mode.replace("random_pruning_", "")
        try:
            merged["random_pruning_ratio"] = float(ratio_text) / 100.0
        except ValueError:
            merged["random_pruning_ratio"] = merged.get("random_pruning_ratio", 0.1)
        merged["method"] = "random_pruning"
        merged["method_label"] = mode
    elif mode != "baseline_graphrag":
        merged["method"] = mode
        merged["method_label"] = mode
    else:
        merged.setdefault("method_label", merged.get("method", "consist"))
    return merged


def _param_string(config: Dict[str, Any]) -> str:
    method = config.get("method_label", config.get("method", "consist"))
    dataset = _canonical_dataset_name()
    return (
        f"{method}_{dataset}"
        f"_m{config.get('seed_count', 3)}"
        f"_lc{config.get('lambda_c', 0.3)}"
        f"_le{config.get('lambda_e', 0.05)}"
        f"_lcomp{config.get('lambda_comp', 0.1)}"
        f"_h{config.get('rollout_h', 2)}"
        f"_ka{config.get('k_add', 3)}"
        f"_kd{config.get('k_del', 2)}"
        f"_tau{config.get('gumbel_tau', 1.0)}"
        f"_st{config.get('support_threshold', 0.8)}"
    )


def _create_run_dir(config: Dict[str, Any]) -> Path:
    output_root = Path(config.get("output_root") or "results")
    dataset_root = output_root / _canonical_dataset_name()
    dataset_root.mkdir(parents=True, exist_ok=True)
    prefix = _param_string(config)
    index = 1
    while True:
        run_dir = dataset_root / f"{prefix}_{index:03d}"
        if not run_dir.exists():
            run_dir.mkdir(parents=True)
            return run_dir
        index += 1


def _save_json(data: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _append_jsonl(data: Dict[str, Any], path: Path) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def _token_snapshot() -> Dict[str, int]:
    return {
        "input": int(configs.TOKEN_COUNT.get("input", 0) or 0),
        "output": int(configs.TOKEN_COUNT.get("output", 0) or 0),
    }


def _token_delta(before: Dict[str, int]) -> Dict[str, int]:
    current = _token_snapshot()
    input_tokens = max(0, current["input"] - before["input"])
    output_tokens = max(0, current["output"] - before["output"])
    return {
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_total_tokens": input_tokens + output_tokens,
    }


def _path_record(result: Dict[str, Any]) -> Dict[str, Any]:
    efficiency = result.get("efficiency", {}) or {}
    return {
        "id": result.get("id"),
        "question": result.get("question", ""),
        "answer": result.get("answer", ""),
        "output": result.get("output", ""),
        "retrieved_path_count": result.get("retrieved_path_count", 0),
        "final_path_count": result.get("final_path_count", 0),
        "context": result.get("context", []),
        "edit_trace": efficiency.get("edit_trace", []),
        "seed_components": efficiency.get("seed_components", []),
        "seed_selection_method": efficiency.get("seed_selection_method", ""),
        "seed_entities": efficiency.get("seed_entities", []),
        "frontier_nodes": efficiency.get("frontier_nodes", []),
        "seeds_connected": efficiency.get("seeds_connected", False),
        "stop_reason": efficiency.get("stop_reason", ""),
        "sufficiency_label": efficiency.get("sufficiency_label", ""),
    }


def _edit_record(result: Dict[str, Any]) -> Dict[str, Any]:
    efficiency = result.get("efficiency", {}) or {}
    return {
        "id": result.get("id"),
        "question": result.get("question", ""),
        "retrieved_path_count": result.get("retrieved_path_count", 0),
        "final_path_count": result.get("final_path_count", 0),
        "graph_size": efficiency.get("graph_size", 0),
        "candidate_edges": efficiency.get("candidate_edges", 0),
        "edit_steps": efficiency.get("edit_steps", 0),
        "selected_add_edges": efficiency.get("selected_add_edges", 0),
        "selected_del_edges": efficiency.get("selected_del_edges", 0),
        "support": efficiency.get("support", 0.0),
        "utility": efficiency.get("utility", 0.0),
        "utility_breakdown": efficiency.get("utility_breakdown", {}),
        "conflict_pairs": efficiency.get("conflict_pairs", 0),
        "comp_seed": efficiency.get("comp_seed", 0),
        "seeds_connected": efficiency.get("seeds_connected", False),
        "seed_components": efficiency.get("seed_components", []),
        "seed_selection_method": efficiency.get("seed_selection_method", ""),
        "seed_entities": efficiency.get("seed_entities", []),
        "frontier_nodes": efficiency.get("frontier_nodes", []),
        "stopped_by_sufficiency": efficiency.get("stopped_by_sufficiency", False),
        "stop_reason": efficiency.get("stop_reason", ""),
        "sufficiency_label": efficiency.get("sufficiency_label", ""),
        "edit_trace": efficiency.get("edit_trace", []),
    }


def _edge_edit_records(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    efficiency = result.get("efficiency", {}) or {}
    records: List[Dict[str, Any]] = []
    edit_trace = efficiency.get("edit_trace", []) or []
    if not edit_trace:
        records.append({
            "id": result.get("id"),
            "question": result.get("question", ""),
            "step": None,
            "has_edit": False,
            "accepted": False,
            "added_edges": [],
            "deleted_edges": [],
            "before_utility": None,
            "after_utility": None,
            "before_breakdown": {},
            "after_breakdown": {},
            "stop_reason": efficiency.get("stop_reason", ""),
            "sufficiency_label": efficiency.get("sufficiency_label", ""),
        })
        return records

    for step in edit_trace:
        added_edges = step.get("added_edges", []) or []
        deleted_edges = step.get("deleted_edges", []) or []
        records.append({
            "id": result.get("id"),
            "question": result.get("question", ""),
            "step": step.get("step"),
            "accepted": bool(step.get("accepted", True)),
            "has_edit": bool(added_edges or deleted_edges),
            "added_edges": added_edges,
            "deleted_edges": deleted_edges,
            "before_utility": step.get("before_utility"),
            "after_utility": step.get("after_utility"),
            "before_breakdown": step.get("before_breakdown", {}),
            "after_breakdown": step.get("after_breakdown", {}),
            "stop_reason": efficiency.get("stop_reason", ""),
            "sufficiency_label": efficiency.get("sufficiency_label", ""),
        })
    return records


def _log(message: str, run_dir: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    with open(run_dir / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _load_samples(max_samples: int) -> List[Dict[str, Any]]:
    dataset_file = Path(configs.DATASET_CONFIG["dataset_file"])
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_file}. "
            f"Current dataset alias resolved to {configs.DATASET_CONFIG['dataset_name']}."
        )
    with open(dataset_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if max_samples is not None and max_samples >= 0:
        samples = samples[:max_samples]
    return samples


def evaluate_predictions(predictions: List[Dict[str, Any]], dataset_name: str, method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    preds = {i: item.get("output", "") for i, item in enumerate(predictions)}
    golds = [{"_id": i, "answer": item.get("answer", "")} for i, item in enumerate(predictions)]
    metrics = hotpot_eval({"answer": preds}, golds) if predictions else {"em": 0.0, "f1": 0.0, "prec": 0.0, "recall": 0.0}
    exact = [exact_match_score(item.get("output", ""), item.get("answer", "")) for item in predictions]
    metrics["accuracy"] = sum(exact) / len(exact) if exact else 0.0
    metrics.update({
        "dataset": dataset_name,
        "method": method,
        "ablation": method if method != "consist" else "none",
        "sample_count": len(predictions),
        "seed_count": config.get("seed_count"),
        "use_llm_seed_selector": config.get("use_llm_seed_selector", False),
        "lambda_c": config.get("lambda_c"),
        "lambda_e": config.get("lambda_e"),
        "lambda_comp": config.get("lambda_comp"),
        "rollout_h": config.get("rollout_h"),
        "k_add": config.get("k_add"),
        "k_del": config.get("k_del"),
        "use_gumbel": config.get("use_gumbel", True),
        "gumbel_tau": config.get("gumbel_tau", 1.0),
        "support_threshold": config.get("support_threshold", 0.8),
        "use_ras_sufficiency_judge": config.get("use_ras_sufficiency_judge", True),
        "stop_on_no_retrieval": config.get("stop_on_no_retrieval", True),
        "random_pruning_ratio": config.get("random_pruning_ratio", 0.0),
    })
    return metrics


def aggregate_efficiency(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        return {
            "avg_graph_size": 0.0,
            "avg_latency_seconds": 0.0,
            "avg_conflict_pairs": 0.0,
            "avg_llm_input_tokens": 0.0,
            "avg_llm_output_tokens": 0.0,
            "avg_llm_total_tokens": 0.0,
            "total_llm_input_tokens": 0,
            "total_llm_output_tokens": 0,
            "total_llm_tokens": 0,
            "total_latency_seconds": 0.0,
            "tokens_per_second": 0.0,
            "sufficiency_stop_rate": 0.0,
            "avg_support": 0.0,
            "stop_reason_counts": {},
            "sample_count": 0,
        }
    total_latency = sum(item.get("latency_seconds", 0.0) for item in items)
    total_input_tokens = sum(int(item.get("llm_input_tokens", 0) or 0) for item in items)
    total_output_tokens = sum(int(item.get("llm_output_tokens", 0) or 0) for item in items)
    total_tokens = total_input_tokens + total_output_tokens
    stop_reason_counts: Dict[str, int] = {}
    for item in items:
        reason = str(item.get("stop_reason", "unknown") or "unknown")
        stop_reason_counts[reason] = stop_reason_counts.get(reason, 0) + 1
    return {
        "avg_graph_size": sum(item.get("graph_size", 0) for item in items) / len(items),
        "avg_latency_seconds": total_latency / len(items),
        "avg_conflict_pairs": sum(item.get("conflict_pairs", 0) for item in items) / len(items),
        "avg_candidate_edges": sum(item.get("candidate_edges", 0) for item in items) / len(items),
        "avg_edit_steps": sum(item.get("edit_steps", 0) for item in items) / len(items),
        "avg_comp_seed": sum(item.get("comp_seed", 0) for item in items) / len(items),
        "avg_support": sum(float(item.get("support", 0.0) or 0.0) for item in items) / len(items),
        "all_seeds_connected_rate": sum(1 for item in items if item.get("seeds_connected")) / len(items),
        "sufficiency_stop_rate": sum(1 for item in items if item.get("stopped_by_sufficiency")) / len(items),
        "stop_reason_counts": stop_reason_counts,
        "avg_llm_input_tokens": total_input_tokens / len(items),
        "avg_llm_output_tokens": total_output_tokens / len(items),
        "avg_llm_total_tokens": total_tokens / len(items),
        "total_llm_input_tokens": total_input_tokens,
        "total_llm_output_tokens": total_output_tokens,
        "total_llm_tokens": total_tokens,
        "total_latency_seconds": total_latency,
        "tokens_per_second": total_tokens / total_latency if total_latency > 0 else 0.0,
        "sample_count": len(items),
    }


def process_sample(
    sample: Dict[str, Any],
    step: int,
    retriever: GraphRetriever,
    generator: MyGenerator,
    editor: CONSISTGraphEditor,
) -> Dict[str, Any]:
    question = sample["question"]
    domain = (
        sanitize_label(question)
        if configs.DATASET_CONFIG["query_setting"] == "local"
        else configs.DATASET_CONFIG["domain"]
    )
    start = time.time()
    token_before = _token_snapshot()
    retrieved_paths = retriever.retrieve(
        question,
        domain=domain,
        max_depth=configs.RETRIEVER_CONFIG["max_depth"],
        max_width=configs.RETRIEVER_CONFIG["max_width"],
    )
    edited_paths, edit_stats = editor.run(retrieved_paths, question)
    answer = generator.generate_answer(edited_paths, question)
    latency = time.time() - start
    edit_stats["latency_seconds"] = latency
    edit_stats.update(_token_delta(token_before))
    return {
        "id": sample.get("_id", step),
        "question": question,
        "answer": sample.get("answer", sample.get("answers", "")),
        "output": answer,
        "context": edited_paths,
        "retrieved_path_count": len(retrieved_paths),
        "final_path_count": len(edited_paths),
        "efficiency": edit_stats,
    }


def main() -> Path:
    config = _effective_method_and_config(configs.CONSIST_CONFIG)
    llm_config = config.get("llm", {}) or {}
    if llm_config.get("timeout_seconds") is not None:
        os.environ["LLM_TIMEOUT_SECONDS"] = str(llm_config["timeout_seconds"])
    if llm_config.get("max_tokens") is not None:
        os.environ["LLM_MAX_TOKENS"] = str(llm_config["max_tokens"])
    method = config.get("method", "consist")
    if method not in CONSIST_MODES:
        raise ValueError(f"Unsupported consist mode: {method}. Valid modes: {sorted(CONSIST_MODES)}")

    raw_max_samples = config.get("max_samples", -1)
    max_samples = -1 if raw_max_samples is None else int(raw_max_samples)
    samples = _load_samples(max_samples)
    run_dir = _create_run_dir(config)
    _save_json(config, run_dir / "config_snapshot.json")
    with open(run_dir / "config_snapshot.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    _log(f"Starting {config.get('method_label', method)} on {_canonical_dataset_name()} with {len(samples)} samples", run_dir)
    if not samples:
        metrics = evaluate_predictions([], _canonical_dataset_name(), config.get("method_label", method), config)
        efficiency = aggregate_efficiency([])
        _save_json(metrics, run_dir / "metrics.json")
        _save_json(efficiency, run_dir / "efficiency.json")
        (run_dir / "predictions.jsonl").touch()
        (run_dir / "path.jsonl").touch()
        (run_dir / "edits.jsonl").touch()
        (run_dir / "edge_edits.jsonl").touch()
        _save_json([], run_dir / "path.json")
        _save_json([], run_dir / "edits.json")
        _save_json([], run_dir / "edge_edits.json")
        _log(f"Finished empty run. Results saved to {run_dir}", run_dir)
        return run_dir

    configs.TOKEN_COUNT["input"] = 0
    configs.TOKEN_COUNT["output"] = 0

    retriever = GraphRetriever()
    generator = MyGenerator()
    editor = CONSISTGraphEditor(CONSISTConfig.from_dict(config), llm=retriever.llm)

    predictions: List[Dict[str, Any]] = []
    path_records: List[Dict[str, Any]] = []
    edit_records: List[Dict[str, Any]] = []
    edge_edit_records: List[Dict[str, Any]] = []
    efficiency_items: List[Dict[str, Any]] = []
    predictions_path = run_dir / "predictions.jsonl"
    paths_path = run_dir / "path.jsonl"
    edits_path = run_dir / "edits.jsonl"
    edge_edits_path = run_dir / "edge_edits.jsonl"
    for step, sample in enumerate(tqdm(samples, desc=f"{config.get('method_label', method)}", dynamic_ncols=True)):
        try:
            result = process_sample(sample, step, retriever, generator, editor)
        except Exception as exc:
            _log(f"sample={step} failed: {type(exc).__name__}: {exc}", run_dir)
            result = {
                "id": sample.get("_id", step),
                "question": sample.get("question", ""),
                "answer": sample.get("answer", sample.get("answers", "")),
                "output": "",
                "context": [],
                "error": f"{type(exc).__name__}: {exc}",
                "efficiency": {"graph_size": 0, "conflict_pairs": 0, "latency_seconds": 0.0},
            }
        predictions.append(result)
        efficiency_items.append(result.get("efficiency", {}))
        _append_jsonl(result, predictions_path)
        path_record = _path_record(result)
        path_records.append(path_record)
        _append_jsonl(path_record, paths_path)
        edit_record = _edit_record(result)
        edit_records.append(edit_record)
        _append_jsonl(edit_record, edits_path)
        for edge_edit_record in _edge_edit_records(result):
            edge_edit_records.append(edge_edit_record)
            _append_jsonl(edge_edit_record, edge_edits_path)

    metrics = evaluate_predictions(predictions, _canonical_dataset_name(), config.get("method_label", method), config)
    efficiency = aggregate_efficiency(efficiency_items)
    _save_json(metrics, run_dir / "metrics.json")
    _save_json(efficiency, run_dir / "efficiency.json")
    _save_json(path_records, run_dir / "path.json")
    _save_json(edit_records, run_dir / "edits.json")
    _save_json(edge_edit_records, run_dir / "edge_edits.json")
    _log(f"Finished. metrics={metrics}", run_dir)
    _log(f"Results saved to {run_dir}", run_dir)
    return run_dir


if __name__ == "__main__":
    main()
