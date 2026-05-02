from pathlib import Path

from loguru import logger
import sys
import yaml
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Run BaseGraphRAG/CONSIST experiments.')
parser.add_argument('--mode', default="baseline_graphrag", type=str,
                    help='baseline_graphrag | consist | w_o_conflict | w_o_nash | add_only | random_pruning')
parser.add_argument('--consist_config', default="run_config/consist_default.yaml", type=str,
                    help='unified config file for consist/ablation runs')
parser.add_argument('--expand_COG', default=None, type=str2bool)
parser.add_argument('--dataset_name', default="wikiMQA", type=str, help='dataset name')
parser.add_argument('--query_setting', default="global", type=str, help=' global | local')
parser.add_argument('--rank_strategy', default=None, type=str, help='Options: [embedding|llm|hybrid|trained_ranker|ranker_llm]')
parser.add_argument('--max_depth', default=None, type=int, help='max depth of search')
parser.add_argument('--max_width', default=None, type=int, help='max depth of search')
parser.add_argument('--llm_backend', default=None, type=str, help='openai | transformers')
parser.add_argument('--local_llm_path', default=os.getenv("LOCAL_LLM_PATH", ""), type=str, help='local transformers model path')
parser.add_argument('--max_workers', default=None, type=int, help='override worker count')
parser.add_argument('--sample_limit', default=-1, type=int, help='limit samples for smoke tests; <=0 means all samples')
parser.add_argument('--max_samples', default=None, type=int, help='consist alias for sample_limit')
parser.add_argument('--seed_count', default=None, type=int, help='consist seed count m')
parser.add_argument('--lambda_c', default=None, type=float, help='consist conflict penalty')
parser.add_argument('--lambda_e', default=None, type=float, help='consist edge penalty')
parser.add_argument('--lambda_comp', default=None, type=float, help='consist component penalty')
parser.add_argument('--rollout_h', default=None, type=int, help='consist rollout steps H')
parser.add_argument('--k_add', default=None, type=int, help='consist add edge count')
parser.add_argument('--k_del', default=None, type=int, help='consist delete edge count')
parser.add_argument('--random_pruning_ratio', default=None, type=float, help='consist random pruning ratio')
parser.add_argument('--use_gumbel', default=None, type=str2bool, help='use Gumbel-TopK for Nash edit selection')
parser.add_argument('--gumbel_tau', default=None, type=float, help='Gumbel-TopK temperature')
parser.add_argument('--support_threshold', default=None, type=float, help='fallback support threshold for stopping')
parser.add_argument('--use_ras_sufficiency_judge', default=None, type=str2bool, help='use RAS-style [SUFFICIENT]/[SUBQ] stop judge')
parser.add_argument('--stop_on_no_retrieval', default=None, type=str2bool, help='stop when RAS planner returns [NO_RETRIEVAL]')
parser.add_argument('--use_llm_seed_selector', default=None, type=str2bool, help='use LLM prompt to select multi-seed nodes')
parser.add_argument('--record_efficiency', default=None, type=str2bool, help='write efficiency.json')
parser.add_argument('--output_root', default=None, type=str, help='consist result root')
parser.add_argument('--llm_model', default=None, type=str, help='override LLM model name')
parser.add_argument('--llm_api_base', default=None, type=str, help='override LLM API base URL')
parser.add_argument('--llm_temperature', default=None, type=float, help='override LLM temperature')
parser.add_argument('--llm_max_tokens', default=None, type=int, help='recorded max tokens for consist runs')
parser.add_argument('--use_llm_pair_judge', default=None, type=str2bool, help='use LLM to score conflict pairs')
parser.add_argument('--use_llm_support_judge', default=None, type=str2bool, help='use LLM to score graph answer support')

args = parser.parse_args()

DATASET_ALIASES = {
    "hotpotqa": "hotpot",
    "hotpot": "hotpot",
    "2wikimultihopqa": "wikiMQA",
    "2wiki": "wikiMQA",
    "wikimqa": "wikiMQA",
    "wikiMQA": "wikiMQA",
    "musique": "MuSiQue",
    "MuSiQue": "MuSiQue",
    "conflictqa_popqa": "ConcurrentQA",
    "conflictqa-popqa": "ConcurrentQA",
    "popqa": "ConcurrentQA",
    "concurrentqa": "ConcurrentQA",
    "ConcurrentQA": "ConcurrentQA",
}
DATASET_RESULT_ALIASES = {
    "hotpot": "hotpotqa",
    "wikiMQA": "2wikimultihopqa",
    "MuSiQue": "musique",
    "ConcurrentQA": "conflictqa_popqa",
}
DATASET_CONFIG_NAME = DATASET_ALIASES.get(args.dataset_name, DATASET_ALIASES.get(args.dataset_name.lower(), args.dataset_name))
DATASET_CONFIG_FILE = f'run_config/{DATASET_CONFIG_NAME}.yaml'
CHECKPOINTS = os.getenv("PATH_SCORER_CHECKPOINT_DIR", "checkpoints/")
PROJECT_ROOT = Path(__file__).resolve().parent
HF_HOME = Path(os.getenv("HF_HOME", PROJECT_ROOT / "hf_cache")).resolve()
HF_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str((HF_HOME / "hub").resolve()))
LLM_BACKEND = (args.llm_backend or os.getenv("LLM_BACKEND", "openai")).lower().strip()
LOCAL_LLM_PATH = args.local_llm_path.strip()

with open(DATASET_CONFIG_FILE, 'r', encoding='utf-8') as file:
    dataset_config = yaml.safe_load(file)

def _load_yaml_if_exists(path, default=None):
    config_path = Path(path)
    if not config_path.exists():
        return default or {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def _set_nested(config, dotted_key, value):
    target = config
    parts = dotted_key.split('.')
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value

def _merge_consist_cli_overrides(config):
    overrides = {
        "method": args.mode,
        "dataset_name": args.dataset_name,
        "max_samples": args.max_samples,
        "seed_count": args.seed_count,
        "lambda_c": args.lambda_c,
        "lambda_e": args.lambda_e,
        "lambda_comp": args.lambda_comp,
        "rollout_h": args.rollout_h,
        "k_add": args.k_add,
        "k_del": args.k_del,
        "random_pruning_ratio": args.random_pruning_ratio,
        "use_gumbel": args.use_gumbel,
        "gumbel_tau": args.gumbel_tau,
        "support_threshold": args.support_threshold,
        "use_ras_sufficiency_judge": args.use_ras_sufficiency_judge,
        "stop_on_no_retrieval": args.stop_on_no_retrieval,
        "use_llm_seed_selector": args.use_llm_seed_selector,
        "record_efficiency": args.record_efficiency,
        "output_root": args.output_root,
        "rank_strategy": args.rank_strategy,
        "max_depth": args.max_depth,
        "max_width": args.max_width,
        "expand_COG": args.expand_COG,
        "llm.backend": args.llm_backend,
        "llm.model": args.llm_model,
        "llm.api_base": args.llm_api_base,
        "llm.temperature": args.llm_temperature,
        "llm.max_tokens": args.llm_max_tokens,
        "llm.use_llm_pair_judge": args.use_llm_pair_judge,
        "llm.use_llm_support_judge": args.use_llm_support_judge,
    }
    for key, value in overrides.items():
        if value is not None:
            _set_nested(config, key, value)
    if args.sample_limit and args.sample_limit > 0 and args.max_samples is None:
        config["max_samples"] = args.sample_limit
    if args.mode == "random_pruning" and config.get("random_pruning_ratio") is None:
        config["random_pruning_ratio"] = 0.1
    return config

EXPERIMENT_MODE = args.mode
CONSIST_CONFIG = _merge_consist_cli_overrides(_load_yaml_if_exists(args.consist_config, default={}))
CONSIST_CONFIG.setdefault("method", EXPERIMENT_MODE)
CONSIST_CONFIG.setdefault("dataset_name", args.dataset_name)
CONSIST_CONFIG.setdefault("max_samples", args.max_samples if args.max_samples is not None else args.sample_limit)
if args.llm_backend is None and CONSIST_CONFIG.get("llm", {}).get("backend"):
    LLM_BACKEND = str(CONSIST_CONFIG["llm"]["backend"]).lower().strip()
if not LOCAL_LLM_PATH and CONSIST_CONFIG.get("llm", {}).get("local_model_path"):
    LOCAL_LLM_PATH = str(CONSIST_CONFIG["llm"]["local_model_path"]).strip()

if EXPERIMENT_MODE == "baseline_graphrag":
    EFFECTIVE_EXPAND_COG = True if args.expand_COG is None else args.expand_COG
    EFFECTIVE_RANK_STRATEGY = args.rank_strategy or "trained_ranker"
    EFFECTIVE_MAX_DEPTH = args.max_depth if args.max_depth is not None else 5
    EFFECTIVE_MAX_WIDTH = args.max_width if args.max_width is not None else 5
else:
    EFFECTIVE_EXPAND_COG = args.expand_COG if args.expand_COG is not None else CONSIST_CONFIG.get("expand_COG", True)
    EFFECTIVE_RANK_STRATEGY = args.rank_strategy or CONSIST_CONFIG.get("rank_strategy", "trained_ranker")
    EFFECTIVE_MAX_DEPTH = args.max_depth if args.max_depth is not None else int(CONSIST_CONFIG.get("max_depth", 5))
    EFFECTIVE_MAX_WIDTH = args.max_width if args.max_width is not None else int(CONSIST_CONFIG.get("max_width", 5))
NEO4J_CONFIG = {
    "uri": dataset_config['neo4j']['graph']['uri'],
    "auth": (dataset_config['neo4j']['graph']['username'], dataset_config['neo4j']['graph']['password']),
}
FEEDBACK_BASE = NEO4J_CONFIG
OCCURRENCE_GRAPH = {
    "uri": dataset_config['neo4j']['occurrence_graph']['uri'],
    "auth": (dataset_config['neo4j']['occurrence_graph']['username'], dataset_config['neo4j']['graph']['password']),
}

default_max_workers = 1 if LLM_BACKEND == "transformers" else 4
effective_sample_limit = args.max_samples if args.max_samples is not None else args.sample_limit
RUNNING_CONFIG = {
    'max_workers': args.max_workers or default_max_workers,
    'use_multithreading': True,
    'sample_limit': effective_sample_limit,
}

PIPLINE_CONFIG = {
    'TCR': True,
    "QF": False,
    "expand_COG": EFFECTIVE_EXPAND_COG,
    'two_stage': True
}


# Parameters related to the retrieval method
RETRIEVER_CONFIG = {
    "max_width": EFFECTIVE_MAX_WIDTH,
    'max_depth': EFFECTIVE_MAX_DEPTH,
    "min_similarity": 0,
    "rank_batch_size": 20, # Adjust different models appropriately; the stronger the model, the larger the batch_size.
    "rank_strategy": EFFECTIVE_RANK_STRATEGY,  # [embedding|llm|hybrid|trained_ranker|ranker_llm]
    "sufficiency_check": True
}

# 
DATASET_CONFIG = {
    "query_setting": args.query_setting,
    "dataset_name": f"{dataset_config['dataset']}",
    "domain": f"{dataset_config['dataset']}",
    "dataset_path": f"data/{dataset_config['dataset']}/",
    "output_dir": f"data/{dataset_config['dataset']}/output/",
    "dataset_file": f"data/{dataset_config['dataset']}/dataset/samples.json",
    "document_file": f"data/{dataset_config['dataset']}/dataset/documents.json",
    'document_limit': -1,
    "kg_triples_file": f"data/{dataset_config['dataset']}/output/extracted_triples.json",
    "entities_vector_store_path": f"data/{dataset_config['dataset']}/vector_stores/entities_vector_store",
    "doc_vector_store_path": f"data/{dataset_config['dataset']}/vector_stores/doc_vector_store",
    "results_store_path": f"results/{dataset_config['dataset']}/"
}


# -----------------------------
# embedding model config
# -----------------------------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENROUTER_BASE_URL = args.llm_api_base or CONSIST_CONFIG.get("llm", {}).get("api_base") or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = args.llm_model or CONSIST_CONFIG.get("llm", {}).get("model") or os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
LOCAL_LLM_NAME = os.getenv("LOCAL_LLM_NAME", Path(LOCAL_LLM_PATH).name if LOCAL_LLM_PATH else "local-transformers-model")

EMBEDDING_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", ""),
    "base_url": OPENAI_BASE_URL,
    "model_name": OPENAI_EMBED_MODEL
}

# -----------------------------
# LLM Config
# -----------------------------

LLM_CONFIG = {
    "endpoints": [
        {
            "backend": LLM_BACKEND,
            "api_key": os.getenv("OPENROUTER_API_KEY", ""),
            "api_base_url": OPENROUTER_BASE_URL,
            "local_model_path": LOCAL_LLM_PATH,
            "candidate_models": [LOCAL_LLM_NAME, LOCAL_LLM_NAME] if LLM_BACKEND == "transformers" else [OPENROUTER_MODEL, OPENROUTER_MODEL],
            "max_attempts": 3,
        }
    ]
}
LLM_RUNTIME_CONFIG = {
    "temperature": args.llm_temperature if args.llm_temperature is not None else CONSIST_CONFIG.get("llm", {}).get("temperature", 0.3),
    "max_tokens": args.llm_max_tokens if args.llm_max_tokens is not None else CONSIST_CONFIG.get("llm", {}).get("max_tokens", 512),
}

# -----------------------------
# Redis config
# -----------------------------
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6579,
    "decode_responses": True,
}


# -----------------------------
# log config
# -----------------------------
logger.remove()
logger.add(
    sys.stdout,
    level="ERROR",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level:<8}</level> | "
           "<cyan>{name:<15}</cyan>:<cyan>{function:<15}</cyan>:<cyan>{line:<4}</cyan> - "
           "<level>{message}</level>"
)



TOKEN_COUNT = {
    "input": 0,
    "output": 0
}

STATISTIC = {
    "average_depth": 0.0,
    "n_of_path": 0,
    'path_lens': []
}

import threading
file_lock = threading.Lock()
file_path = Path(f"results/{dataset_config['dataset']}/"+"path.jsonl")
file_path.parent.mkdir(parents=True, exist_ok=True)
path_container = open(file_path, 'w')
def write_to_file(data):
    with file_lock:
        path_container.write(data + '\n')
        path_container.flush()

print(f"""
Successfully read configuration file: {DATASET_CONFIG_FILE}
""")
print(f"""
{dataset_config['dataset']}
{PIPLINE_CONFIG}
{NEO4J_CONFIG}
{RETRIEVER_CONFIG}
""")


TRIPLE_COUNT = {
    "latent": 0,
    "total": 0
}
