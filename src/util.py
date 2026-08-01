import os
import random
import argparse
import torch
import torch.nn.functional as F
import numpy as np

def seed_everything(seed: int):
    "set all random seed for reproducible results."
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def model_zoo(args):
    vocab_size = {
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "TinyLlama-1.1B-Chat-v1.0-GPTQ": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
        # "deepseek-1.3b": 32256,
        # "deepseek-6.7b": 32256,
        "deepseek-1.3b": 32256,
        "deepseek-coder-1.3b-base-GGUF": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-33b": 32256,
        "qwen3-0.6b": 151936,
        "qwen3-8b": 151936,
        "qwen2.5-0.5b": 151936,
        "qwen2.5-7b": 151936,
    }
    
    zoo = {
        "codellama-7b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "codellama-34b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "codellama-70b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "TinyLlama-1.1B-Chat-v1.0-GPTQ": "../models/tinyllama-1.1b",
        "llama-2-7b": "../models/llama2-7b",
        "llama-2-70b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        # "deepseek-1.3b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "deepseek-6.7b": "../models/deepseek-coder-6.7b",
        # "deepseek-6.7b": "/home/jianhongbai/gyq/FastSD/deepseek-1.3b",
        # "deepseek-33b": "{REPLACE THIS WITH THE MODEL PATH IN YOUR ENVIRONMENT}",
        "deepseek-1.3b": "../models/deepseek-coder-1.3b",
        "deepseek-coder-1.3b-base-GGUF": "/home/jianhongbai/gyq/FastSD/deepseek-coder-1.3b-base-GGUF",
        "deepseek-33b": "deepseek-ai/deepseek-coder-33b-base",
        "qwen3-0.6b": "../models/qwen3_0.6b",
        "qwen3-8b": "../models/qwen3_8b",
        "qwen2.5-0.5b": "../models/qwen2.5-0.5b",
        "qwen2.5-7b": "../models/qwen2.5-7b",
    }

    if args.draft_model in vocab_size:
        args.vocab_size = vocab_size[args.draft_model]
    else:
        args.vocab_size = 32000

    if args.draft_model in zoo:
        args.draft_model = zoo[args.draft_model]
    if args.target_model in zoo:
        args.target_model = zoo[args.target_model]

def parse_arguments():
    """Specified arguments for running scripts."""
    parser = argparse.ArgumentParser(description='args for this file')

    # parser.add_argument('--dataset', type=str, default="humaneval")
    # parser.add_argument('--dataset', type=str, default="gsm8k")
    parser.add_argument('--dataset', type=str, default="mt_bench")

    # gsm8k data path
    # parser.add_argument('--data_path', type=str, default="/home/jianhongbai/gyq/FastSD/ParallelSpeculativeDecoding-main/data/gsm8k.jsonl")
    # humaneval data path
    parser.add_argument('--data_path', type=str, default="./data")
    # mt_bench data path
    # parser.add_argument('--data_path', type=str, default="/home/jianhongbai/gyq/FastSD/ParallelSpeculativeDecoding-main/data/mt_bench.jsonl")

    # parser.add_argument('--draft_model', type=str, default="deepseek-1.3b")
    # parser.add_argument('--target_model', type=str, default="deepseek-6.7b")
    parser.add_argument('--draft_model', type=str, default="TinyLlama-1.1B-Chat-v1.0-GPTQ")
    parser.add_argument('--target_model', type=str, default="llama-2-7b")
    
    parser.add_argument('--exp_name', '-e', type=str, default="test", help='folder name for storing results.')
    parser.add_argument('--eval_mode', type=str, default="sd", choices=["small", "large", "sd", "para_sd", "para_sd_wo_1", "para_sd_wo_2"], help='eval mode.')
    parser.add_argument('--num_samples_per_task', '-n', type=int, default=1, help='num_samples for a task (prompt) in humaneval dataset.')
    parser.add_argument('--seed', '-s', type=int, default=1234, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--max_tokens', type=int, default=400, help='max token number generated.')
    parser.add_argument('--temp', type=float, default=0, help='temperature for generating new tokens.')
    parser.add_argument('--top_k', type=int, default=0, help='top_k for ungreedy sampling strategy.')
    parser.add_argument('--top_p', type=float, default=0.95, help='top_p for ungreedy sampling strategy.')
    parser.add_argument('--gamma', type=int, default=6, help='guess time.')
    parser.add_argument("--num_drafts", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--token_budget", type=int, default=512)
    parser.add_argument(
        "--max_tasks_per_draft",
        type=int,
        default=10,
        help="max number of tasks each draft process will execute",
    )
    parser.add_argument(
        "--measure_energy",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable cloud-side GPU energy measurement service",
    )
    parser.add_argument(
        "--energy_api_host",
        type=str,
        default="127.0.0.1",
        help="FastAPI host for energy control endpoints",
    )
    parser.add_argument(
        "--energy_api_port",
        type=int,
        default=18080,
        help="FastAPI port for energy control endpoints",
    )
    parser.add_argument(
        "--energy_sample_interval_ms",
        type=int,
        default=50,
        help="power sampling interval in milliseconds",
    )
    parser.add_argument(
        "--energy_api_timeout_s",
        type=float,
        default=3.0,
        help="HTTP timeout for energy control requests",
    )
    parser.add_argument(
        "--server_sched_mode",
        type=str,
        default="fastsd",
        # pipeline 是sepcedge
        choices=["fastsd", "pipeline", "vanilla"],
        help="cloud scheduling mode: fastsd uses custom optimizations; pipeline uses baseline pipeline scheduling only",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="custom",
        choices=["custom", "vanilla", "proactive_only", "pipeline_only", "both"],
        help="ablation profile shortcut for vanilla/proactive/pipeline combinations",
    )
    parser.add_argument(
        "--enable_proactive_draft",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable proactive drafting while waiting verify response",
    )
    parser.add_argument(
        "--enable_pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable pipeline protocol (tail-only verify + gamma feedback)",
    )
    parser.add_argument(
        "--pipeline_gamma_adapt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable server-side gamma suggestion in pipeline mode",
    )
    parser.add_argument(
        "--pipeline_gamma_min",
        type=int,
        default=1,
        help="min gamma allowed by pipeline adaptation",
    )
    parser.add_argument(
        "--pipeline_gamma_max",
        type=int,
        default=16,
        help="max gamma allowed by pipeline adaptation",
    )
    parser.add_argument(
        "--pipeline_ema_alpha",
        type=float,
        default=0.2,
        help="EMA factor for pipeline timing estimators",
    )
    parser.add_argument(
        "--debug_pipeline",
        action="store_true",
        help="print pipeline-mode adaptation diagnostics on edge/cloud",
    )
    parser.add_argument(
        "--debug_verify_tokens",
        action="store_true",
        help="print draft/target token comparison during verify stage",
    )
    parser.add_argument(
        "--debug_max_print_steps",
        type=int,
        default=8,
        help="max token comparison steps to print per verify request",
    )
    args = parser.parse_args()
    if args.profile != "custom":
        # Baseline profiles must not be mixed with FastSD scheduler.
        if args.server_sched_mode == "fastsd":
            raise ValueError(
                f"profile='{args.profile}' is a baseline setting and cannot be used with "
                "server_sched_mode='fastsd'. Use --profile custom for FastSD runs."
            )
        if args.profile == "vanilla":
            args.enable_proactive_draft = False
            args.enable_pipeline = False
        elif args.profile == "proactive_only":
            args.enable_proactive_draft = True
            args.enable_pipeline = False
        elif args.profile == "pipeline_only":
            args.enable_proactive_draft = False
            args.enable_pipeline = True
        elif args.profile == "both":
            args.enable_proactive_draft = True
            args.enable_pipeline = True
    args.exp_name = os.path.join(os.getcwd(), "exp", args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)
    model_zoo(args)
    return args

def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits

def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    if temperature == 0:
        idx = logits.argmax(dim=1)
        new_logits = torch.zeros_like(logits, device=logits.device)
        new_logits[:, idx] = 1
        return new_logits.float()
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs

def sample(probs : torch.Tensor, num_samples: int = 1):
    idx_next = torch.multinomial(probs, num_samples=num_samples)
    return idx_next

def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum
