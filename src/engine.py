import os
import json
from dataclasses import dataclass, field
import torch
import transformers
import warnings
transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:  # pragma: no cover - optional dependency on some servers
    AutoGPTQForCausalLM = None
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .kvcache import KVCacheModel
from .kvcache_batching import KVCacheModel_batching
from .kvcache_varlen import varlen_generate, supports_varlen_model
from .kvcache4RC import KVCacheModel as KVCache2Model
from .cache_offload_policy import should_offload_target_cache
from .session_lifecycle import SessionLifecycle
from .cache_transfer import CacheTransferManager
from .progress_control import AcceptanceProfile, BatchCostModel, choose_gamma, draft_seconds_per_token
from .util import seed_everything, norm_logits, sample, max_fn
from .runtime import resolve_dtype, synchronize
from .speculative_sampling import (
    decode_sparse_block,
    greedy_accept_count,
    make_generator,
    rejection_sample,
    slice_sparse_block,
    sparse_block_from_dense,
    validate_candidate_support,
)
from transformers.cache_utils import DynamicCache
import queue
from collections import defaultdict, deque
import math
import random
import time
import threading
import requests

from .energy_meter import EnergyControlService, EnergyServiceConfig
from .fastsd_scheduler import (
    AdmissionPlan,
    FASTSD_DEFAULT_R1,
    FASTSD_DEFAULT_R2,
    FASTSD_DYNAMIC_WINDOW,
    build_fixed_wrr_order as _build_fixed_wrr_order,
    compute_priority_score as _compute_priority_score,
    full_prefix_bridge_tokens,
    length_category as _length_category,
    predict_next_verify_proc_ids as _predict_next_verify_proc_ids,
    WorkItem,
    abort_admission_plan,
    commit_admission_plan,
    plan_iteration,
    project_after_plan,
    reserve_admission_plan,
    should_switch_to_prefill as _should_switch_to_prefill,
    update_length_thresholds as _update_length_thresholds,
    verify_logit_position,
)
from .request_validation import canonicalize_request
from .scheduler_runtime import (
    CacheResidency,
    IngressRelay,
    ExecutionOutcome,
    KVPrefetchWorker,
    PrefetchCoordinator,
    SchedulerQueues,
    ShutdownRequested,
    SinglePlanExecutor,
    TargetSchedulerRuntime,
    cancel_prefill_work_items,
    commit_admission_progress,
    make_forward_batch_runner,
    make_forward_job_runner,
    set_thread_device,
    terminate_pending_work_items,
)


@dataclass
class _ExecutionJob:
    """One admitted plan handed from the control owner to the executor."""

    plan: AdmissionPlan
    item_by_id: dict
    verify_batch: list
    prefill_batch: list
    reserved_pids: set
    completed: set = field(default_factory=set)
    progress: dict = field(default_factory=dict)
    verify_applied: bool = False


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        
        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()
        
        # ! only parallel speculative decoding can use 2 processes
        assert (self.accelerator.num_processes == 1 and args.eval_mode in ["small", "large", "sd"]) or (self.accelerator.num_processes == 2 and args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1", "rc_para_sd"])

        # record metrics for report
        self.draft_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.last_generation_metrics = {}

    def _load_model_on_device(
        self,
        model_path: str,
        device: str,
        dtype: str | torch.dtype | None = None,
    ):
        quant_config = os.path.join(model_path, "quantize_config.json")
        if os.path.exists(quant_config):
            if AutoGPTQForCausalLM is None:
                raise RuntimeError(
                    f"auto_gptq is required for quantized model loading: {model_path}"
                )
            return AutoGPTQForCausalLM.from_quantized(
                model_path,
                device=device,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
            ).eval()
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device},
            torch_dtype=resolve_dtype(dtype, device),
            trust_remote_code=True,
        ).eval()
    
    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        if self.args.eval_mode == "small":
            self.draft_model = self._load_model_on_device(
                self.args.draft_model,
                self.args.draft_device,
                getattr(self.args, "draft_dtype", "auto"),
            )
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        elif self.args.eval_mode == "sd":
            self.draft_model = self._load_model_on_device(
                self.args.draft_model,
                self.args.draft_device,
                getattr(self.args, "draft_dtype", "auto"),
            )
            self.target_model = self._load_model_on_device(
                self.args.target_model,
                self.args.target_device,
                getattr(self.args, "target_dtype", "bfloat16"),
            )

        elif self.args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1"]:
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="balanced_low_0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        elif self.args.eval_mode == "rc_para_sd":
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
                self.draft_model_2 = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map=f"cuda:{torch.cuda.device_count()-1}", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        
        self.vocab_size = self.args.vocab_size

    def _load_target_model_for_service(self, model_path: str, device: str):
        quant_config = os.path.join(model_path, "quantize_config.json")
        if os.path.exists(quant_config):
            if AutoGPTQForCausalLM is None:
                raise RuntimeError(
                    f"auto_gptq is required for quantized model loading: {model_path}"
                )
            return AutoGPTQForCausalLM.from_quantized(
                model_path,
                device=device,
                use_safetensors=True,
                trust_remote_code=True,
                use_triton=False,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={"": device},
            torch_dtype=resolve_dtype(
                getattr(self.args, "target_dtype", "bfloat16"), device
            ),
            trust_remote_code=True,
        ).eval()

    def _service_target_device(self) -> str:
        # Priority: explicit --target_device argument (non-default) >
        # FASTSD_TARGET_DEVICE environment variable > cuda:0.
        target_device = getattr(self.args, "target_device", None)
        if target_device and str(target_device) != "cuda:0":
            return str(target_device)
        return os.environ.get("FASTSD_TARGET_DEVICE", "cuda:0")

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define either pad_token_id or eos_token_id")
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _energy_api_url(self, action: str) -> str:
        return f"http://{self.args.energy_api_host}:{self.args.energy_api_port}/measure/{action}"

    def _post_energy_event(self, action: str, payload: dict, retries: int = 1):
        url = self._energy_api_url(action)
        last_error = None
        for _ in range(max(1, retries)):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=float(self.args.energy_api_timeout_s),
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(f"failed to call energy endpoint {url}: {last_error}") from last_error

    def _maybe_start_energy_service(self):
        if not getattr(self.args, "measure_energy", False):
            return None

        output_path = os.path.join(self.args.exp_name, "energy_metrics.json")
        cfg = EnergyServiceConfig(
            gpu_index=0,
            host=self.args.energy_api_host,
            port=int(self.args.energy_api_port),
            sample_interval_ms=int(self.args.energy_sample_interval_ms),
            output_path=output_path,
        )
        service = EnergyControlService(cfg)
        service.start()
        self.color_print(
            f"[ENERGY] FastAPI started at {self.args.energy_api_host}:{self.args.energy_api_port}",
            2,
        )
        return service

    @abstractmethod
    def load_data(self):
        pass
    
    @abstractmethod
    def preprocess(self, input_text):
        pass
    
    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass
    
    @torch.no_grad()
    def autoregressive_sampling(self, prefix):
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError("Auto-Regressive Decoding can be used only in small / large eval mode!")
        
        prefix = prefix.to(model.device)
        generation_start = time.perf_counter()
        first_token_time = None

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens
        
        x = prefix
        past_key_values = None
        while x.shape[1] < max_tokens:
            if past_key_values:
                last_ids = x[:, -1]
                if last_ids.dim() == 1:
                    last_ids = last_ids.unsqueeze(0)
                outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
            else:
                outputs = model(x)

            if self.accelerator.is_main_process:
                if self.args.eval_mode == "small":
                    self.draft_forward_times += 1
                elif self.args.eval_mode == "large":
                    self.target_forward_times += 1

            last_p = norm_logits(outputs.logits[::, -1, :], self.args.temp, self.args.top_k, self.args.top_p)
            past_key_values = outputs.past_key_values
            idx_next = sample(last_p)
            x = torch.cat((x, idx_next), dim=1)
            if first_token_time is None:
                synchronize(model.device)
                first_token_time = time.perf_counter()
        synchronize(model.device)
        completion_time = time.perf_counter()
        generated_tokens = int(x.shape[1] - prefix_len)
        self.last_generation_metrics = {
            "ttft_ms": (first_token_time - generation_start) * 1000.0,
            "tpot_ms": (
                (completion_time - first_token_time) * 1000.0 / (generated_tokens - 1)
                if generated_tokens > 1
                else 0.0
            ),
            "e2e_ms": (completion_time - generation_start) * 1000.0,
            "generated_tokens": generated_tokens,
        }
        return x

    @torch.no_grad()
    def speculative_decoding(self, prefix):
        original_prefix_len = prefix.shape[1]
        max_tokens = original_prefix_len + self.args.max_tokens
        generation_start = time.perf_counter()
        first_token_time = None
        
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        
        verify_method = str(getattr(self.args, "verify_method", "rejection")).lower()
        draft_top_k = int(getattr(self.args, "draft_top_k", 64))
        draft_generator = make_generator(self.seed, draft_device)
        verify_generator = make_generator(self.seed + 1, target_device)
        draft_cache_top_k = draft_top_k if verify_method == "rejection" else self.args.top_k
        draft_cache_top_p = 0.0 if verify_method == "rejection" else self.args.top_p
        approx_model_cache = KVCacheModel(
            self.draft_model,
            self.args.temp,
            draft_cache_top_k,
            draft_cache_top_p,
            generator=draft_generator,
            exact_top_k=(verify_method == "rejection"),
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            if verify_method == "rejection":
                x, draft_rows = approx_model_cache.generate_with_probs(
                    prefix.to(draft_device), self.args.gamma
                )
            else:
                x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
                draft_rows = None
            _ = target_model_cache.generate(x.to(target_device), 1, sample_next=False)
            if self.accelerator.is_main_process:
                self.draft_forward_times += self.args.gamma
                self.target_forward_times += 1
            target_history = target_model_cache._prob_history[:, :, :self.vocab_size]
            target_start = prefix_len - 1
            target_rows = target_history[:, target_start:target_start + self.args.gamma + 1, :]
            draft_tokens = x[:, prefix_len:prefix_len + self.args.gamma]

            if verify_method == "rejection":
                sparse_ids, sparse_probs = sparse_block_from_dense(
                    draft_rows, draft_top_k
                )
                accepted_cnt, t = rejection_sample(
                    draft_tokens,
                    target_rows[0],
                    sparse_ids,
                    sparse_probs,
                    generator=verify_generator,
                )
            else:
                accepted_cnt = greedy_accept_count(draft_tokens, target_rows)
                correction_row = target_rows[0, accepted_cnt]
                t = torch.argmax(correction_row, dim=-1).reshape(1)

            accepted_len = prefix_len + accepted_cnt
            self.num_acc_tokens.append(accepted_cnt)
            prefix = x[:, :accepted_len]

            approx_model_cache.rollback(accepted_len)
            t = t.to(draft_device).reshape(1, 1)
            # The target cache contains one extra row only when all draft
            # tokens were accepted and its next-token row is retained.
            target_model_cache.rollback(
                accepted_len + 1 if accepted_cnt == self.args.gamma else accepted_len
            )
            prefix = torch.cat((prefix, t), dim=1)
            prefix = prefix[:, :max_tokens]
            if first_token_time is None:
                synchronize(draft_device)
                first_token_time = time.perf_counter()
        synchronize(draft_device)
        completion_time = time.perf_counter()
        generated_tokens = int(prefix.shape[1] - original_prefix_len)
        self.last_generation_metrics = {
            "ttft_ms": (first_token_time - generation_start) * 1000.0,
            "tpot_ms": (
                (completion_time - first_token_time) * 1000.0 / (generated_tokens - 1)
                if generated_tokens > 1
                else 0.0
            ),
            "e2e_ms": (completion_time - generation_start) * 1000.0,
            "generated_tokens": generated_tokens,
        }
        return prefix

    @torch.no_grad()
    def run_draft_process(
        self,
        tokenizer,
        request_queue,
        response_queues,
        proc_id,
        measure_started,
        completed_tasks,
        control_lock,
        total_tasks,
    ):
        """
        每个 Draft 模型独立读取 HumanEval 样本，不断向 target 请求推理验证
        """
        # self.color_print(f"Loading models:\n{self.args.draft_model}", 3)
        gpu_id = (proc_id % 7) + 1  # proc_id=0 对应 cuda:1, ..., proc_id=6 对应 cuda:7
        device = f"cuda:{gpu_id}"
        self.color_print(f"[Draft {proc_id}] Loading model on {device}", 3)

        draft_model = AutoGPTQForCausalLM.from_quantized(
            self.args.draft_model,
            device=device,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
        )

        seed_everything(42 + proc_id)
        # approx_model_cache = KVCacheModel(draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        # approx_model_cache.vocab_size = tokenizer.vocab_size

        with open(self.args.data_path, "r") as f:
            samples = [eval(l) for l in f.readlines()]  # each line is a dict: {"task_id":..., "prompt":...}

        samples = samples[: self.args.max_tasks_per_draft]
        task_type_flag = 0
        for idx, sample in enumerate(samples):
            approx_model_cache = KVCacheModel(draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            approx_model_cache.vocab_size = self.vocab_size

            if self.args.dataset == "gsm8k":
                input_text = sample["question"].strip() # for gsm8k dataset
            elif self.args.dataset == "humaneval":
                input_text = sample["prompt"].strip() # for humaneval dataset
            elif self.args.dataset == "mt_bench":
                sample["task_id"] = idx
                input_text = sample["turns"][0].strip() # for mt_bench dataset

            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(draft_model.device)
            prefix = input_ids.clone()
            self.color_print(f"[Draft {proc_id}] Loop {idx}: Finished {sample['task_id']} ({prefix.shape[1]} tokens)",3)

            if getattr(self.args, "measure_energy", False):
                should_start = False
                with control_lock:
                    if not bool(measure_started.value):
                        measure_started.value = 1
                        should_start = True
                if should_start:
                    self._post_energy_event(
                        "start",
                        {
                            "exp_id": self.args.exp_name,
                            "metadata": {"trigger_proc_id": int(proc_id)},
                        },
                        retries=60,
                    )
                    self.color_print("[ENERGY] START_MEASURE sent", 2)

            while prefix.shape[1] < self.args.max_tokens:
                prefix_len = prefix.shape[1]

                draft_comp_start_time = time.time()
                x = approx_model_cache.generate(prefix, self.args.gamma)
                draft_comp_time = time.time() - draft_comp_start_time
                request = {
                    "task_id": sample["task_id"],
                    "draft_output": x.cpu(),
                    "draft_cache": approx_model_cache._prob_history[:, prefix_len:, :].cpu(),
                    "prefix_len": prefix_len,
                    "proc_id": proc_id,
                    "lag": draft_comp_time,
                    "current_time": time.time(),
                    "task_type": "prefill" if task_type_flag == 0 else "verify",
                }
                request_queue.put(request)
                # response = response_queue.get()
                response = response_queues[proc_id].get()

                accepted = response["accepted"]
                final_token = response["final_token"]
                prefix = torch.cat((x[:, :accepted], final_token.to(x.device)), dim=1)

                approx_model_cache.rollback(accepted)  # 更新 Draft 的 cache 状态

                task_type_flag += 1

            task_type_flag = 0
            print(f"[Draft {proc_id}] Finished {sample['task_id']} ({prefix.shape[1]} tokens)\n")

            if getattr(self.args, "measure_energy", False):
                should_stop = False
                finished = 0
                with control_lock:
                    completed_tasks.value += 1
                    finished = int(completed_tasks.value)
                    if finished == int(total_tasks):
                        should_stop = True
                if should_stop:
                    result = self._post_energy_event(
                        "stop",
                        {
                            "exp_id": self.args.exp_name,
                            "total_tasks": finished,
                            "metadata": {"trigger_proc_id": int(proc_id)},
                        },
                        retries=10,
                    )
                    self.color_print(
                        f"[ENERGY] STOP_MEASURE sent, campaign={result['result_b_campaign_window']['energy_j']:.4f}J",
                        2,
                    )

    @torch.no_grad()
    def run_target_process(self, tokenizer, request_queue, response_queues):
        self.color_print(f"Loading models:\n{self.args.target_model}", 3)
        target_model = self._load_target_model_for_service(
            self.args.target_model, self._service_target_device()
        )
        self.vocab_size = self.args.vocab_size
        energy_service = self._maybe_start_energy_service()

        target_model_caches = {}
        sampling_generators = {}
        accept_stats = defaultdict(lambda: [0, 1])  # [accepted_sum, total_sum], avoid div0

        def move_dynamic_cache_to(cache, device):
            legacy_cache = cache.to_legacy_cache()
            current_device = legacy_cache[0][0].device
            if str(current_device) == device:
                return cache

            new_cache = []
            for k, v in legacy_cache:
                new_cache.append((k.to(device), v.to(device)))
            return DynamicCache.from_legacy_cache(new_cache)

        # 两组队列
        task_queues = {
            "prefill": {"short": queue.Queue(), "mid": queue.Queue(), "long": queue.Queue()},
            "verify": {"short": queue.Queue(), "mid": queue.Queue(), "long": queue.Queue()},
        }

        def get_length_category(seq_len):
            if seq_len <= 128:
                return "short"
            elif seq_len <= 512:
                return "mid"
            else:
                return "long"

        def handle_request(request):
            x = request["draft_output"].to(target_model.device)
            prefix_len = request["prefix_len"]
            proc_id = request["proc_id"]
            tail_only = request.get("tail_only", False)
            has_bridge_token = request.get("has_bridge_token", False)
            self.color_print(f"processing request from draft {proc_id}", 3)

            # if proc_id not in target_model_caches or request["task_type"] == "prefill":
            if request["task_type"] == "prefill":
                cache = KVCacheModel(target_model, self.args.temp, self.args.top_k, self.args.top_p)
                cache.vocab_size = self.vocab_size
                target_model_caches[proc_id] = cache
            else:
                cache = target_model_caches[proc_id]
                cache._past_key_values = move_dynamic_cache_to(cache._past_key_values, target_model.device)

            x_for_target = x
            if request["task_type"] == "verify" and tail_only:
                # edge tail_only 请求只带尾部 token，这里补齐一个虚拟前缀长度，
                # 让 KVCacheModel 内部按 cached_len 正确截取 residual tokens。
                cached_len = cache._past_key_values.get_seq_length()
                pad = torch.full(
                    (1, cached_len),
                    tokenizer.pad_token_id,
                    dtype=x.dtype,
                    device=x.device,
                )
                x_for_target = torch.cat((pad, x), dim=1)

            _ = cache.generate(x_for_target, 1, sample_next=False)

            if request["task_type"] == "prefill":
                # prefill 仅用于初始化目标侧 cache，需要回滚到 prefix 长度
                cache.rollback(prefix_len)
                if should_offload_target_cache("prefill", pid=proc_id):
                    cache._past_key_values = move_dynamic_cache_to(cache._past_key_values, "cpu")
                response_queues[proc_id].put({
                    "status": "prefill_ok",
                })
                print(f"Finished prefill for draft {proc_id}, prefix length {prefix_len}")
                return

            req_gamma = max(1, int(request.get("gamma", self.args.gamma)))
            verify_method = str(
                request.get("verify_method", getattr(self.args, "verify_method", "rejection"))
            ).lower()
            if tail_only:
                tail_offset = 1 if has_bridge_token else 0
                draft_tokens = x[:, tail_offset:tail_offset + req_gamma]
            else:
                draft_tokens = x[:, prefix_len:prefix_len + req_gamma]
            target_rows = torch.cat(
                [
                    cache._prob_history[:, prefix_len + i - 1, :self.vocab_size]
                    for i in range(req_gamma)
                ]
                + [cache._prob_history[:, -1, :self.vocab_size]],
                dim=0,
            ).float()
            if verify_method == "rejection":
                sparse_ids, sparse_probs = decode_sparse_block(
                    request.get("draft_prob_block"),
                    device=target_model.device,
                    vocab_size=self.vocab_size,
                )
                generator = sampling_generators.get(proc_id)
                if generator is None:
                    generator = make_generator(
                        int(request.get("sampling_seed", getattr(self, "seed", 0))),
                        target_model.device,
                    )
                    sampling_generators[proc_id] = generator
                accepted_cnt, new_token = rejection_sample(
                    draft_tokens,
                    target_rows,
                    sparse_ids,
                    sparse_probs,
                    generator=generator,
                )
            elif verify_method == "greedy":
                accepted_cnt = greedy_accept_count(draft_tokens, target_rows[:-1])
                new_token = torch.argmax(target_rows[accepted_cnt], dim=-1).reshape(1)
            else:
                raise ValueError(f"unknown verify_method: {verify_method!r}")
            accepted_len = prefix_len + accepted_cnt
            new_token = new_token.reshape(1, 1)

            # Keep only accepted prefix in target cache.
            # final_token will be consumed in the next verify call together with new draft tokens.
            cache.rollback(accepted_len)
            if should_offload_target_cache("verify", pid=proc_id):
                cache._past_key_values = move_dynamic_cache_to(cache._past_key_values, "cpu")

            response_queues[proc_id].put({
                "accepted": accepted_len,
                "final_token": int(new_token.item()),
            })

            # Update accept statistics
            accept_stats[proc_id][0] += accepted_len - prefix_len
            accept_stats[proc_id][1] += req_gamma

        def schedule_tasks(task_type):
            schedule = ["short"] * 3 + ["mid"] * 2 + ["long"]
            for category in schedule:
                batch = []
                while len(batch) < 1 and not task_queues[task_type][category].empty():
                    batch.append(task_queues[task_type][category].get())
                for req in batch:
                    if energy_service is not None:
                        energy_service.enter_active()
                    try:
                        handle_request(req)
                    finally:
                        if energy_service is not None:
                            energy_service.exit_active()

        def compute_priority(req, accept_stats, alpha=1, beta=1):
            proc_id = req["proc_id"]
            lag = req["lag"]
            elapsed = time.time() - req["current_time"]

            if req["task_type"] == "prefill":
                lamda = 0.01
            else:
                lamda = 0.001

            # 防止除以0
            if proc_id not in accept_stats or accept_stats[proc_id][1] == 0:
                accept_prob = 1.0
            else:
                accept_prob = accept_stats[proc_id][0] / accept_stats[proc_id][1]
            if accept_prob <= 0:
                accept_prob = 1.0

            return (-alpha * (lag + self.args.gamma * random.uniform(1e-10, 1.0)) / accept_prob +
                    beta * (math.exp(lamda * elapsed) - 1))

        # 时间监控
        last_prefill_time = time.time()

        try:
            while True:
                try:
                    while True:
                        request = request_queue.get(timeout=0.01)
                        if request is None:
                            return
                        category = get_length_category(request["prefix_len"])
                        # task_type = request.get("type", "verify")
                        task_type = request["task_type"]
                        task_queues[task_type][category].put(request)
                except queue.Empty:
                    pass

                # Priority-based sorting before scheduling
                for task_type in ["verify", "prefill"]:
                    for category in task_queues[task_type]:
                        items = []
                        while not task_queues[task_type][category].empty():
                            items.append(task_queues[task_type][category].get())
                        items.sort(key=lambda req: compute_priority(req, accept_stats), reverse=True)
                        for item in items:
                            task_queues[task_type][category].put(item)

                has_verify = any(not q.empty() for q in task_queues["verify"].values())
                has_prefill = any(not q.empty() for q in task_queues["prefill"].values())

                # 若仅有 verify 任务，必须始终可调度；否则会在启动超过 10s 后饿死 verify。
                if has_verify and ((time.time() - last_prefill_time) < 10 or not has_prefill):
                    schedule_tasks("verify")
                    last_prefill_time = time.time()
                elif has_prefill:
                    schedule_tasks("prefill")
                    last_prefill_time = time.time()
                else:
                    time.sleep(0.01)
        finally:
            if energy_service is not None:
                energy_service.shutdown()

    # ----------------------------------------------------------------------
    # Target-side process – batch prefill / verify 版本
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def run_target_process_batching(self, tokenizer,
                                    request_queue,  # Draft → Target 公共队列
                                    response_queues):  # proc_id → Queue

        self.color_print(f"Loading target model: {self.args.target_model}", 3)
        target_model = self._load_target_model_for_service(
            self.args.target_model, self._service_target_device()
        )
        self.vocab_size = self.args.vocab_size
        energy_service = self._maybe_start_energy_service()

        # 批处理版 KVCache 管理器
        kv_cache_manager = KVCacheModel_batching(target_model,
                                        temperature=self.args.temp,
                                        top_k=self.args.top_k,
                                        top_p=self.args.top_p)
        kv_cache_manager.vocab_size = self.vocab_size
        kv_cache_manager.bounded_probabilities = True
        requested_kv_batch_mode = getattr(self.args, "kv_batch_mode", "varlen")
        if requested_kv_batch_mode == "varlen" and not supports_varlen_model(target_model):
            raise ValueError(
                "--kv_batch_mode varlen requires a supported decoder layout (such as Llama); "
                "this model would require padding. Choose a compatible checkpoint "
                "or explicitly select --kv_batch_mode padded."
            )

        # --------------------------- 队列与统计 ---------------------------
        # One control owner (this thread) mutates queues/WorkItems/scheduler
        # state; one executor thread runs forwards; one worker thread moves
        # KV caches.  See src/scheduler_runtime.py for the ownership contract.
        scheduler_state = {"wrr_cursor": 0, "current_cycle": 0}
        scheduler_metrics = {
            "iterations": 0,
            "used_tokens": 0,
            "plans": 0,
            "verify_slices": 0,
            "prefill_slices": 0,
            "partial_verify": 0,
            "partial_prefill": 0,
            "prefetch_plans": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_evictions": 0,
            "prefetch_reserved_excluded": 0,
            "prefetch_empty_desired": 0,
            "prefetch_no_verify_candidates": 0,
            "prefetch_desired_unchanged": 0,
            "prefetch_intent_inflight": 0,
            "prefetch_retry": 0,
            "prefetch_retry_throttled": 0,
            "prefetch_noop": 0,
            "prefetch_async_submitted": 0,
            "prefetch_async_dropped": 0,
            "prefetch_async_completed": 0,
            "prefetch_async_moved": 0,
            "prefetch_async_failed": 0,
            "prefetch_async_skipped_stale": 0,
            "prefetch_async_superseded": 0,
            "prefetch_async_missing_cache": 0,
            "prefetch_pending_underflow": 0,
            "prefetch_async_last_error": None,
            "finished_work_items_removed": 0,
            "empty_plans": 0,
            "empty_plans_with_live_work": 0,
            "idle_waits": 0,
            "incoming_during_forward": 0,
            "shutdown_inflight_timeout": 0,
            "shutdown_drained_plans": 0,
            "shutdown_terminated_work_items": 0,
            "timing_updates": 0,
            "timing_ready_prefill_forwards": 0,
            "prefill_cancels": 0,
            "prefill_cancels_deferred": 0,
        }
        task_queues = SchedulerQueues(scheduler_metrics)
        work_items = {}
        pending_prefill_timing = {}
        pending_prefill_cancel = set()
        accept_stats = defaultdict(lambda: [0, 1])  # {proc_id: [accepted_sum, total_sum]}
        # The executor updates acceptance statistics while the control owner
        # may be planning a lookahead; the planner always reads a copy.
        accept_stats_lock = threading.Lock()
        committed_prefix_tokens = {}  # {proc_id: List[int]} for debug context display
        sampling_generators = {}  # session/proc id -> target-side rejection RNG
        progress_profiles = {}
        batch_cost = BatchCostModel()
        batch_cost_lock = threading.RLock()
        verify_time_ema = {}
        edge_draft_per_token_ema = {}
        edge_rtt_ema = {}
        recent_prefix_lens = deque(maxlen=FASTSD_DYNAMIC_WINDOW)
        len_r1 = FASTSD_DEFAULT_R1
        len_r2 = FASTSD_DEFAULT_R2
        residency = CacheResidency()
        cache_lock = residency.lock
        preloaded_gpu_pids = residency.resident
        # pids whose cache is currently owned by the executing batch (forward +
        # rollback window) or by an admitted-but-not-yet-forwarded plan. The
        # async prefetch worker must never touch these.
        active_pids = residency.active
        in_flight = {"plan": None}

        def cancel_session(pid):
            removed = set()
            for work_id, item in list(work_items.items()):
                if item.proc_id != pid:
                    continue
                item.cancel_pending = True
                removed.add(work_id)
                work_items.pop(work_id, None)
                response_queues[item.response_key or pid].put({"error": "session released"})
            task_queues.remove_work_ids(removed)

        def cleanup_session(pid):
            with cache_lock:
                residency.bump_locked(pid)
                kv_cache_manager.reset(pid)
                residency.resident.discard(pid)
                # In-flight prefetch installation also checks cache identity.
                # Removing this entry does not let a stale copy resurrect it.
                residency.versions.pop(pid, None)
            with accept_stats_lock:
                accept_stats.pop(pid, None)
            for store in (committed_prefix_tokens, sampling_generators, verify_time_ema,
                          edge_draft_per_token_ema, edge_rtt_ema, pending_prefill_timing, progress_profiles):
                store.pop(pid, None)
            pending_prefill_cancel.discard(pid)

        lifecycle = SessionLifecycle(
            cleanup=cleanup_session, cancel=cancel_session,
            protected=residency.protected_snapshot,
            busy=lambda: {item.proc_id for item in work_items.values() if not item.finished},
            reply=lambda key, payload: response_queues[key].put(payload),
            idle_timeout=float(getattr(self.args, "session_idle_timeout", 300.0)),
        )

        def flush_session_releases():
            transfers.collect()
            lifecycle.sweep()
            scheduler_metrics["released_sessions"] = lifecycle.released
            scheduler_metrics["cache_sessions"] = len(kv_cache_manager._past_key_values)
            scheduler_metrics["probability_sessions"] = len(kv_cache_manager._prob_history)
            scheduler_metrics["peak_cache_sessions"] = max(
                scheduler_metrics.get("peak_cache_sessions", 0),
                scheduler_metrics["cache_sessions"],
            )

        def _safe_float(value, default=0.0) -> float:
            """Sanitize a request-sourced float: JSON can carry inf/NaN or
            non-numeric types, which must never reach arithmetic downstream."""
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                return float(default)
            if not math.isfinite(f):
                return float(default)
            return f

        def _safe_int(value, default=0) -> int:
            """Sanitize a request-sourced integer field (gamma / prefix_len):
            bare int() on crafted values (1e999, 'abc') would crash the worker."""
            if isinstance(value, bool):
                return int(value)
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                return int(default)
            if not math.isfinite(f) or abs(f) > 1e12:
                return int(default)
            return int(f)

        def _sampling_generator(pid, seed):
            """Return one persistent target RNG per logical decode session."""

            generator = sampling_generators.get(pid)
            if generator is None:
                generator = make_generator(int(seed), target_model.device)
                sampling_generators[pid] = generator
            return generator

        def _strict_duration(value, field: str) -> float:
            """Validate timing control values; never turn bad telemetry into RTT/2."""

            try:
                parsed = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field} must be finite and non-negative") from exc
            if not math.isfinite(parsed) or parsed < 0.0:
                raise ValueError(f"{field} must be finite and non-negative")
            return parsed

        def apply_prefill_timing(control: dict) -> bool:
            """Atomically unlock the queued Prefill for one session."""

            pid = control.get("proc_id")
            local_prefill_s = _strict_duration(control.get("local_prefill_s"), "local_prefill_s")
            local_decode_s = _strict_duration(
                control.get("local_decode_per_token_s"), "local_decode_per_token_s"
            )
            gamma = _safe_int(control.get("prefill_gamma", control.get("m_i", 0)), default=0)
            if gamma <= 0 or gamma > int(getattr(self.args, "max_tokens", 400) or 400):
                raise ValueError("prefill timing gamma must be positive and within max_tokens")
            push_s = _strict_duration(control.get("push_s"), "push_s")
            update = {
                "local_prefill_s": local_prefill_s,
                "local_decode_per_token_s": local_decode_s,
                "prefill_gamma": gamma,
                "m_i": gamma,
                "push_s": push_s,
                "timing_ready": True,
                "timing_required": True,
            }
            target = next(
                (
                    item
                    for item in work_items.values()
                    if item.proc_id == pid and item.task_type == "prefill" and not item.finished
                ),
                None,
            )
            if target is None:
                # Queue FIFO normally means the initial request is already
                # present.  Keep an update that races ingress so it is applied
                # exactly once when that WorkItem is created.
                pending_prefill_timing[pid] = update
                scheduler_metrics["timing_updates"] += 1
                return False
            updated_request = dict(target.request)
            updated_request.update(update)
            target.request = updated_request
            scheduler_metrics["timing_updates"] += 1
            return True

        def apply_prefill_cancel(control: dict) -> bool:
            """Discard one failed timing-gated Prefill before it can execute.

            A cancel that matches a WorkItem already admitted into the running
            plan cannot tear down the in-flight forward.  It is deferred: the
            item is removed from the queues immediately, and the completion
            path drops its response after the executor returns.
            """

            pid = control.get("proc_id")
            task_id = str(control.get("task_id", ""))
            pending_prefill_timing.pop(pid, None)
            removed_ids, deferred = cancel_prefill_work_items(
                work_items=work_items,
                queues=task_queues,
                proc_id=pid,
                task_id=task_id,
            )
            if removed_ids:
                scheduler_metrics["prefill_cancels"] += len(removed_ids)
                scheduler_metrics["prefill_cancels_deferred"] += deferred
                return True

            # The cancel can race ahead of initial request ingress.  Remember
            # it once, and the ingress path below will drop exactly that
            # matching Prefill rather than creating an unexecutable WorkItem.
            pending_prefill_cancel.add(pid)
            scheduler_metrics["prefill_cancels"] += 1
            return False

        def update_ema(store: dict, pid, value: float) -> float:
            alpha = float(getattr(self.args, "pipeline_ema_alpha", 0.2))
            old = store.get(pid, None)
            if old is None:
                store[pid] = value
            else:
                store[pid] = (1 - alpha) * old + alpha * value
            return store[pid]

        def token_repr(token_id: int) -> str:
            tok_id = int(token_id)
            piece = tokenizer.convert_ids_to_tokens(tok_id)
            if piece is None:
                piece = "<None>"
            # Human-readable view (escaped), e.g. '\n', '    ', '('
            text = tokenizer.decode([tok_id], skip_special_tokens=False)
            text_repr = repr(text)
            return f"{tok_id}(text={text_repr}, piece={piece})"

        def ids_text_repr(token_ids: list[int]) -> str:
            if not token_ids:
                return "''"
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
            return repr(text)

        transfers = CacheTransferManager(
            target_model.device,
            enabled=bool(getattr(self.args, "async_kv_transfer", True)),
            pinned_pool_bytes=int(getattr(self.args, "pinned_pool_mb", 128)) * 1024 ** 2,
        )
        move_dynamic_cache_to = transfers.move

        def cache_memory_pressure():
            if target_model.device.type != "cuda":
                return False
            free, _ = torch.cuda.mem_get_info(target_model.device)
            reserve = float(getattr(self.args, "kv_headroom_mb", 512.0)) * 1024 ** 2
            return free < reserve

        def finish_cache_phase(batch):
            # Retain unfinished work across slices. Under pressure fall back
            # to offload; the lookahead worker may evict it when other work is
            # predicted to run first.
            with cache_lock:
                for req in batch:
                    pid = req["proc_id"]
                    transfers.mark_computed(kv_cache_manager._past_key_values[pid])
                    continuing = (not req.get("_prefill_final", True)
                                  if req["task_type"] == "prefill"
                                  else bool(req.get("_internal_chunk_nonfinal", False)))
                    cache = kv_cache_manager._past_key_values[pid]
                    offload = should_offload_target_cache(
                        req["task_type"], pid=pid, pinned_gpu_pids=preloaded_gpu_pids,
                        continuing=continuing, memory_pressure=cache_memory_pressure(),
                    )
                    if offload:
                        kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(cache, "cpu")
                        preloaded_gpu_pids.discard(pid)
                    elif target_model.device.type == "cuda":
                        preloaded_gpu_pids.add(pid)

        # --------------------- 批量处理（Prefill / Verify） ----------------
        def handle_request_batch(batch: list[dict], return_responses: bool = False):
            """
            batch : 同一种 task_type (全部 prefill or 全部 verify)
            """
            kv_batch_mode = requested_kv_batch_mode
            proc_ids = [req["proc_id"] for req in batch]
            # Mark this batch's caches as active for the whole forward +
            # rollback window so the async prefetch worker never evicts or
            # moves a cache the main thread is currently using.  Activation
            # also bumps each pid's cache generation, invalidating any prefetch
            # copy that was snapshotted before this forward.
            residency.activate(proc_ids)
            try:
                started = time.perf_counter()
                result = _handle_request_batch_inner(batch, return_responses, kv_batch_mode, proc_ids)
                rows = [(int(req["prefix_len"]), int(req.get("_forward_tokens", req.get("gamma") or req["draft_output"].shape[1])))
                        for req in batch]
                with batch_cost_lock:
                    batch_cost.observe(batch[0]["task_type"], rows, time.perf_counter() - started)
                return result
            finally:
                residency.deactivate(proc_ids)

        def _handle_request_batch_inner(
            batch, return_responses, kv_batch_mode, proc_ids
        ):
            response_keys = {
                req["proc_id"]: req.get("response_key", req["proc_id"])
                for req in batch
            }
            collected_responses = {}

            def deliver(pid, payload):
                if return_responses:
                    collected_responses[pid] = payload
                else:
                    response_queues[response_keys.get(pid, pid)].put(payload)
            prefix_len = [req["prefix_len"] for req in batch]

            if batch[0]["task_type"] == "prefill":
                seqs = [req["draft_output"].to(target_model.device) for req in batch]
                input_lens = [x.shape[1] for x in seqs]
                continuation = [bool(req.get("_prefill_continuation", False)) for req in batch]
                # New prompts reset only the first chunk. Continuation chunks
                # append to the persistent cache instead of rebuilding it.
                with cache_lock:
                    for pid in proc_ids:
                        preloaded_gpu_pids.discard(pid)
                        if pid not in [req["proc_id"] for req, cont in zip(batch, continuation) if cont]:
                            kv_cache_manager.reset(pid)

                if kv_batch_mode == "varlen":
                    # Padding-free path: linear layers run on the concatenation
                    # of real tokens and attention runs per sequence. Fresh rows
                    # pass the full prompt as residual (cache was reset above);
                    # continuation rows pass only the new chunk.
                    with cache_lock:
                        for pid, cont in zip(proc_ids, continuation):
                            if cont:
                                cache = kv_cache_manager._past_key_values[pid]
                                kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(
                                    cache, target_model.device
                                )
                    residuals = []
                    for pid, cont, x in zip(proc_ids, continuation, seqs):
                        if cont:
                            cached_len = kv_cache_manager._past_key_values[pid].get_seq_length()
                            residuals.append(x[:, cached_len:])
                        else:
                            residuals.append(x)
                    varlen_generate(
                        kv_cache_manager,
                        target_model,
                        residuals,
                        proc_ids,
                        tokenizer.pad_token_id,
                        is_prefill=True,
                        sample_next=False,
                    )
                else:
                    # ---- original padded path (kept verbatim) ----
                    # 按最大长度 pad
                    max_T = max(x.shape[1] for x in seqs)
                    padded = []
                    for x in seqs:
                        pad_len = max_T - x.shape[1]
                        if pad_len:
                            pad = torch.full((1, pad_len), tokenizer.pad_token_id, device=x.device, dtype=x.dtype)
                            padded.append(torch.cat([x, pad], dim=1))
                        else:
                            padded.append(x)
                    x_batch = torch.cat(padded, dim=0)  # (B, max_T)

                    if any(continuation):
                        # A continuation batch may contain fresh and existing
                        # rows; forward_new_tokens routes them to the correct
                        # primitive while preserving per-row lengths.
                        with cache_lock:
                            for pid, cont in zip(proc_ids, continuation):
                                if cont:
                                    cache = kv_cache_manager._past_key_values[pid]
                                    kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(
                                        cache, target_model.device
                                    )
                        kv_cache_manager.forward_new_tokens(
                            x_batch,
                            proc_ids=proc_ids,
                            pad_token_id=tokenizer.pad_token_id,
                            input_lens=input_lens,
                            reset_pids=[pid for pid, cont in zip(proc_ids, continuation) if not cont],
                            sample_next=False,
                        )
                    else:
                        kv_cache_manager.generate(
                            x_batch,
                            1,
                            proc_ids=proc_ids,
                            pad_token_id=tokenizer.pad_token_id,
                            is_prefill=True,
                            input_lens=input_lens,
                            sample_next=False,
                        )

                self.color_print(f"process prefill tasks from: {proc_ids}", 3)
            else:  # verify 批量
                seqs = [req["draft_output"].to(target_model.device) for req in batch]
                debug_enabled = getattr(self.args, "debug_verify_tokens", False)
                debug_tail = 16

                # verify前，将KV cache移回GPU以供推理使用
                with cache_lock:
                    preloaded_gpu_pids.difference_update(proc_ids)
                    for req, pid in zip(batch, proc_ids):
                        if debug_enabled:
                            cached_len = kv_cache_manager._past_key_values[pid].get_seq_length()
                            req_prefix_len = req["prefix_len"]
                            recv_tokens = req["draft_output"][0].tolist()
                            self.color_print(
                                f"[VERIFY-CHECK][pid={pid}] cached_len={cached_len} "
                                f"prefix_len={req_prefix_len} same={cached_len == req_prefix_len}",
                                3,
                            )
                            self.color_print(
                                f"[VERIFY-CLOUD-RECV][pid={pid}] tail_ids={recv_tokens[-debug_tail:]} "
                                f"prefix_tail_ids={recv_tokens[max(0, req_prefix_len - debug_tail):req_prefix_len]}",
                                3,
                            )
                        cache = kv_cache_manager._past_key_values[pid]
                        kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(cache, target_model.device)

                verify_compute_start = time.perf_counter()
                gpu_start = gpu_end = None
                if target_model.device.type == "cuda":
                    gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    gpu_start.record()
                if kv_batch_mode == "varlen":
                    # Padding-free path: tail_only rows pass the received tail
                    # as residual; full-prefix rows pass only the uncached part.
                    residuals = []
                    for req, x in zip(batch, seqs):
                        pid = req["proc_id"]
                        cached_len = kv_cache_manager._past_key_values[pid].get_seq_length()
                        if req.get("tail_only", False):
                            residuals.append(x)
                        else:
                            residuals.append(x[:, cached_len:])
                    _ = varlen_generate(
                        kv_cache_manager,
                        target_model,
                        residuals,
                        proc_ids,
                        tokenizer.pad_token_id,
                        is_prefill=False,
                        sample_next=False,
                    )
                else:
                    # ---- original padded path (kept verbatim) ----
                    x_batch = []
                    verify_input_lens = []
                    for req, x in zip(batch, seqs):
                        # tail_only 模式下，edge 仅发送 [final_token + gamma_draft] 或 [gamma_draft]。
                        # 这里补一个虚拟前缀长度，让 kvcache 内部按 cached_len 正确截取 residual。
                        if req.get("tail_only", False):
                            pid = req["proc_id"]
                            cached_len = kv_cache_manager._past_key_values[pid].get_seq_length()
                            pad = torch.full(
                                (1, cached_len),
                                tokenizer.pad_token_id,
                                dtype=x.dtype,
                                device=x.device,
                            )
                            x_batch.append(torch.cat((pad, x), dim=1))
                            verify_input_lens.append(cached_len + x.shape[1])
                        else:
                            x_batch.append(x)
                            verify_input_lens.append(x.shape[1])
                    # The KV primitive accepts one dense tensor.  Padding is
                    # right-sided and excluded through ``verify_input_lens``;
                    # admission accounting still counts only real new tokens.
                    max_verify_T = max(x.shape[1] for x in x_batch)
                    padded_verify = []
                    for x in x_batch:
                        pad_len = max_verify_T - x.shape[1]
                        if pad_len:
                            pad = torch.full(
                                (1, pad_len),
                                tokenizer.pad_token_id,
                                dtype=x.dtype,
                                device=x.device,
                            )
                            padded_verify.append(torch.cat((x, pad), dim=1))
                        else:
                            padded_verify.append(x)
                    x_batch = torch.cat(padded_verify, dim=0)
                    _ = kv_cache_manager.generate(
                        x_batch,
                        1,
                        proc_ids=proc_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        is_prefill=False,
                        input_lens=verify_input_lens,
                        sample_next=False,
                    )
                verify_elapsed = time.perf_counter() - verify_compute_start
                verify_gpu_ms = None
                if gpu_end is not None:
                    gpu_end.record()
                    gpu_end.synchronize()
                    verify_gpu_ms = gpu_start.elapsed_time(gpu_end)
                    verify_elapsed = time.perf_counter() - verify_compute_start

            # ---------- 逐样本接受 / 采样 ---------------------------------
            probs_full = [kv_cache_manager._prob_history[pid] for pid in proc_ids]  # (1, *, V) each
            for idx, req in enumerate(batch):
                pid = req["proc_id"]
                prefix_len = req["prefix_len"]
                probs = probs_full[idx]  # (1, L, V)
                x = req["draft_output"].to(target_model.device)
                tail_only = req.get("tail_only", False) or req.get("_chunked_internal", False)
                has_bridge_token = req.get("has_bridge_token", False)
                req_gamma = min(
                    int(getattr(self.args, "max_tokens", 400) or 400),
                    max(1, _safe_int(req.get("gamma", self.args.gamma), default=self.args.gamma)),
                )

                if req["task_type"] == "prefill":
                    # prefill 仅初始化 cache，回滚到 prefix 长度供下一轮 verify 使用
                    kv_cache_manager.rollback(pid, prefix_len)
                    committed_prefix_tokens[pid] = req["draft_output"][0, :prefix_len].tolist()
                    if req.get("_prefill_final", True):
                        deliver(pid, {"status": "prefill_ok"})
                    continue

                # Verify either with the legacy strict-match rule or with the
                # exact sparse rejection sampler.  ``target_rows`` contains
                # one row per draft token plus the row used for an all-accept
                # correction token.
                verify_method = str(
                    req.get("verify_method", getattr(self.args, "verify_method", "rejection"))
                ).lower()
                mismatch_pos = None
                if tail_only:
                    tail_offset = 1 if has_bridge_token else 0
                    draft_tokens = x[:, tail_offset:tail_offset + req_gamma]
                else:
                    draft_tokens = x[:, prefix_len:prefix_len + req_gamma]
                target_rows = torch.cat(
                    [
                        kv_cache_manager.probability_rows(pid, verify_logit_position(prefix_len, i), verify_logit_position(prefix_len, i) + 1)[:, 0, :self.vocab_size]
                        for i in range(req_gamma)
                    ]
                    + [probs[:, -1, :self.vocab_size]],
                    dim=0,
                ).float()

                debug_enabled = getattr(self.args, "debug_verify_tokens", False)
                debug_max_steps = max(0, int(getattr(self.args, "debug_max_print_steps", 8)))
                if debug_enabled:
                    if pid in committed_prefix_tokens:
                        ctx_ids = committed_prefix_tokens[pid][max(0, len(committed_prefix_tokens[pid]) - 20):]
                        ctx_src = "committed_prefix_tail20"
                    else:
                        start = max(0, prefix_len - 20)
                        ctx_ids = x[0, start:prefix_len].tolist()
                        ctx_src = "request_prefix_tail20_fallback"
                    self.color_print(
                        f"[VERIFY-CONTEXT][pid={pid}] src={ctx_src} "
                        f"prefix_len={prefix_len} ctx_ids={ctx_ids} ctx_text={ids_text_repr(ctx_ids)}",
                        3,
                    )
                if verify_method == "rejection":
                    sparse_ids, sparse_probs = decode_sparse_block(
                        req.get("draft_prob_block"),
                        device=target_model.device,
                        vocab_size=self.vocab_size,
                    )
                    accepted_cnt, new_token = rejection_sample(
                        draft_tokens,
                        target_rows,
                        sparse_ids,
                        sparse_probs,
                        generator=_sampling_generator(
                            pid, req.get("sampling_seed", getattr(self, "seed", 0))
                        ),
                        sample_correction=not bool(req.get("_internal_chunk_nonfinal", False)),
                    )
                    if accepted_cnt < req_gamma:
                        mismatch_pos = prefix_len + accepted_cnt
                elif verify_method == "greedy":
                    accepted_cnt = greedy_accept_count(draft_tokens, target_rows[:-1])
                    new_token = (
                        torch.argmax(target_rows[accepted_cnt], dim=-1).reshape(1)
                        if accepted_cnt < req_gamma
                        or not bool(req.get("_internal_chunk_nonfinal", False))
                        else None
                    )
                    if accepted_cnt < req_gamma:
                        mismatch_pos = prefix_len + accepted_cnt
                else:
                    raise ValueError(f"unknown verify_method: {verify_method!r}")

                if debug_enabled:
                    for i in range(min(req_gamma, debug_max_steps)):
                        draft_token_id = int(draft_tokens[0, i].item())
                        target_token_id = int(torch.argmax(target_rows[i]).item())
                        match_flag = "MATCH" if draft_token_id == target_token_id else "MISMATCH"
                        self.color_print(
                            f"[VERIFY][pid={pid}] step={i} pos={prefix_len + i} "
                            f"draft={token_repr(draft_token_id)} "
                            f"target={token_repr(target_token_id)} {match_flag}",
                            6,
                        )

                accepted_len = prefix_len + accepted_cnt
                if new_token is not None:
                    new_token = new_token.reshape(1, 1)

                # Keep only accepted prefix in target cache.
                # final_token will be consumed in the next verify call together with new draft tokens.
                kv_cache_manager.rollback(pid, accepted_len)

                # 更新统计并回传
                with accept_stats_lock:
                    accept_stats[pid][0] += accepted_len - prefix_len
                    accept_stats[pid][1] += req_gamma
                if pid in committed_prefix_tokens:
                    if tail_only:
                        req_tokens = x[0].tolist()
                        draft_start_idx = 1 if has_bridge_token else 0
                        accepted_draft_tokens = req_tokens[draft_start_idx:draft_start_idx + accepted_cnt]
                        committed_prefix_tokens[pid].extend(accepted_draft_tokens)
                    if new_token is not None:
                        committed_prefix_tokens[pid].append(int(new_token.item()))
                else:
                    committed_prefix_tokens[pid] = x[0, :accepted_len].tolist()
                    if new_token is not None:
                        committed_prefix_tokens[pid].append(int(new_token.item()))
                if debug_enabled:
                    accepted_cnt = accepted_len - prefix_len
                    mismatch_info = mismatch_pos if mismatch_pos is not None else "none"
                    self.color_print(
                        f"[VERIFY][pid={pid}] prefix_len={prefix_len} accepted={accepted_cnt}/{req_gamma} "
                        f"accepted_len={accepted_len} mismatch_pos={mismatch_info} "
                        f"new_token={token_repr(int(new_token.item())) if new_token is not None else '<deferred>'}",
                        2,
                    )

                response_payload = {
                    "accepted": accepted_len,
                    "final_token": int(new_token.item()) if new_token is not None else None,
                    "verify_ms": verify_elapsed * 1000.0,
                    "verify_gpu_ms": verify_gpu_ms,
                }
                if getattr(self.args, "enable_pipeline", True) and getattr(self.args, "pipeline_gamma_adapt", True):
                    # Target: edge 在等待 verify 的完整往返窗口
                    # （T_verify + RTT）内持续 proactive draft，所以
                    # T_edge_draft(gamma) ~= T_verify + RTT。
                    # Use online EMA to suggest per-session gamma for next round.
                    # All request floats are sanitized first: JSON can carry
                    # inf/NaN, and int(round(inf)) would crash the target
                    # worker (remotely triggerable DoS).
                    logical_gamma = max(1, _safe_int(req.get("logical_gamma", req_gamma), req_gamma))
                    drafted_tokens = logical_gamma
                    edge_per_tok = _safe_float(req.get("local_decode_per_token_s"), 0.0)
                    if edge_per_tok <= 0:
                        edge_per_tok = _safe_float(req.get("lag", 0.0)) / drafted_tokens
                    avg_cloud_total_ms = max(
                        0.0, _safe_float(req.get("avg_cloud_total_ms", 0.0))
                    )
                    # Use global average cloud_total_ms across all tasks when available.
                    # Fallback to local verify elapsed for warmup.
                    verify_budget = (
                        avg_cloud_total_ms / 1000.0 if avg_cloud_total_ms > 0.0 else verify_elapsed
                    )
                    transport_rtt = max(
                        0.0,
                        _safe_float(req.get("transport_rtt", req.get("edge_rtt", 0.0))),
                    )
                    v_ema = update_ema(verify_time_ema, pid, max(1e-6, verify_budget))
                    rtt_ema = update_ema(edge_rtt_ema, pid, max(0.0, transport_rtt))
                    if edge_per_tok > 0:
                        d_ema = update_ema(edge_draft_per_token_ema, pid, edge_per_tok)
                    else:
                        # Edge reported lag=0: keep the previous estimate instead
                        # of seeding d_ema with 1e-6, which would pin gamma at max.
                        d_ema = edge_draft_per_token_ema.get(pid)
                    # 目标：让 edge 在等待 verify 的完整往返窗口（verify 时间 +
                    # 传输 RTT）内恰好 draft 出 gamma 个 token。旧公式
                    # (v_ema - rtt_ema) / d_ema 在 rtt_ema > v_ema 时恒为负、
                    # 被 min_gamma 兜底成 1（r5 实验 TPOT 恶化 44% 的根因）。
                    # 新公式恒为正，RTT 大时 gamma 随之增大以填满等待窗口。
                    min_gamma = int(getattr(self.args, "pipeline_gamma_min", 1))
                    max_gamma = int(getattr(self.args, "pipeline_gamma_max", 16))
                    if d_ema is None or d_ema <= 0:
                        suggested = int(getattr(self.args, "gamma", 4) or 4)
                        draft_window = 0.0  # debug log below must stay defined
                    else:
                        draft_window = v_ema + rtt_ema
                        # Clamp before int(): even sanitized inputs can push the
                        # quotient past int range via extreme EMA values.
                        quotient = min(1e9, max(0.0, draft_window / d_ema))
                        suggested = int(round(quotient))
                    suggested = max(min_gamma, min(max_gamma, suggested))
                    gamma_step = max(1, int(getattr(self.args, "pipeline_gamma_step", 2)))
                    suggested = max(logical_gamma - gamma_step, min(logical_gamma + gamma_step, suggested))
                    suggested = max(min_gamma, min(max_gamma, suggested))
                    response_payload["suggested_gamma"] = suggested
                    if getattr(self.args, "debug_pipeline", False):
                        self.color_print(
                            f"[PIPELINE-CLOUD][pid={pid}] req_gamma={req_gamma} "
                            f"accepted={accepted_cnt}/{req_gamma} verify={verify_elapsed*1000:.2f}ms "
                            f"avg_cloud_total={avg_cloud_total_ms:.2f}ms "
                            f"transport_rtt={transport_rtt*1000:.2f}ms v_ema={v_ema*1000:.2f}ms "
                            f"d_ema={d_ema*1000:.4f}ms/tok rtt_ema={rtt_ema*1000:.2f}ms "
                            f"draft_window={draft_window*1000:.2f}ms suggested_gamma={suggested}",
                            3,
                        )

                deliver(pid, response_payload)

            finish_cache_phase(batch)
            return collected_responses if return_responses else None

        def accept_stats_snapshot():
            with accept_stats_lock:
                return {
                    pid: (values[0], values[1])
                    for pid, values in accept_stats.items()
                }

        def _sort_queues_if_dirty():
            # Wait times use the cloud monotonic clock; only re-sort when
            # ingress/completion actually changed the queues.
            task_queues.sort(
                accept_stats_snapshot(),
                now=time.monotonic(),
                category_for=lambda item: _length_category(
                    item.remaining_tokens, len_r1, len_r2
                ),
                priority=lambda request, stats, now: _compute_priority_score(
                    request, stats, now=now
                ),
            )

        # ---- 异步 KV 预取：后台线程执行 CPU<->GPU 搬移 ----
        # The control owner only computes the latest residency intent; the
        # daemon worker performs the copies.  Execution re-derives
        # stale/load sets and refuses to touch active or reserved pids.
        prefetch_enabled = os.environ.get(
            "FASTSD_DISABLE_TARGET_CACHE_OFFLOAD", ""
        ).lower() not in {"1", "true", "yes", "on"}

        def _cache_get(pid):
            return kv_cache_manager._past_key_values.get(pid)

        def _cache_set(pid, cache):
            kv_cache_manager._past_key_values[pid] = cache

        def _bind_target_device():
            # CUDA's current device is per-thread; executor/prefetch threads
            # must bind it explicitly (CPU returns without touching CUDA).
            set_thread_device(target_model.device)

        prefetch_worker = KVPrefetchWorker(
            residency=residency,
            get_cache=_cache_get,
            set_cache=_cache_set,
            move_cache=move_dynamic_cache_to,
            gpu_device=target_model.device,
            metrics=scheduler_metrics,
            on_error=lambda message: self.color_print(
                f"[KV-PREFETCH-FAILED] {message}", 2
            ),
            thread_setup=_bind_target_device,
        )
        prefetch_coordinator = PrefetchCoordinator(
            submit=prefetch_worker.submit,
            resident_snapshot=residency.resident_snapshot,
            protected_snapshot=residency.protected_snapshot,
            metrics=scheduler_metrics,
            pending_snapshot=prefetch_worker.has_pending,
        )

        def _slice_request(item: WorkItem, sl):
            req = dict(item.request)
            source = req["draft_output"]
            req["_forward_tokens"] = sl.forward_token_count
            if item.task_type == "prefill":
                end = sl.offset + sl.draft_token_count
                req["draft_output"] = source[:, :end]
                req["prefix_len"] = end
                req["_prefill_continuation"] = sl.offset > 0
                req["_prefill_final"] = end >= item.total_tokens
                return req

            req_gamma = sl.draft_token_count
            base_prefix = item.base_prefix_len + item.accepted_so_far
            tail_only = bool(req.get("tail_only", False))
            bridge_token = getattr(item, "internal_bridge_token", None)
            # Internally every chunk is represented as a tail-only residual:
            # this gives the executor one unambiguous bridge-token convention
            # even when the external request used full-prefix mode.
            original_bridge = bool(req.get("has_bridge_token", False))
            draft_start = 1 if original_bridge else 0
            if original_bridge:
                draft_base = 1
            elif req.get("tail_only", False):
                # Pipeline Edge sends only gamma drafts on the first round;
                # the logical prefix lives in the target KV cache, not in
                # this HTTP payload.
                draft_base = 0
            else:
                draft_base = item.base_prefix_len
            draft_tokens = source[:, draft_base + sl.offset:draft_base + sl.offset + req_gamma]
            if str(req.get("verify_method", getattr(self.args, "verify_method", "rejection"))).lower() == "rejection":
                block = req.get("draft_prob_block")
                if not block:
                    raise ValueError("rejection verify request is missing draft_prob_block")
                req["draft_prob_block"] = slice_sparse_block(block, sl.offset, req_gamma)
            if bridge_token is None and original_bridge and sl.offset == 0:
                # The first internal slice must carry the bridge supplied by
                # Edge.  Subsequent slices use the correction token returned
                # by the preceding target forward.
                bridge_token = int(source[0, 0].item())
            elif bridge_token is None and item.bridge_pending and sl.offset == 0:
                if int(item.bridge_pending) != 1:
                    raise ValueError(
                        f"unsupported full-prefix bridge count: {item.bridge_pending}"
                    )
                bridge_token = int(source[0, item.base_prefix_len - 1].item())
            if bridge_token is not None:
                bridge = torch.tensor([[int(bridge_token)]], dtype=source.dtype, device=source.device)
                req["draft_output"] = torch.cat((bridge, draft_tokens), dim=1)
                req["has_bridge_token"] = True
            else:
                req["draft_output"] = draft_tokens
                req["has_bridge_token"] = False
            req["prefix_len"] = base_prefix
            req["logical_gamma"] = int(item.total_tokens)
            req["gamma"] = req_gamma
            req["tail_only"] = True
            req["_chunked_internal"] = True
            req["_internal_chunk_nonfinal"] = (
                sl.offset + req_gamma < int(item.total_tokens)
            )
            return req

        def candidate_value(item, selected):
            same = [s for s in selected if s.task_type == item.task_type]
            rows = [(work_items[s.work_id].base_prefix_len, s.forward_token_count)
                    for s in same if s.work_id in work_items]
            remaining_budget = max(1, int(self.args.token_budget) - sum(s.forward_token_count for s in selected))
            tokens = min(item.remaining_tokens, remaining_budget)
            if item.task_type == "prefill":
                tokens = min(tokens, int(self.args.prefill_chunk_quantum))
            context = item.base_prefix_len + item.accepted_so_far if item.task_type == "verify" else item.cursor
            with batch_cost_lock:
                marginal = max(1e-6, batch_cost.estimate(item.task_type, rows + [(context, tokens)])
                              - batch_cost.estimate(item.task_type, rows))
            with cache_lock:
                cache = kv_cache_manager._past_key_values.get(item.proc_id)
                legacy = cache.to_legacy_cache() if cache is not None else ()
                if legacy and legacy[0][0].device != target_model.device:
                    nbytes = sum(t.numel() * t.element_size() for layer in legacy for t in layer)
                    marginal += nbytes / max(1.0, float(getattr(self.args, "kv_transfer_gbps", 10.0)) * 1e9)
            if item.task_type == "prefill":
                # Prefill progresses toward a first token; it does not produce
                # tokens immediately like Verify. Favor short remaining work
                # while the planner enforces its wait/starvation bound.
                wait = max(0.0, time.monotonic() - item.request.get("server_enqueue_monotonic", time.monotonic()))
                return (1.0 + wait) / max(1e-6, marginal * item.remaining_tokens / max(1, tokens))
            profile = progress_profiles.get(item.proc_id, AcceptanceProfile())
            return profile.expected_progress(tokens) / marginal

        def build_plan(reserved_work_ids=(), cycle=None, cursor=None, projected_plan=None):
            _sort_queues_if_dirty()
            return plan_iteration(
                project_after_plan(task_queues.snapshot(), projected_plan) if projected_plan is not None else task_queues.snapshot(),
                token_budget=int(self.args.token_budget),
                current_cycle=(
                    scheduler_state["current_cycle"] if cycle is None else cycle
                ),
                wrr_cursor=scheduler_state["wrr_cursor"] if cursor is None else cursor,
                max_num_seqs=int(self.args.batch_size),
                min_prefill_chunk_tokens=int(self.args.min_prefill_chunk_tokens),
                prefill_chunk_quantum=int(self.args.prefill_chunk_quantum),
                accept_stats=accept_stats_snapshot(),
                now=time.monotonic(),
                reserved_work_ids=reserved_work_ids,
                selection_policy=getattr(self.args, "scheduler_policy", "legacy"),
                candidate_value=candidate_value,
                max_wait_s=float(getattr(self.args, "scheduler_max_wait_s", 1.0)),
            )

        def refresh_prefetch():
            """Recompute the lookahead residency intent from a stable snapshot.

            Prediction starts from the post-plan cursor/cycle and a copy of
            the projected residual work. Partially completed requests remain
            candidates; the residency layer still protects active caches.
            """

            if not prefetch_enabled or not getattr(self.args, "kv_prefetch", True):
                return
            plan = in_flight["plan"]
            if plan is None and (executor is None or not executor.busy()):
                # A forward is not running: the next plan will be admitted
                # momentarily.  Prefetching it here would race the reservation
                # and be invalidated, so wait until a forward is in flight.
                return
            if plan is not None:
                lookahead = build_plan(
                    projected_plan=plan,
                    cycle=plan.next_cycle,
                    cursor=plan.next_wrr_cursor,
                )
            else:
                lookahead = build_plan()
            if not lookahead.verify_proc_ids and not task_queues.has_live_task("verify"):
                scheduler_metrics["prefetch_no_verify_candidates"] += 1
            prefetch_coordinator.refresh(lookahead.verify_proc_ids)

        def submit_plan(plan):
            """Atomically admit the whole plan, then hand it to the executor."""

            item_by_id = {item.work_id: item for item in plan._selected_items}
            verify_batch = [
                _slice_request(item_by_id[sl.work_id], sl) for sl in plan.verify_slices
            ]
            prefill_batch = [
                _slice_request(item_by_id[sl.work_id], sl) for sl in plan.prefill_slices
            ]
            reserved_pids = {sl.proc_id for sl in plan.verify_slices} | {
                sl.proc_id for sl in plan.prefill_slices
            }
            reserve_admission_plan(plan)
            # Protect pids for the entire admitted-but-not-yet-forwarded
            # window, not just while handle_request_batch marks them active.
            residency.reserve(reserved_pids)
            job = _ExecutionJob(
                plan=plan,
                item_by_id=item_by_id,
                verify_batch=verify_batch,
                prefill_batch=prefill_batch,
                reserved_pids=reserved_pids,
            )
            in_flight["plan"] = plan
            scheduler_metrics["iterations"] += 1
            try:
                executor.submit(job)
            except BaseException:
                in_flight["plan"] = None
                residency.release(reserved_pids)
                abort_admission_plan(plan)
                raise

        # The Verify->Prefill phase protocol and the no_grad/energy wrapper
        # live in scheduler_runtime so tests exercise the production executor.
        run_job = make_forward_job_runner(
            make_forward_batch_runner(
                lambda batch: handle_request_batch(batch, return_responses=True),
                on_batch_start=(
                    energy_service.enter_active if energy_service is not None else None
                ),
                on_batch_end=(
                    energy_service.exit_active if energy_service is not None else None
                ),
            )
        )

        def _retire_finished(job, completed):
            finished = set(completed)
            for item in job.item_by_id.values():
                if (
                    item.finished
                    or item.remaining_tokens <= 0
                    or getattr(item, "cancel_pending", False)
                ):
                    finished.add(item.work_id)
            removed = task_queues.remove_work_ids(finished)
            for work_id in finished:
                item = work_items.pop(work_id, None)
                if item is not None:
                    lifecycle.touch(item.proc_id)
            scheduler_metrics["finished_work_items_removed"] += removed
            # Also drop any item that terminated outside this plan.
            task_queues.prune_finished()

        def finish_logical_verify(item, response):
            profile = progress_profiles.setdefault(item.proc_id, AcceptanceProfile())
            profile.observe(item.accepted_so_far, item.total_tokens)
            response["expected_progress"] = profile.expected_progress(item.total_tokens)
            response["verify_ms"] = item.verify_ms_total
            if getattr(item, "verify_gpu_ms_total", None) is not None:
                response["verify_gpu_ms"] = item.verify_gpu_ms_total
            if (getattr(self.args, "pipeline_gamma_adapt", True)
                    and getattr(self.args, "enable_pipeline", True)
                    and getattr(self.args, "gamma_policy", "progress") == "progress"):
                service = item.verify_ms_total / 1000.0
                elapsed = max(0.0, time.monotonic() - item.request.get("server_enqueue_monotonic", time.monotonic()))
                response["suggested_gamma"] = choose_gamma(
                    profile, current=item.total_tokens,
                    draft_s=draft_seconds_per_token(item.request), verify_s=service,
                    rtt_s=max(0.0, _safe_float(item.request.get("transport_rtt", 0.0))),
                    queue_s=max(0.0, elapsed - service),
                    minimum=int(getattr(self.args, "pipeline_gamma_min", 1)),
                    maximum=int(getattr(self.args, "pipeline_gamma_max", 16)),
                    step=int(getattr(self.args, "pipeline_gamma_step", 2)),
                )
            return response

        def _apply_verify_phase(job, verify_responses):
            """Commit Verify progress and return responses to callers."""

            if job.verify_applied:
                return
            plan = job.plan
            for sl, sliced_req in zip(plan.verify_slices, job.verify_batch):
                item = job.item_by_id[sl.work_id]
                if getattr(item, "cancel_pending", False):
                    item.finished = True
                    job.completed.add(item.work_id)
                    continue
                response = verify_responses.get(item.proc_id)
                if response is None:
                    raise RuntimeError(f"missing verify response for {item.work_id}")
                item.verify_ms_total = getattr(item, "verify_ms_total", 0.0) + float(response.get("verify_ms", 0.0))
                if response.get("verify_gpu_ms") is not None:
                    item.verify_gpu_ms_total = getattr(item, "verify_gpu_ms_total", 0.0) + float(response["verify_gpu_ms"])
                physical_prefix = int(sliced_req["prefix_len"])
                accepted = int(response["accepted"])
                accepted_chunk = max(0, accepted - physical_prefix)
                current_prefix = item.base_prefix_len + item.accepted_so_far
                item.cursor += accepted_chunk
                item.accepted_so_far += accepted_chunk
                job.progress[item.work_id] = accepted_chunk
                if accepted_chunk < sl.draft_token_count:
                    item.finished = True
                    job.completed.add(item.work_id)
                    response = dict(response)
                    response["accepted"] = item.base_prefix_len + item.accepted_so_far
                    response = finish_logical_verify(item, response)
                    response_queues[item.response_key or item.proc_id].put(response)
                elif item.cursor >= item.total_tokens:
                    item.finished = True
                    job.completed.add(item.work_id)
                    response = dict(response)
                    response["accepted"] = item.base_prefix_len + item.accepted_so_far
                    response = finish_logical_verify(item, response)
                    response_queues[item.response_key or item.proc_id].put(response)
                else:
                    # An all-accepted internal chunk is still one logical
                    # speculative round.  Do not insert the target's
                    # provisional next token between d_k and d_{k+1};
                    # the correction/final token is used only when the
                    # complete round finishes or a mismatch occurs.
                    item.internal_bridge_token = None
                    item.bridge_pending = 0
                    if item.proc_id in committed_prefix_tokens:
                        committed_prefix_tokens[item.proc_id] = committed_prefix_tokens[item.proc_id][:current_prefix + accepted_chunk]
                    scheduler_metrics["partial_verify"] += 1
            job.verify_applied = True

        def _apply_prefill_phase(job, prefill_responses):
            plan = job.plan
            for sl in plan.prefill_slices:
                item = job.item_by_id[sl.work_id]
                item.cursor = sl.offset + sl.draft_token_count
                scheduler_metrics["timing_ready_prefill_forwards"] += 1
                if getattr(item, "cancel_pending", False):
                    # The cancel arrived after admission; the forward is not
                    # interruptible, but its response is suppressed.
                    item.finished = True
                    item.state = "finished"
                    job.completed.add(item.work_id)
                    continue
                if item.cursor >= item.total_tokens:
                    item.finished = True
                    job.completed.add(item.work_id)
                    response = prefill_responses.get(item.proc_id)
                    if response is None:
                        raise RuntimeError(f"missing prefill response for {item.work_id}")
                    response_queues[item.response_key or item.proc_id].put(response)
                else:
                    scheduler_metrics["partial_prefill"] += 1
            for sl in plan.prefill_slices:
                job.progress[sl.work_id] = sl.draft_token_count

        def apply_outcome(outcome):
            """Control-owner body: apply one phase exactly once.

            A Verify-only intermediate phase keeps the plan in flight; the
            final phase commits progress, advances WR state and retires
            terminated WorkItems.
            """

            job = outcome.job
            plan = job.plan
            committed = False
            try:
                if outcome.error is not None:
                    abort_admission_plan(plan)
                    raise outcome.error

                if outcome.phase == "verify":
                    # Verify always runs before Prefill, even when one plan
                    # admits both types.  Each WorkItem appears at most once.
                    _apply_verify_phase(job, outcome.verify_responses)
                    return

                if not job.verify_applied and outcome.verify_responses:
                    _apply_verify_phase(job, outcome.verify_responses)
                _apply_prefill_phase(job, outcome.prefill_responses)
                commit_admission_progress(
                    plan,
                    task_queues,
                    commit_admission_plan,
                    completed_work_ids=job.completed,
                    progress=job.progress,
                )
                committed = True

                cycle_delta = plan.next_cycle - scheduler_state["current_cycle"]
                if cycle_delta > 0:
                    selected_ids = set(plan.selected_work_ids)
                    for queued_item in work_items.values():
                        if (
                            queued_item.task_type == "prefill"
                            and queued_item.work_id not in selected_ids
                            and not queued_item.finished
                        ):
                            queued_item.missed_cycles += cycle_delta
                scheduler_state["wrr_cursor"] = plan.next_wrr_cursor
                scheduler_state["current_cycle"] = plan.next_cycle
                scheduler_metrics["plans"] += 1
                scheduler_metrics["used_tokens"] += plan.used_tokens
                scheduler_metrics["verify_slices"] += len(plan.verify_slices)
                scheduler_metrics["prefill_slices"] += len(plan.prefill_slices)
                _retire_finished(job, job.completed)
            except BaseException:
                if not committed:
                    abort_admission_plan(plan)
                # An intermediate-stage failure still ends this plan.
                in_flight["plan"] = None
                residency.release(job.reserved_pids)
                raise
            finally:
                if outcome.final:
                    in_flight["plan"] = None
                    residency.release(job.reserved_pids)

        # ---------------------- 主循环 -------------------------------
        sched_mode = getattr(self.args, "server_sched_mode", "fastsd")
        use_strict_fcfs = sched_mode in {"pipeline", "vanilla"}

        def tensorize_draft_output(req: dict) -> dict:
            draft_output = req.get("draft_output")
            if not torch.is_tensor(draft_output):
                draft_output = torch.tensor(draft_output, dtype=torch.long)
                if draft_output.dim() == 1:
                    draft_output = draft_output.unsqueeze(0)
                if draft_output.dim() != 2:
                    raise ValueError(
                        "draft_output must be a flat token list or a 2-D tensor"
                    )
                req["draft_output"] = draft_output
            return req

        def canonicalize_ingress(req: dict):
            """Validate one queue message without allowing worker termination."""
            try:
                req = tensorize_draft_output(req)
                req = canonicalize_request(
                    req,
                    max_tokens=int(getattr(self.args, "max_tokens", 400) or 400),
                )
                if req.get("task_type") == "verify" and req.get("verify_method") == "rejection":
                    sparse_ids, sparse_probs = decode_sparse_block(
                        req.get("draft_prob_block"),
                        device="cpu",
                        vocab_size=self.vocab_size,
                    )
                    gamma = int(req["gamma"])
                    if int(sparse_ids.shape[0]) != gamma:
                        raise ValueError(
                            "draft probability block row count must equal gamma"
                        )
                    draft_output = req["draft_output"]
                    if req.get("tail_only", False):
                        offset = 1 if req.get("has_bridge_token", False) else 0
                        draft_tokens = draft_output[:, offset:offset + gamma]
                    else:
                        draft_tokens = draft_output[:, req["prefix_len"]:req["prefix_len"] + gamma]
                    validate_candidate_support(draft_tokens, sparse_ids, sparse_probs)
                return req
            except (TypeError, ValueError, KeyError) as exc:
                response_key = req.get("response_key", req.get("proc_id")) if isinstance(req, dict) else None
                response_queue = None
                if response_key is not None:
                    try:
                        response_queue = response_queues[response_key]
                    except (KeyError, TypeError, AttributeError):
                        response_queue = None
                if response_queue is not None:
                    response_queue.put({"error": str(exc)})
                self.color_print(f"[REQUEST-REJECTED] {exc}", 2)
                return None

        def process_ingress_message(req) -> None:
            """Apply one inbound request/control on the control-owner thread."""

            nonlocal len_r1, len_r2
            if isinstance(req, dict) and req.get("control_type") == "session_release":
                lifecycle.release(req.get("proc_id"), req.get("response_key"))
                flush_session_releases()
                return
            if isinstance(req, dict):
                lifecycle.touch(req.get("proc_id"))
            if isinstance(req, dict) and req.get("control_type") == "prefill_timing":
                try:
                    apply_prefill_timing(req)
                except (TypeError, ValueError, KeyError) as exc:
                    self.color_print(f"[TIMING-REJECTED] {exc}", 2)
                return
            if isinstance(req, dict) and req.get("control_type") == "prefill_cancel":
                try:
                    apply_prefill_cancel(req)
                    lifecycle.release(req.get("proc_id"))
                except (TypeError, ValueError, KeyError) as exc:
                    self.color_print(f"[PREFILL-CANCEL-REJECTED] {exc}", 2)
                return
            if (
                isinstance(req, dict)
                and req.get("task_type") == "prefill"
                and req.get("proc_id") in pending_prefill_cancel
            ):
                # A cancel control may arrive before the initial /prefill
                # request reaches worker ingress.
                pending_prefill_cancel.discard(req.get("proc_id"))
                return
            req = canonicalize_ingress(req)
            if req is None:
                return
            if req["task_type"] == "verify" and req["proc_id"] not in kv_cache_manager._past_key_values:
                response_queues[req.get("response_key", req["proc_id"])].put({"error": "session has no target cache; prefill required"})
                return
            recent_prefix_lens.append(req["prefix_len"])
            len_r1, len_r2 = _update_length_thresholds(recent_prefix_lens)
            cat = _length_category(req["prefix_len"], len_r1, len_r2)
            work_id = str(req.get("work_id", f"{req['proc_id']}:{time.monotonic_ns()}"))
            req["work_id"] = work_id
            req["response_key"] = req.get("response_key", req["proc_id"])
            # FastSD arrival-gated Prefill records its cloud arrival before the
            # token upload.  Keep that timestamp as the scheduler wait origin;
            # falling back to worker ingress preserves the legacy request path.
            arrival_monotonic = req.get("arrival_monotonic_s")
            try:
                arrival_monotonic = float(arrival_monotonic)
            except (TypeError, ValueError, OverflowError):
                arrival_monotonic = None
            if arrival_monotonic is None or not math.isfinite(arrival_monotonic):
                arrival_monotonic = time.monotonic()
            req["server_enqueue_monotonic"] = arrival_monotonic
            # Priority scores use wall-clock ``current_time``; keep the
            # monotonic timestamp separately for queue latency diagnostics.
            req["current_time"] = time.time()
            if req["task_type"] == "verify" and not req.get("tail_only", False):
                with cache_lock:
                    cached_len = kv_cache_manager._past_key_values[
                        req["proc_id"]
                    ].get_seq_length()
                req["bridge_tokens"] = full_prefix_bridge_tokens(
                    req["prefix_len"], cached_len
                )
            item = WorkItem.from_request(
                req,
                category=cat,
                cycle=scheduler_state["current_cycle"],
                work_id=work_id,
            )
            update = pending_prefill_timing.pop(req["proc_id"], None)
            if update is not None:
                req = dict(req)
                req.update(update)
                item.request = req
            work_items[work_id] = item
            task_queues.enqueue(item)

        def poll_ingress(timeout):
            """Drain bounded ingress; ``timeout=None`` is an event wait.

            This runs on the control owner even while the executor is inside
            a forward, which is what makes newly arrived verify requests
            visible to lookahead prefetch before the current batch finishes.
            """

            changed = 0
            limit = max(1, int(self.args.batch_size) * 10)
            next_timeout = timeout
            while changed < limit:
                try:
                    if next_timeout is None:
                        req = request_queue.get()
                    elif next_timeout <= 0:
                        req = request_queue.get_nowait()
                    else:
                        req = request_queue.get(timeout=next_timeout)
                except queue.Empty:
                    break
                next_timeout = 0
                if req is None:  # 终止信号
                    raise ShutdownRequested
                changed += 1
                if executor is not None and executor.busy():
                    scheduler_metrics["incoming_during_forward"] += 1
                process_ingress_message(req)
            return changed

        def terminate_pending_work() -> int:
            """Explicitly finish accepted work that can never run at shutdown."""

            removed = terminate_pending_work_items(
                work_items=work_items,
                queues=task_queues,
                response_queues=response_queues,
                log=lambda message: self.color_print(message, 2),
            )
            pending_prefill_timing.clear()
            pending_prefill_cancel.clear()
            return removed

        # Pipeline baseline: strict FCFS, single-request handling only.
        # No queue categorization, no batching, no preload, no priority scheduling.
        executor = None
        ingress_relay = None
        wake_event = threading.Event()
        try:
            if use_strict_fcfs:
                while True:
                    try:
                        req = request_queue.get(timeout=0.01)
                        if req is None:
                            return
                        if isinstance(req, dict) and req.get("control_type") == "prefill_timing":
                            # A timing update is meaningful only to FastSD;
                            # baselines never send this control message.
                            continue
                        if isinstance(req, dict) and req.get("control_type") == "session_release":
                            lifecycle.release(req.get("proc_id"), req.get("response_key"))
                            flush_session_releases()
                            continue
                        if isinstance(req, dict):
                            lifecycle.touch(req.get("proc_id"))
                        req = canonicalize_ingress(req)
                        if req is None:
                            continue
                        if req["task_type"] == "verify" and req["proc_id"] not in kv_cache_manager._past_key_values:
                            response_queues[req.get("response_key", req["proc_id"])].put({"error": "session has no target cache; prefill required"})
                            continue
                        if energy_service is not None:
                            energy_service.enter_active()
                        try:
                            handle_request_batch([req])
                        finally:
                            if energy_service is not None:
                                energy_service.exit_active()
                    except queue.Empty:
                        flush_session_releases()
                return

            # FastSD: one control owner (this thread) continuously accepts
            # ingress, one executor thread runs the single model forward path,
            # and one worker thread moves KV caches.  The control owner spends
            # the forward window updating queues and prefetch intent instead of
            # being blocked inside the model call.
            ingress_relay = IngressRelay(request_queue, wake_event)
            request_queue = ingress_relay.queue
            ingress_relay.start()
            executor = SinglePlanExecutor(
                run_job,
                name="fastsd-target-executor",
                thread_setup=_bind_target_device,
                wake_event=wake_event,
            )
            executor.start()
            prefetch_worker.start()
            runtime = TargetSchedulerRuntime(
                executor=executor,
                queues=task_queues,
                poll_ingress=poll_ingress,
                build_plan=build_plan,
                submit_plan=submit_plan,
                apply_outcome=apply_outcome,
                refresh_prefetch=refresh_prefetch,
                metrics=scheduler_metrics,
                terminate_unrunnable=terminate_pending_work,
                wake_event=wake_event,
                maintenance=flush_session_releases,
            )
            runtime.run()
        finally:
            # Stop worker threads before declaring cleanup complete.  A model
            # forward and a cache copy are not interruptible, so wait for the
            # actual call to return rather than releasing its cache leases
            # after a forced timeout.  Normal shutdown drains first; a fatal
            # stop only skips subsequent phases, so the wait is bounded by
            # the current (already-running) call.
            if executor is not None:
                executor.stop()
                executor.join()
            prefetch_worker.stop()
            prefetch_worker.join()
            if ingress_relay is not None:
                ingress_relay.close()
            for pid in list(lifecycle.last_seen):
                lifecycle.release(pid)
            flush_session_releases()
            transfers.close()
            scheduler_metrics["cache_transfers"] = dict(transfers.metrics)
            scheduler_metrics["batch_cost_profiles"] = {str(k): v for k, v in batch_cost.table.items()}
            if energy_service is not None:
                energy_service.shutdown()

            # Preserve scheduler evidence for the experiment report.  GPU
            # runs may terminate through the worker's normal shutdown path,
            # so best-effort persistence belongs in ``finally``.
            self.last_generation_metrics["fastsd_scheduler"] = dict(scheduler_metrics)
            metrics_dir = getattr(self.args, "exp_name", None)
            if metrics_dir:
                try:
                    os.makedirs(metrics_dir, exist_ok=True)
                    metrics_path = os.path.join(metrics_dir, "scheduler_metrics.json")
                    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
                        json.dump(self.last_generation_metrics, metrics_file, indent=2, default=str)
                except Exception as exc:  # pragma: no cover - diagnostics only
                    self.color_print(f"[FASTSD] failed to persist scheduler metrics: {exc}", 2)


    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            
            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size].to(torch.float32)
                prob = prob.to("cuda:1")
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            draft_ids = all_prob[0, [0], 1:self.args.gamma*2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    
                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    
                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                    
                    prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma +n+1)
            
        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding_RC(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCache2Model(self.draft_model, self.draft_model_2, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = torch.device("cuda:0")
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = torch.device("cuda:1")

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            
            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size]
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size]
                # ! the prob of the target model should be moved to a different device of the draft device to avoid deadlock
                prob = prob.to("cuda:1")
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            draft_ids = all_prob[0, [0], 1:self.args.gamma*2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]

            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    
                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0
                    
                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                    
                    prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma +n+1)
            
            self.accelerator.wait_for_everyone()
            
        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding_without_strategy_1(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine whether to use the strategy 2
        cur_mode = False

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len-self.args.gamma-2:prefix_len, :self.vocab_size].to(torch.float32)
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma*2+1] = x[:, prefix_len-self.args.gamma:prefix_len+self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len-self.args.gamma-2:prefix_len, :self.vocab_size].to(torch.float32)
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()

            all_prob = self.accelerator.gather(prob).to(device)

            assert all_prob[0, 0, 0] == -1
            draft_ids = all_prob[0, [0], 1:self.args.gamma*2+1].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]

            if cur_mode:
                n = self.args.gamma + 1
                for i in range(self.args.gamma + 1):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma-1 + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma + 1:
                    # accept all guess tokens
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma + 1
                    cur_mode = False
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))

                    prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n], t), dim=1)
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma +n)

            else:
                prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)
                cur_mode = True

        return prefix

    @torch.no_grad()
    def parallel_speculative_decoding_without_strategy_2(self, prefix):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        # this flag is used to determine whether to use the strategy 1
        cur_mode = True

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            
            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size]
                prob[:, 0, 0] = -1
                prob[:, 0, 1:self.args.gamma*2] = x[:, prefix_len-self.args.gamma+1:prefix_len+self.args.gamma]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[:, prefix_len-self.args.gamma-1:prefix_len, :self.vocab_size]
                self.target_forward_times += 1
            
            self.accelerator.wait_for_everyone()
            
            all_prob = self.accelerator.gather(prob)
            
            assert all_prob[0, 0, 0] == -1
            draft_ids = all_prob[0, [0], 1:self.args.gamma*2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)
                r = torch.rand(1, device=device)
                if  r > target_prob[:, -1, first_token] / draft_prob[:, -1, first_token]:
                    # reject the first token
                    t = sample(max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :]))
                    prefix = torch.cat((input_ids, t), dim=1)
                    
                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat((input_ids, draft_ids[:, -self.args.gamma:]), dim=1)

            else:
                n = self.args.gamma-1
                for i in range(self.args.gamma-1):
                    token = draft_ids[:, i]
                    torch.manual_seed(self.seed + prefix_len - self.args.gamma + i)
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break

                cur_mode = True
                if n == self.args.gamma -1:
                    t = sample(target_prob[:, n, :])
                else:
                    t = sample(max_fn(target_prob[:, n, :] - draft_prob[:, n, :]))
                
                prefix = torch.cat((input_ids[:, :prefix_len-self.args.gamma + n + 1], t), dim=1)
                # rollback both the large model and the small model kv cache
                model.rollback(prefix_len - self.args.gamma +n+1)
            
        return prefix
    
    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int=4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
