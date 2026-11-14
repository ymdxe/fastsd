import os
import json
from dataclasses import dataclass, field
from contextlib import nullcontext
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
from .kvcache_varlen import varlen_generate, supports_varlen_model, require_full_attention
from .kvcache4RC import KVCacheModel as KVCache2Model
from .cache_offload_policy import should_offload_target_cache
from .cache_residency import (
    CPU_RESIDENT,
    EVICTION_PENDING,
    GPU_RESIDENT,
    PREFETCH_PENDING,
    RESTORE_PENDING,
    ResidencyManager,
)
from .session_lifecycle import SessionLifecycle
from .cache_transfer import CacheTransferManager
from .cache_protocol import PrefetchTarget
from .minimal_prefetch import CudaH1Copy, H1Prefetcher
from .progress_control import AcceptanceProfile, BatchCostModel, GammaController, choose_gamma, draft_seconds_per_token
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
from diagnostic_trace import trace as DIAG, request_meta as diag_req, item_meta as diag_item, queue_rows as diag_queues, backlog as diag_backlog
import threading
import requests

from .energy_meter import EnergyControlService, EnergyServiceConfig
from .fastsd_scheduler import (
    AdmissionPlan,
    cost_features,
    FASTSD_DEFAULT_R1,
    FASTSD_DEFAULT_R2,
    FASTSD_DYNAMIC_WINDOW,
    WAIT_LAMBDA_DEFAULT,
    WAIT_LINEAR_UNTIL_S,
    WAIT_LINEAR_ALPHA,
    build_fixed_wrr_order as _build_fixed_wrr_order,
    compute_priority_score as _compute_priority_score,
    full_prefix_bridge_tokens,
    length_category as _length_category,
    predict_next_verify_proc_ids as _predict_next_verify_proc_ids,
    WorkItem,
    TaskLengthClassifier,
    abort_admission_plan,
    commit_admission_plan,
    plan_iteration,
    predict_next_plan,
    reserve_admission_plan,
    should_switch_to_prefill as _should_switch_to_prefill,
    update_length_thresholds as _update_length_thresholds,
    verify_logit_position,
)
from .request_validation import canonicalize_request

# Prediction depth is independent of transfer depth. Deeper predictions are
# still measured, but consume no arena/DMA budget unless explicitly enabled.
PREFETCH_LOOKAHEAD_ROUNDS = max(
    1, int(os.environ.get("KV_PREFETCH_LOOKAHEAD_ROUNDS", "1"))
)
PREFETCH_HORIZON_LIMIT = max(0, int(os.environ.get("KV_PREFETCH_HORIZON_LIMIT", "1")))
PREFETCH_H2_MAX_PIDS = max(0, int(os.environ.get("KV_PREFETCH_H2_MAX_PIDS", "4")))
PREFETCH_BEFORE_SUBMIT = os.environ.get("KV_PREFETCH_BEFORE_SUBMIT", "0").lower() in {
    "1", "true", "yes", "on",
}
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
    cache_tickets: dict = field(default_factory=dict)
    ready_at: float = field(default_factory=time.monotonic)


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

    def _start_energy_campaign(self, energy_service, *, metadata=None) -> bool:
        """Start the measurement window owned by this target benchmark.

        The cloud target runs in its own worker process, so it cannot rely on
        the draft-process lifecycle to start the campaign.  Keep the HTTP
        control call here, after the local FastAPI service is ready and just
        before entering the target request loop.
        """
        if energy_service is None:
            return False
        payload_metadata = {
            "role": "target",
            "event": "benchmark_start",
        }
        if metadata:
            payload_metadata.update(metadata)
        self._post_energy_event(
            "start",
            {
                "exp_id": str(self.args.exp_name),
                "metadata": payload_metadata,
            },
            retries=60,
        )
        self.color_print("[ENERGY] START_MEASURE sent (target benchmark)", 2)
        return True

    def _stop_energy_campaign(self, energy_service, started: bool, *, metadata=None) -> None:
        """Stop the target-owned campaign before shutting down its service."""
        if energy_service is None or not started:
            return
        payload_metadata = {
            "role": "target",
            "event": "benchmark_exit",
        }
        if metadata:
            payload_metadata.update(metadata)
        try:
            result = self._post_energy_event(
                "stop",
                {
                    "exp_id": str(self.args.exp_name),
                    "total_tasks": None,
                    "metadata": payload_metadata,
                },
                retries=10,
            )
            campaign = result.get("result_b_campaign_window", {})
            self.color_print(
                f"[ENERGY] STOP_MEASURE sent (target benchmark), "
                f"campaign={float(campaign.get('energy_j', 0.0)):.4f}J",
                2,
            )
        except Exception as exc:  # pragma: no cover - diagnostics must not mask shutdown
            self.color_print(f"[ENERGY] STOP_MEASURE failed during target shutdown: {exc}", 1)

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
    def autoregressive_sampling(self, prefix, *, request_start=None):
        generation_start = time.perf_counter() if request_start is None else request_start
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError("Auto-Regressive Decoding can be used only in small / large eval mode!")
        
        prefix = prefix.to(model.device)
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
            "first_token_time_s": first_token_time,
            "completion_time_s": completion_time,
        }
        return x

    @torch.no_grad()
    def speculative_decoding(self, prefix, *, request_start=None):
        generation_start = time.perf_counter() if request_start is None else request_start
        original_prefix_len = prefix.shape[1]
        max_tokens = original_prefix_len + self.args.max_tokens
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
            "first_token_time_s": first_token_time,
            "completion_time_s": completion_time,
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
    def _gamma_controller(self):
        return GammaController(policy=getattr(self.args, "gamma_policy", "progress"),
            adaptive=getattr(self.args, "pipeline_gamma_adapt", True),
            minimum=int(getattr(self.args, "pipeline_gamma_min", 1)),
            maximum=int(getattr(self.args, "pipeline_gamma_max", 16)),
            step=int(getattr(self.args, "pipeline_gamma_step", 2)),
            alpha=float(getattr(self.args, "pipeline_ema_alpha", 0.2)))

    @torch.no_grad()
    def run_target_process(self, tokenizer, request_queue, response_queues):
        self.color_print(f"Loading models:\n{self.args.target_model}", 3)
        target_model = self._load_target_model_for_service(
            self.args.target_model, self._service_target_device()
        )
        self.vocab_size = self.args.vocab_size
        energy_service = self._maybe_start_energy_service()
        energy_campaign_started = False

        require_full_attention(target_model)
        gamma_control = self._gamma_controller()
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

            synchronize(target_model.device)
            compute_started = time.perf_counter()
            _ = cache.generate(x_for_target, 1, sample_next=False)
            synchronize(target_model.device)
            verify_s = time.perf_counter() - compute_started

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
                "verify_ms": verify_s * 1000,
                **gamma_control.finish(proc_id, accepted=accepted_cnt,
                    proposed=req_gamma, request=request, verify_s=verify_s),
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
            energy_campaign_started = self._start_energy_campaign(
                energy_service,
                metadata={"worker_mode": "legacy_target"},
            )
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
            self._stop_energy_campaign(
                energy_service,
                energy_campaign_started,
                metadata={"worker_mode": "legacy_target"},
            )
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
        require_full_attention(target_model)
        compute_stream = None
        from .attention_backend import require_attention_backend
        attention_backend = getattr(self.args, "target_attention_backend", "sdpa_per_sequence")
        if attention_backend == "flash_attn_varlen" and getattr(self.args, "kv_batch_mode", "varlen") != "varlen":
            raise ValueError("flash_attn_varlen requires --kv_batch_mode varlen")
        require_attention_backend(attention_backend, device=target_model.device, dtype=target_model.dtype)
        if target_model.device.type == "cuda":
            compute_stream = torch.cuda.Stream(device=target_model.device)
            compute_stream.wait_stream(torch.cuda.current_stream(target_model.device))
        self.vocab_size = self.args.vocab_size
        energy_service = self._maybe_start_energy_service()
        energy_campaign_started = False

        # 批处理版 KVCache 管理器
        kv_cache_manager = KVCacheModel_batching(target_model,
                                        temperature=self.args.temp,
                                        top_k=self.args.top_k,
                                        top_p=self.args.top_p,
                                        attention_backend=attention_backend)
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
            "prefetch_ready_hit": 0,
            "prefetch_pending_hit": 0,
            "prefetch_demand_miss": 0,
            # Verify-only readiness counters.  Prefetch intents are built from
            # verify plans, so these counters share one denominator.  Prefill
            # restore claims are reported separately below.
            "prefetch_verify_ready_hit": 0,
            "prefetch_verify_pending_hit": 0,
            "prefetch_verify_demand_miss": 0,
            "prefetch_prefill_ready_hit": 0,
            "prefetch_prefill_pending_hit": 0,
            "prefetch_prefill_demand_miss": 0,
            "prefetch_claim_lead_ms_sum": 0.0,
            "prefetch_claim_lead_ms_max": 0.0,
            "prefetch_claims": 0,
            "prefetch_ready_by_horizon": {},
            "prefetch_pending_by_horizon": {},
            "prefetch_demand_miss_by_horizon": {},
            "prefetch_prefill_ready_by_horizon": {},
            "prefetch_prefill_pending_by_horizon": {},
            "prefetch_prediction_plans": 0,
            "prefetch_prediction_predicted": 0,
            "prefetch_prediction_actual": 0,
            "prefetch_prediction_hits": 0,
            "prefetch_prediction_misses": 0,
            "prefetch_prediction_wasted": 0,
            "batch_compute_samples": 0,
            "batch_compute_total_s": 0.0,
            "batch_compute_avg_s": 0.0,
            "prefetch_lookahead_rounds": PREFETCH_LOOKAHEAD_ROUNDS,
            "prefetch_prediction_by_horizon": {},
            "prefetch_refresh_count": 0,
            "lookahead_build_count": 0,
            "prediction_snapshots_consumed": 0,
            "actual_plans": 0,
            "prefetch_skipped_horizon": {},
            "prefetch_horizon_limit": PREFETCH_HORIZON_LIMIT,
            "prefetch_h2_max_pids": PREFETCH_H2_MAX_PIDS,
            "prefetch_before_submit": PREFETCH_BEFORE_SUBMIT,
            "wait_priority_config": {
                "formula": "piecewise" if WAIT_LINEAR_UNTIL_S > 0 else "exp",
                "lambda": WAIT_LAMBDA_DEFAULT,
                "linear_until_s": WAIT_LINEAR_UNTIL_S,
                "linear_alpha": WAIT_LINEAR_ALPHA,
            },
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
            "max_batch_kv_bytes": 0,
            "max_gpu_resident_kv_bytes": 0,
            "max_total_kv_plus_prefetch_bytes": 0,
            "max_prefetch_reserved_bytes": 0,
            "residency_pressure_samples": 0,
            "residency_high_samples": 0,
            "residency_pressure_guard_trips": 0,
            "residency_used_bytes_sum": 0,
            "residency_used_bytes_max": 0,
            "max_active_batch_sequences": 0,
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
        accept_stats_version = [0]
        committed_prefix_tokens = {}  # {proc_id: List[int]} for debug context display
        sampling_generators = {}  # session/proc id -> target-side rejection RNG
        gamma_control = self._gamma_controller()
        progress_profiles = gamma_control.profiles
        batch_cost = BatchCostModel()
        batch_cost_lock = threading.RLock()
        length_classifier = TaskLengthClassifier()
        residency = CacheResidency()
        # Explicit-budget placement policy.  budget_bytes == 0 keeps every
        # cache on the GPU and never runs the transfer machinery at all.
        residency_manager = ResidencyManager(
            budget_bytes=int(getattr(self.args, "kv_residency_budget_mb", 0)) * 1024 ** 2,
            high_ratio=float(getattr(self.args, "kv_residency_high_ratio", 0.90)),
            low_ratio=float(getattr(self.args, "kv_residency_low_ratio", 0.75)),
            headroom_bytes=int(float(getattr(self.args, "kv_headroom_mb", 512.0)) * 1024 ** 2),
        )
        plan_sequence = {"tick": 0}
        use_h1_prefetch = getattr(self.args, "kv_prefetch_implementation", "h1") == "h1"
        h1_prefetch = None
        prefetch_reclaim = {"required_free": 0}
        scheduler_metrics["kv_prefetch_implementation"] = "h1" if use_h1_prefetch else "legacy"
        if use_h1_prefetch:
            scheduler_metrics.update(prefetch_lookahead_rounds=1, prefetch_horizon_limit=1,
                                     prefetch_h2_max_pids=0, prefetch_before_submit=False)
        cache_lock = residency.lock
        preloaded_gpu_pids = residency.resident
        # pids whose cache is currently owned by the executing batch (forward +
        # rollback window) or by an admitted-but-not-yet-forwarded plan. The
        # async prefetch worker must never touch these.
        active_pids = residency.active
        in_flight = {"plan": None, "job": None}

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
                ticket = residency.retire_session_locked(pid)
                kv_cache_manager.reset(pid)
                transfers.release_cache(owner=pid)
            if ticket is not None:
                if h1_prefetch is not None:
                    h1_prefetch.retire(ticket)
                else:
                    transfers.retire(ticket)
            residency_manager.release(pid)
            with accept_stats_lock:
                accept_stats.pop(pid, None)
            for store in (committed_prefix_tokens, sampling_generators, pending_prefill_timing):
                store.pop(pid, None)
            gamma_control.forget(pid)
            pending_prefill_cancel.discard(pid)

        lifecycle = SessionLifecycle(
            cleanup=cleanup_session, cancel=cancel_session,
            protected=residency.protected_snapshot,
            busy=lambda: {item.proc_id for item in work_items.values() if not item.finished},
            reply=lambda key, payload: response_queues[key].put(payload),
            idle_timeout=float(getattr(self.args, "session_idle_timeout", 300.0)),
        )

        def flush_session_releases():
            # BYPASS must not touch the transfer registry at all.
            if residency_manager.active:
                transfers.collect()
                if h1_prefetch is not None:
                    h1_prefetch.collect()
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
            mode=getattr(self.args, "kv_prefetch_mode", "overlap"),
            prefetch_bytes=int(getattr(self.args, "kv_prefetch_mb", 64)) * 1024 ** 2,
            chunk_bytes=int(getattr(self.args, "kv_prefetch_chunk_kb", 4096)) * 1024,
            speculative_chunk_bytes=int(getattr(self.args, "kv_prefetch_speculative_chunk_kb", 4096)) * 1024,
            headroom_bytes=int(getattr(self.args, "kv_headroom_mb", 512)) * 1024 ** 2,
            disturbance_budget=float(getattr(self.args, "kv_overlap_max_slowdown", 0.1)),
            adaptive_fallback=bool(getattr(self.args, "kv_overlap_auto_fallback", False)),
            rate_bytes_per_second=(
                float("inf") if float(getattr(self.args, "kv_prefetch_rate_mb_s", 0)) <= 0
                else float(getattr(self.args, "kv_prefetch_rate_mb_s")) * 1024 ** 2
            ),
            idle_rate_bytes_per_second=(
                float("inf") if float(getattr(self.args, "kv_prefetch_idle_rate_mb_s", 0)) <= 0
                else float(getattr(self.args, "kv_prefetch_idle_rate_mb_s")) * 1024 ** 2
            ),
            online_fraction=float(getattr(self.args, "kv_prefetch_online_fraction", 1.0)),
            event_pool_size=int(getattr(self.args, "kv_prefetch_event_pool_size", 256)),
            warmup_pool=False,
        )
        # The prefetch arena, pinned pool and copy streams are created only
        # when the residency manager enters ACTIVE; BYPASS allocates nothing.
        calibration_path = getattr(self.args, "kv_prefetch_calibration", None)
        if calibration_path and not use_h1_prefetch:
            gpu_uuid = (str(torch.cuda.get_device_properties(target_model.device).uuid)
                        if target_model.device.type == "cuda" else "cpu")
            transfers.load_calibration(calibration_path, context={
                "gpu_uuid": gpu_uuid,
                "model": os.path.realpath(str(self.args.target_model)),
                "dtype": str(target_model.dtype),
                "attention_backend": str(getattr(self.args, "target_attention_backend", "sdpa_per_sequence")),
                "forward_primitive": ("varlen_generate" if requested_kv_batch_mode == "varlen"
                                      else "KVCacheModel_batching"),
                "kv_batch_mode": requested_kv_batch_mode,
            })
        phase_leases = []
        phase_prefetch_tickets = []

        def acquire_cache(req):
            pid = req["proc_id"]
            if not residency_manager.enabled:
                # Management off: the cache is always on the GPU.
                with cache_lock:
                    return kv_cache_manager._past_key_values[pid]
            record = residency_manager.get(pid)
            if record is None or record.state == GPU_RESIDENT:
                # Resident fast path: one dict lookup, no ticket/event/lease.
                residency_manager.note_used(pid, plan_sequence["tick"])
                with cache_lock:
                    return kv_cache_manager._past_key_values[pid]
            with cache_lock:
                cache = kv_cache_manager._past_key_values[pid]
                snapshot = residency.snapshot_locked(pid, cache)
                ticket = req.pop("_cache_ticket", None)
            record = residency_manager.get(pid)
            if h1_prefetch is not None and req.get("task_type") == "verify":
                pending_before = scheduler_metrics["prefetch_pending_hit"]
                ticket = h1_prefetch.claim(pid, snapshot)
                if ticket is not None:
                    # Completed DMA only: consume records allocation lifetime,
                    # never inserts a stream wait for a speculative copy.
                    ready_cache = h1_prefetch.consume(ticket, compute_stream)
                    phase_prefetch_tickets.append(ticket)
                    with cache_lock:
                        kv_cache_manager._past_key_values[pid] = ready_cache
                        transfers.retain_cache(ready_cache, owner=pid)
                    residency_manager.transition(pid, RESTORE_PENDING)
                    record.gpu_cache = ready_cache
                    record.cpu_cache = None
                    residency_manager.transition(pid, GPU_RESIDENT)
                    residency_manager.note_used(pid, plan_sequence["tick"])
                    residency_manager.counters["restores"] += 1
                    residency_manager.counters["prefetch_claimed"] += 1
                    scheduler_metrics["prefetch_verify_ready_hit"] += 1
                    bucket = scheduler_metrics["prefetch_ready_by_horizon"].setdefault("1", {"ready": 0})
                    bucket["ready"] += 1
                    return ready_cache
                if scheduler_metrics["prefetch_pending_hit"] > pending_before:
                    scheduler_metrics["prefetch_verify_pending_hit"] += 1
                    bucket = scheduler_metrics["prefetch_pending_by_horizon"].setdefault("1", {"pending": 0})
                    bucket["pending"] += 1
            if ticket is not None and not ticket.matches(snapshot):
                transfers.retire(ticket)
                ticket = None
            restored = record is not None and record.state != GPU_RESIDENT
            if ticket is None:
                if record is not None and record.state == EVICTION_PENDING:
                    # Planning must exclude these; never install a half-moved
                    # cache into a forward.
                    raise RuntimeError(f"session {pid!r} is being evicted")
                ticket = transfers.submit_h2d(snapshot, purpose="demand")
                if restored:
                    miss_key = ("prefetch_verify_demand_miss"
                                if req.get("task_type") == "verify"
                                else "prefetch_prefill_demand_miss")
                    scheduler_metrics[miss_key] = scheduler_metrics.get(miss_key, 0) + 1
                    # Keep the historical aggregate for report consumers.
                    scheduler_metrics["prefetch_demand_miss"] = (
                        scheduler_metrics.get("prefetch_demand_miss", 0) + 1)
                    horizon = req.get("_prediction_horizon")
                    if horizon is not None:
                        bucket = scheduler_metrics["prefetch_demand_miss_by_horizon"].setdefault(
                            str(horizon), {"demand_miss": 0})
                        bucket["demand_miss"] += 1
            elif ticket.purpose == "prefetch":
                residency_manager.counters["prefetch_claimed"] += 1
            if record is not None and record.state in (CPU_RESIDENT, PREFETCH_PENDING):
                residency_manager.transition(pid, RESTORE_PENDING)
            lease = transfers.bind_to_consumer(ticket, compute_stream=compute_stream)
            # Establish only this PID's dependency before any cache mutation.
            transfers.protect_source_write(lease.cache, producer_stream=compute_stream)
            phase_leases.append(lease)
            with cache_lock:
                kv_cache_manager._past_key_values[pid] = lease.cache
                transfers.retain_cache(lease.cache, owner=pid)
            residency_manager.note_used(pid, plan_sequence["tick"])
            if record is not None:
                record.gpu_cache = lease.cache
                record.cpu_cache = None
                if record.state in (RESTORE_PENDING,):
                    residency_manager.transition(pid, GPU_RESIDENT)
                    residency_manager.counters["restores"] += 1
            return lease.cache

        forward_measurements = []
        current_batch = []

        measure_forward_gpu = bool(getattr(self.args, "measure_forward_gpu", False))

        def timed_forward(callback, *args, **kwargs):
            start = end = None
            if compute_stream is not None and measure_forward_gpu:
                # Event timing requires a host/device rendezvous.  It is opt-in
                # because the synchronize itself serializes the control loop.
                compute_stream.synchronize()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record(compute_stream)
            launched = time.monotonic()
            diag_forward_id = time.monotonic_ns()
            DIAG.emit("forward_host_start", forward_id=diag_forward_id, requests=[diag_req(r) for r in current_batch])
            for req in current_batch:
                req["_batch_start_delay_ms"] = (launched - req.get("_plan_ready_at", launched)) * 1000.0
            spec_before = transfers.metrics["speculative_submitted_bytes"]
            spec_active = (transfers.speculative_inflight()
                           if residency_manager.active else False)
            result = callback(*args, **kwargs)
            if end is not None:
                end.record(compute_stream)
                end.synchronize()
                elapsed_gpu = start.elapsed_time(end)
                DIAG.gpu_span("target_forward", start, end, forward_id=diag_forward_id, requests=[diag_req(r) for r in current_batch])
                forward_measurements.append(elapsed_gpu)
                shape = forward_shape(current_batch)
                transfers.observe_forward(shape, elapsed_gpu, speculative=spec_active or
                    transfers.metrics["speculative_submitted_bytes"] > spec_before)
            DIAG.emit("forward_host_end", forward_id=diag_forward_id)
            return result

        residency_cache = {"bytes_per_token": 0}

        def cache_bytes_per_token():
            """O(1) size estimate, measured once from a live cache."""

            if residency_cache["bytes_per_token"]:
                return residency_cache["bytes_per_token"]
            for cache in list(kv_cache_manager._past_key_values.values()):
                try:
                    length = int(cache.get_seq_length())
                    legacy = cache.to_legacy_cache()
                except Exception:
                    continue
                if legacy and length > 0:
                    total = sum(t.numel() * t.element_size() for layer in legacy for t in layer)
                    residency_cache["bytes_per_token"] = max(1, total // length)
                    return residency_cache["bytes_per_token"]
            return 0

        def ensure_cache_management():
            """One-time setup: seed live sessions, allocate the transfer pool.

            Management is enabled by the budget, not by pressure, so this runs
            on the first plan after start-up instead of on a high sample.
            """

            if not residency_manager.enabled:
                return False
            if not residency_manager.mark_started():
                return False
            per_token = cache_bytes_per_token()
            for pid, cache in list(kv_cache_manager._past_key_values.items()):
                try:
                    length = int(cache.get_seq_length())
                except Exception:
                    continue
                residency_manager.note_created(
                    pid, valid_len=length, bytes_per_token=per_token,
                    cache=cache, version=residency.version(pid),
                    epoch=getattr(residency, "epochs", {}).get(pid, 0))
            if (not use_strict_fcfs and prefetch_enabled
                    and getattr(self.args, "kv_prefetch", True)):
                if not use_h1_prefetch:
                    transfers.warmup_prefetch_pool()
                    prefetch_worker.start()
                elif transfers.enabled and transfers.mode == "overlap":
                    prefetch_worker.start()
            scheduler_metrics["kv_residency_activations"] = (
                scheduler_metrics.get("kv_residency_activations", 0) + 1)
            return True

        def observe_memory():
            """One budget sample per plan; never inside a per-request path."""

            if not residency_manager.enabled:
                return False
            guard_free = None
            if target_model.device.type == "cuda":
                guard_free, _ = torch.cuda.mem_get_info(target_model.device)
            used = residency_manager.used_bytes()
            pressure = residency_manager.update_pressure(
                used, guard_free_bytes=guard_free)
            scheduler_metrics["residency_pressure_samples"] += 1
            scheduler_metrics["residency_high_samples"] += int(residency_manager.high)
            scheduler_metrics["residency_used_bytes_sum"] += int(used)
            scheduler_metrics["residency_used_bytes_max"] = max(
                scheduler_metrics.get("residency_used_bytes_max", 0), int(used)
            )
            scheduler_metrics["residency_pressure_guard_trips"] = int(
                residency_manager.counters.get("pressure_guard_trips", 0)
            )
            return pressure

        def finish_cache_phase(batch):
            if not residency_manager.active:
                # Residency default: no events, no copies, no bookkeeping.
                return
            per_token = cache_bytes_per_token()
            for req in batch:
                pid = req["proc_id"]
                with cache_lock:
                    cache = kv_cache_manager._past_key_values[pid]
                residency.bump_locked(pid)
                try:
                    length = int(cache.get_seq_length())
                except Exception:
                    length = 0
                record = residency_manager.get(pid)
                if record is None:
                    residency_manager.note_created(
                        pid, valid_len=length, bytes_per_token=per_token,
                        cache=cache, version=residency.version(pid),
                        epoch=getattr(residency, "epochs", {}).get(pid, 0))
                else:
                    residency_manager.note_mutated(
                        pid, version=residency.version(pid), valid_len=length)
                    record.gpu_cache = cache
                    record.in_transfer = False
                    if record.state == RESTORE_PENDING:
                        residency_manager.transition(pid, GPU_RESIDENT)
                # Fresh prefills had no record when acquire_cache() ran.
                # Register their recency after creation as well, so new KV
                # does not appear older than every previously used session.
                residency_manager.note_used(pid, plan_sequence["tick"])
                if req.get("task_type") in {"prefill", "verify"}:
                    with accept_stats_lock:
                        accepted, proposed = accept_stats.get(pid, (0, 0))
                    # Match the scheduler's smoothed acceptance estimate.
                    probability = ((accepted + 1.0) / (proposed + 1.0)
                                   if proposed > 0 else 1.0)
                    residency_manager.note_request_cost(
                        pid, draft_total_time=req.get("lag", 0.0),
                        transport_rtt=req.get("transport_rtt", 0.0),
                        accept_prob=probability)
                if target_model.device.type == "cuda":
                    preloaded_gpu_pids.add(pid)
            scheduler_metrics["max_gpu_resident_kv_bytes"] = max(
                scheduler_metrics.get("max_gpu_resident_kv_bytes", 0),
                residency_manager.used_bytes(),
            )
            prefetch_reserved = (h1_prefetch.reserved_bytes if h1_prefetch is not None else
                                 int(transfers.metrics.get("prefetch_reserved_bytes", 0) or 0))
            scheduler_metrics["max_prefetch_reserved_bytes"] = max(
                scheduler_metrics.get("max_prefetch_reserved_bytes", 0), prefetch_reserved
            )
            scheduler_metrics["max_total_kv_plus_prefetch_bytes"] = max(
                scheduler_metrics.get("max_total_kv_plus_prefetch_bytes", 0),
                residency_manager.used_bytes() + prefetch_reserved,
            )

        def rebalance_cache_memory(batch_pids, *, min_free_bytes=0):
            """Trim at high residency or on the 256 MiB free-memory emergency.

            Restore estimates alone cannot start offloading below high. Read
            free memory at the execution boundary so a planner sample from
            the preceding forward cannot hide or prolong an emergency.
            """

            if not residency_manager.enabled:
                return 0
            guard_free = None
            if target_model.device.type == "cuda":
                guard_free, _ = torch.cuda.mem_get_info(target_model.device)
            restore_bytes = sum(
                record.nbytes for record in (
                    residency_manager.get(pid) for pid in batch_pids)
                if record is not None and record.state != GPU_RESIDENT)
            needed = residency_manager.deficit_bytes(restore_bytes, guard_free_bytes=guard_free)
            if guard_free is not None:
                needed = max(needed, min_free_bytes - guard_free)
            if needed <= 0:
                return 0
            emergency = (guard_free is not None
                         and guard_free < residency_manager.emergency_free_bytes)
            if emergency:
                residency_manager.counters["emergency_eviction_rounds"] += 1
            protected = set(batch_pids) | residency.protected_snapshot()
            victims = residency_manager.rank_victims(
                plan_sequence["tick"], exclude=protected)
            evicted = 0
            freed_bytes = 0
            for pid in victims:
                if freed_bytes >= needed:
                    break
                record = residency_manager.get(pid)
                if record is None or record.state != GPU_RESIDENT:
                    continue
                if not should_offload_target_cache(
                        "verify", pid=pid, memory_pressure=True, continuing=False):
                    continue
                with cache_lock:
                    cache = kv_cache_manager._past_key_values.get(pid)
                if cache is None:
                    residency_manager.force_state(pid, CPU_RESIDENT)
                    continue
                try:
                    residency_manager.transition(pid, EVICTION_PENDING)
                except Exception:
                    continue
                try:
                    # Record the producer event only when a copy actually
                    # happens; a resident phase pays nothing.
                    transfers.mark_computed(cache, producer_stream=compute_stream)
                    ticket = transfers.submit_d2h(cache)
                except BaseException as exc:
                    # Staying over budget is recoverable; a half-moved cache
                    # is not.  Restore the durable state and skip this victim.
                    residency_manager.force_state(pid, GPU_RESIDENT)
                    scheduler_metrics["kv_residency_eviction_errors"] = (
                        scheduler_metrics.get("kv_residency_eviction_errors", 0) + 1)
                    scheduler_metrics["kv_residency_last_error"] = str(exc)
                    continue
                moved = ticket.destination_cache
                with cache_lock:
                    kv_cache_manager._past_key_values[pid] = moved
                    transfers.retain_cache(moved, owner=pid)
                    residency.bump_locked(pid)
                record.cpu_cache = moved
                record.gpu_cache = None
                try:
                    record.valid_len = int(moved.get_seq_length())
                except Exception:
                    pass
                record.cpu_valid_len = record.valid_len
                record.version = residency.version(pid)
                residency_manager.transition(pid, CPU_RESIDENT)
                preloaded_gpu_pids.discard(pid)
                evicted += 1
                freed_bytes += record.nbytes
            if evicted:
                residency_manager.counters["evictions"] += evicted
                if emergency:
                    residency_manager.counters["emergency_evictions"] += evicted
                scheduler_metrics["kv_residency_evictions"] = (
                    scheduler_metrics.get("kv_residency_evictions", 0) + evicted)
            elif emergency:
                residency_manager.counters["emergency_blocked_rounds"] += 1
            return evicted

        # --------------------- 批量处理（Prefill / Verify） ----------------
        def forward_shape(batch):
            return (
                batch[0]["task_type"], len(batch),
                tuple(int(cost_features(req)[1]) for req in batch),
                tuple(int(req.get("_cached_context", 0)) for req in batch),
                str(target_model.dtype),
                str(getattr(self.args, "target_attention_backend", "sdpa_per_sequence")),
            )

        def handle_request_batch(batch: list[dict], return_responses: bool = False):
            proc_ids = [req["proc_id"] for req in batch]
            DIAG.emit("batch_start", batch_id=time.monotonic_ns(), requests=[diag_req(r) for r in batch])
            if residency_manager.enabled:
                batch_kv_bytes = sum(
                    int(record.nbytes)
                    for record in (residency_manager.get(pid) for pid in proc_ids)
                    if record is not None
                )
                scheduler_metrics["max_batch_kv_bytes"] = max(
                    scheduler_metrics.get("max_batch_kv_bytes", 0), batch_kv_bytes
                )
                scheduler_metrics["max_active_batch_sequences"] = max(
                    scheduler_metrics.get("max_active_batch_sequences", 0), len(proc_ids)
                )
            residency.activate(proc_ids)
            phase_leases.clear()
            phase_prefetch_tickets.clear()
            forward_measurements.clear()
            current_batch[:] = batch
            try:
                started = time.perf_counter()
                # Evict before touching this batch: victims are chosen outside
                # the active/reserved set and never include a batch pid.
                if h1_prefetch is not None:
                    # Formal placement is owned by the executor. A speculative
                    # memory shortage requests eviction at this safe boundary;
                    # the worker skips rather than blocking this forward.
                    with cache_lock:
                        required_free = prefetch_reclaim["required_free"]
                        prefetch_reclaim["required_free"] = 0
                    rebalance_cache_memory(proc_ids, min_free_bytes=required_free)
                else:
                    rebalance_cache_memory(proc_ids)
                for req in batch:
                    with cache_lock:
                        existing = kv_cache_manager._past_key_values.get(req["proc_id"])
                        req["_cached_context"] = (int(existing.get_seq_length()) if existing is not None
                            and (req["task_type"] != "prefill" or req.get("_prefill_continuation", False)) else 0)
                if residency_manager.active:
                    transfers.begin_execution(shape=forward_shape(batch))
                with torch.cuda.stream(compute_stream) if compute_stream is not None else nullcontext():
                    result = _handle_request_batch_inner(batch, True, requested_kv_batch_mode, proc_ids)
                    stall = sum(lease.exposed_stall_ms() for lease in phase_leases)
                    gpu_ms = sum(forward_measurements)
                    for req in batch:
                        payload = result.get(req["proc_id"])
                        if payload is not None:
                            payload.update(forward_gpu_ms=gpu_ms,
                                exposed_h2d_stall_ms=stall,
                                batch_start_delay_ms=req.get("_batch_start_delay_ms", 0.0))
                    rows = [cost_features(req) for req in batch]
                    with batch_cost_lock:
                        batch_cost.observe(batch[0]["task_type"], rows,
                            gpu_ms / 1000 if compute_stream is not None else time.perf_counter() - started)
                if not return_responses:
                    for req in batch:
                        payload = result.get(req["proc_id"])
                        if payload is not None:
                            if req["task_type"] == "verify":
                                elapsed = max(0.0, time.monotonic() - req.get("server_enqueue_monotonic", started))
                                service = float(payload.get("forward_gpu_ms", 0) if compute_stream is not None
                                                else payload.get("verify_ms", 0)) / 1000.0
                                payload.update(gamma_control.finish(req["proc_id"],
                                    accepted=int(payload["accepted"]) - int(req["prefix_len"]),
                                    proposed=int(req["gamma"]), request=req,
                                    verify_s=service, queue_s=max(0, elapsed - service)))
                                payload["request_e2e_ms"] = elapsed * 1000.0
                                scheduler_metrics["gamma_rounds"] = gamma_control.rounds
                            response_queues[req.get("response_key", req["proc_id"])].put(payload)
                return result if return_responses else None
            finally:
                for lease in phase_leases:
                    transfers.release_consumer(lease, compute_stream=compute_stream)
                for ticket in phase_prefetch_tickets:
                    h1_prefetch.retire(ticket)
                if residency_manager.active:
                    transfers.end_execution(compute_stream=compute_stream)
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
                            transfers.release_cache(owner=pid)

                for req, cont in zip(batch, continuation):
                    if cont:
                        acquire_cache(req)
                if kv_batch_mode == "varlen":
                    residuals = []
                    for pid, cont, x in zip(proc_ids, continuation, seqs):
                        if cont:
                            cached_len = kv_cache_manager._past_key_values[pid].get_seq_length()
                            residuals.append(x[:, cached_len:])
                        else:
                            residuals.append(x)
                    timed_forward(varlen_generate,
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
                        timed_forward(kv_cache_manager.forward_new_tokens,
                            x_batch,
                            proc_ids=proc_ids,
                            pad_token_id=tokenizer.pad_token_id,
                            input_lens=input_lens,
                            reset_pids=[pid for pid, cont in zip(proc_ids, continuation) if not cont],
                            sample_next=False,
                        )
                    else:
                        timed_forward(kv_cache_manager.generate,
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

                with cache_lock:
                    preloaded_gpu_pids.difference_update(proc_ids)
                for req in batch:
                    acquire_cache(req)

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
                    _ = timed_forward(varlen_generate,
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
                    _ = timed_forward(kv_cache_manager.generate,
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
                    deliver(pid, {"status": "prefill_ok" if req.get("_prefill_final", True) else "prefill_partial"})
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
                    accept_stats_version[0] += 1
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
                deliver(pid, response_payload)

            finish_cache_phase(batch)
            return collected_responses if return_responses else None

        def accept_stats_snapshot():
            with accept_stats_lock:
                return {
                    pid: (values[0], values[1])
                    for pid, values in accept_stats.items()
                }

        def _sort_queues_if_dirty(now=None):
            # Wait times use the cloud monotonic clock; only re-sort when
            # ingress/completion actually changed the queues.
            priority = ((lambda request, stats, now: 0.0)
                        if getattr(self.args, "scheduler_policy", "legacy") == "fifo"
                        else lambda request, stats, now: _compute_priority_score(
                            request, stats, now=now,
                            latency_priority=bool(getattr(self.args, "enable_latency_priority", True))))
            task_queues.sort(
                accept_stats_snapshot(),
                now=time.monotonic() if now is None else now,
                category_for=length_classifier.category,
                priority=priority,
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
            transfers.retain_cache(cache, owner=pid)

        def _bind_target_device():
            # CUDA's current device is per-thread; executor/prefetch threads
            # must bind it explicitly (CPU returns without touching CUDA).
            set_thread_device(target_model.device)

        if use_h1_prefetch:
            def request_prefetch_reclaim(required_free):
                with cache_lock:
                    prefetch_reclaim["required_free"] = max(
                        prefetch_reclaim["required_free"], required_free)

            h1_prefetch = H1Prefetcher(
                residency=residency, manager=residency_manager, get_cache=_cache_get,
                driver=CudaH1Copy(target_model.device), metrics=scheduler_metrics,
                max_bytes=int(getattr(self.args, "kv_prefetch_mb", 64)) * 1024 ** 2,
                tick=lambda: plan_sequence["tick"], reclaim=request_prefetch_reclaim,
                thread_setup=_bind_target_device)
            prefetch_worker = prefetch_coordinator = h1_prefetch
        else:
            prefetch_worker = KVPrefetchWorker(
                residency=residency,
                get_cache=_cache_get,
                set_cache=_cache_set,
                move_cache=transfers.move,
                transfers=transfers,
                gpu_device=target_model.device,
                metrics=scheduler_metrics,
                on_error=lambda message: self.color_print(
                    f"[KV-PREFETCH-FAILED] {message}", 2
                ),
                thread_setup=_bind_target_device,
            )
            prefetch_coordinator = PrefetchCoordinator(
                submit=prefetch_worker.submit,
                resident_snapshot=residency.available_snapshot,
                protected_snapshot=residency.protected_snapshot,
                metrics=scheduler_metrics,
                pending_snapshot=prefetch_worker.has_pending,
            )

        def _slice_request(item: WorkItem, sl):
            req = dict(item.request)
            source = req["draft_output"]
            req["_forward_tokens"] = sl.forward_token_count
            req["_cost_features"] = cost_features(item, sl)
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

        default_profile = AcceptanceProfile()

        def candidate_value(item, sl, selected, items):
            rows = [cost_features(items[s.work_id], s) for s in selected
                    if s.task_type == item.task_type]
            features = cost_features(item, sl)
            with batch_cost_lock:
                diag_raw_marginal = batch_cost.estimate(item.task_type, rows + [features]) - batch_cost.estimate(item.task_type, rows)
                marginal = max(1e-6, diag_raw_marginal)
            if diag_raw_marginal <= 0 and DIAG.enabled and getattr(DIAG.context,"plan_kind","") == "admission":
                DIAG.emit("nonpositive_cost_increment", plan_id=getattr(DIAG.context,"plan_id",None), work_id=item.work_id, raw_seconds=diag_raw_marginal)
            with cache_lock:
                cache = kv_cache_manager._past_key_values.get(item.proc_id)
                ticket = residency.candidates.get(item.proc_id)
                ready = (cache is not None and ticket is not None
                         and ticket.matches(residency.snapshot_locked(item.proc_id, cache))
                         and ticket.ready())
            legacy = cache.to_legacy_cache() if cache is not None else ()
            if legacy and legacy[0][0].device != target_model.device and not ready:
                nbytes = sum(t.numel() * t.element_size() for layer in legacy for t in layer)
                if ticket is not None:
                    nbytes = max(0, nbytes - ticket.submitted_bytes)
                marginal += nbytes / max(1.0, float(getattr(self.args, "kv_transfer_gbps", 10)) * 1e9)
            if item.task_type == "prefill":
                wait = max(0.0, time.monotonic() - item.request.get("server_enqueue_monotonic", time.monotonic()))
                return (1.0 + wait) / max(1e-6, marginal * item.remaining_tokens / max(1, sl.draft_token_count))
            profile = progress_profiles.get(item.proc_id, default_profile)
            final = sl.offset + sl.draft_token_count >= item.total_tokens
            return profile.slice_progress(item.accepted_so_far, sl.draft_token_count, final) / marginal

        def category_for(item):
            return length_classifier.category(item)

        def build_plan(reserved_work_ids=(), cycle=None, cursor=None, projected_plan=None,
                       diagnostic_kind=None, excluded_proc_ids=(), planning_time=None):
            started = time.perf_counter()
            planning_time = time.monotonic() if planning_time is None else planning_time
            if projected_plan is None and diagnostic_kind != "lookahead":
                _sort_queues_if_dirty(now=planning_time)
            projected_queues = getattr(projected_plan, "_prediction_queues", None)
            queues = projected_queues if projected_queues is not None else task_queues.snapshot()
            thresholds = dict(length_classifier.thresholds)
            frozen_category = lambda item: _length_category(
                item.remaining_tokens, *thresholds[item.task_type])
            items = {item.work_id: item for cats in queues.values() for rows in cats.values() for item in rows}
            DIAG.context.plan_kind = diagnostic_kind or ("lookahead" if projected_plan is not None else "admission")
            DIAG.context.plan_id = time.monotonic_ns()
            kwargs = dict(
                token_budget=int(self.args.token_budget),
                current_cycle=scheduler_state["current_cycle"] if cycle is None else cycle,
                wrr_cursor=scheduler_state["wrr_cursor"] if cursor is None else cursor,
                max_num_seqs=int(self.args.batch_size),
                min_prefill_chunk_tokens=int(self.args.min_prefill_chunk_tokens),
                prefill_chunk_quantum=int(self.args.prefill_chunk_quantum),
                accept_stats=accept_stats_snapshot(),
                reserved_work_ids=reserved_work_ids,
                excluded_proc_ids=excluded_proc_ids,
                selection_policy=getattr(self.args, "scheduler_policy", "legacy"),
                latency_priority=bool(getattr(self.args, "enable_latency_priority", True)),
                candidate_value=lambda item, sl, selected: candidate_value(item, sl, selected, items),
                max_wait_s=float(getattr(self.args, "scheduler_max_wait_s", 1.0)))
            if diagnostic_kind == "lookahead" or projected_plan is not None:
                job = in_flight["job"] if projected_queues is None else None
                applied_ids = (() if job is None or not job.verify_applied else
                               tuple(sl.work_id for sl in job.plan.verify_slices))
                plan = predict_next_plan(
                    queues, projected_plan, category_for=frozen_category,
                    planning_time=planning_time,
                    known_phase_results=None if job is None else dict(job.progress),
                    completed_work_ids=() if job is None else set(job.completed),
                    applied_work_ids=applied_ids, **kwargs)
            else:
                plan = plan_iteration(queues, now=planning_time, **kwargs)
            metric = "lookahead_ms" if DIAG.context.plan_kind.startswith("lookahead") else "plan_ms"
            value = (time.perf_counter() - started) * 1000.0
            scheduler_metrics[metric + "_total"] = scheduler_metrics.get(metric + "_total", 0) + value
            scheduler_metrics[metric + "_count"] = scheduler_metrics.get(metric + "_count", 0) + 1
            scheduler_metrics[metric + "_max"] = max(scheduler_metrics.get(metric + "_max", 0), value)
            plan.ready_at = time.monotonic()
            plan._diag_id = DIAG.context.plan_id
            DIAG.emit("plan_built", plan_id=plan._diag_id, plan_kind=DIAG.context.plan_kind, duration_ms=value, queue_count=len(items), selected=plan.selected_work_ids, used_tokens=plan.used_tokens)
            if diagnostic_kind != "lookahead":
                # One budget sample per admitted plan, never per request.
                ensure_cache_management()
                observe_memory()
            return plan

        last_prediction = {"signature": None}
        prediction_compare_state = {"pids": set()}
        # source plan number -> ordered [pid set for +1, +2, ...] snapshots.
        # A snapshot is consumed by actual plans, not by the wall-clock tick.
        prediction_snapshots = {}
        # Earliest predicted horizon for each pid; used to label ready/pending
        # claims with 1/3/5/10-step prefetch lead time.
        prediction_horizon_by_pid = {}
        # Learned duration of one admitted plan.  Used only to move the
        # lookahead's priority clock to the expected start of the next plan;
        # it does not change the scheduler's scoring formula.
        forward_time_state = {"seconds": None, "samples": 0}

        def observe_forward_duration(duration_s):
            duration = max(0.001, min(float(duration_s), 10.0))
            if forward_time_state["samples"] == 0:
                forward_time_state["seconds"] = duration
            else:
                forward_time_state["seconds"] = (
                    0.8 * float(forward_time_state["seconds"]) + 0.2 * duration
                )
            forward_time_state["samples"] += 1
            scheduler_metrics["batch_compute_samples"] += 1
            scheduler_metrics["batch_compute_total_s"] += duration
            scheduler_metrics["batch_compute_avg_s"] = (
                scheduler_metrics["batch_compute_total_s"] /
                scheduler_metrics["batch_compute_samples"])

        def refresh_prefetch():
            """Plan the projected next batch, but only while ACTIVE.

            BYPASS must not build a lookahead plan, hash the queue or touch
            the transfer registry: residency means there is nothing to move.
            """

            if not prefetch_enabled or not getattr(self.args, "kv_prefetch", True):
                return
            if use_h1_prefetch and (not transfers.enabled or transfers.mode != "overlap"):
                return
            if not residency_manager.enabled or residency_manager.cpu_resident_count() == 0:
                # Nothing is offloaded, so there is nothing to prefetch.
                return
            job = in_flight["job"]
            # O(1) signature: queue version + acceptance version + plan tick.
            signature = (
                task_queues.version,
                accept_stats_version[0],
                plan_sequence["tick"],
                bool(job and job.verify_applied),
            )
            if signature == last_prediction["signature"]:
                return
            scheduler_metrics["prefetch_refresh_count"] += 1
            now = time.monotonic()
            estimate = forward_time_state["seconds"]
            planning_time = now
            horizon_ms = 0.0
            if job is not None and estimate is not None:
                elapsed = max(0.0, now - float(job.ready_at))
                remaining = max(0.0, float(estimate) - elapsed)
                planning_time = now + remaining
                horizon_ms = remaining * 1000.0
            scheduler_metrics["forward_estimate_s"] = (
                float(estimate) if estimate is not None else 0.0
            )
            scheduler_metrics["lookahead_horizon_ms"] = horizon_ms
            predicted_plans = []
            projected = in_flight["plan"]
            step_gap = float(estimate) if estimate is not None else 0.0
            prediction_tick = int(plan_sequence["tick"])
            lookahead_rounds = 1 if use_h1_prefetch else PREFETCH_LOOKAHEAD_ROUNDS
            for step in range(1, lookahead_rounds + 1):
                lookahead = build_plan(
                    projected_plan=projected,
                    diagnostic_kind="lookahead",
                    planning_time=planning_time + step_gap * (step - 1),
                )
                scheduler_metrics["lookahead_build_count"] += 1
                predicted_plans.append(lookahead)
                projected = lookahead
                if not lookahead.verify_proc_ids:
                    break

            # Preserve planner order across all projected rounds; the worker
            # treats this sequence as its transfer priority.
            ordered_pids = []
            horizons = {}
            seen_pids = set()
            for horizon, predicted in enumerate(predicted_plans, 1):
                for pid in predicted.verify_proc_ids:
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        ordered_pids.append(pid)
                        horizons[pid] = horizon

            first_lookahead = predicted_plans[0] if predicted_plans else None
            prediction_compare_state["pids"] = set(
                first_lookahead.verify_proc_ids if first_lookahead is not None else ())
            # A prediction made at tick T for tick T+step is consumed when that
            # future plan is actually admitted.  Keeping it keyed by tick lets
            # us report accuracy separately for 1/3/5/10-step horizons.
            # prediction_tick is the next plan number: the in-flight plan has
            # already consumed its tick.  Horizon 1 therefore occupies
            # prediction_tick itself, horizon 2 the following tick, etc.
            prediction_snapshots[prediction_tick] = [
                set(predicted.verify_proc_ids) for predicted in predicted_plans
            ]
            for stale in [tick for tick in prediction_snapshots
                          if tick < prediction_tick - lookahead_rounds]:
                prediction_snapshots.pop(stale, None)
            prediction_horizon_by_pid.clear()
            prediction_horizon_by_pid.update(horizons)
            scheduler_metrics["prefetch_lookahead_effective_rounds"] = max(
                scheduler_metrics.get("prefetch_lookahead_effective_rounds", 0),
                len(predicted_plans))

            protected = residency.protected_snapshot()
            # Only a CPU-resident cache can be prefetched; issuing for a cache
            # already on the GPU would mint an empty ticket every plan.
            safe_pids = []
            h2_count = 0
            for pid in ordered_pids:
                horizon = horizons[pid]
                if not use_h1_prefetch and (horizon > PREFETCH_HORIZON_LIMIT or
                        (horizon == 2 and h2_count >= PREFETCH_H2_MAX_PIDS)):
                    skipped = scheduler_metrics["prefetch_skipped_horizon"]
                    key = str(horizon)
                    skipped[key] = skipped.get(key, 0) + 1
                    continue
                if pid not in protected and residency_manager.state_of(pid) == CPU_RESIDENT:
                    safe_pids.append(PrefetchTarget(pid, horizon, prediction_tick))
                    h2_count += int(horizon == 2)
            on_gpu = sum(1 for pid in ordered_pids
                         if residency_manager.state_of(pid) == GPU_RESIDENT)
            if on_gpu:
                scheduler_metrics["prefetch_skipped_on_gpu"] = (
                    scheduler_metrics.get("prefetch_skipped_on_gpu", 0) + on_gpu)
            if first_lookahead is not None:
                DIAG.emit("prefetch_prediction", plan_id=getattr(first_lookahead, "_diag_id", None),
                          protected_proc_ids=sorted(protected), ordered_verify_pids=ordered_pids,
                          transfer_proc_ids=[target.pid for target in safe_pids],
                          transfer_horizons={str(target.pid): target.horizon for target in safe_pids},
                          assumptions=getattr(first_lookahead, "prediction_assumptions", ()),
                          next_cursor=first_lookahead.next_wrr_cursor,
                          next_cycle=first_lookahead.next_cycle,
                          policy=getattr(self.args, "scheduler_policy", "legacy"))
            if not ordered_pids and not task_queues.has_live_task("verify"):
                scheduler_metrics["prefetch_no_verify_candidates"] += 1
            # One intent covers this pid set; the worker decides how many
            # tickets it can actually prepare.
            if prefetch_coordinator.refresh(safe_pids, prediction_tick=prediction_tick):
                scheduler_metrics["prefetch_intents_issued"] = (
                    scheduler_metrics.get("prefetch_intents_issued", 0) + 1)
            reserved_prefetch = (h1_prefetch.reserved_bytes if h1_prefetch is not None else
                                 int(transfers.metrics.get("prefetch_reserved_bytes", 0) or 0))
            scheduler_metrics["max_prefetch_reserved_bytes"] = max(
                scheduler_metrics.get("max_prefetch_reserved_bytes", 0), reserved_prefetch
            )
            scheduler_metrics["max_total_kv_plus_prefetch_bytes"] = max(
                scheduler_metrics.get("max_total_kv_plus_prefetch_bytes", 0),
                residency_manager.used_bytes() + reserved_prefetch,
            )
            last_prediction["signature"] = signature
            scheduler_metrics["prefetch_prediction_versions"] = scheduler_metrics.get("prefetch_prediction_versions", 0) + 1

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
            # Compare the lookahead prediction with the plan that actually
            # got admitted.  This is separate from KV readiness: a correct
            # prediction can still miss if the transfer arrives too late.
            actual_verify_pids = {sl.proc_id for sl in plan.verify_slices}
            predicted_pids = set(prediction_compare_state.get("pids") or ())
            if predicted_pids:
                prediction_hits = predicted_pids & actual_verify_pids
                prediction_misses = actual_verify_pids - predicted_pids
                prediction_wasted = predicted_pids - actual_verify_pids
                scheduler_metrics["prefetch_prediction_plans"] += 1
                scheduler_metrics["prefetch_prediction_predicted"] += len(predicted_pids)
                scheduler_metrics["prefetch_prediction_actual"] += len(actual_verify_pids)
                scheduler_metrics["prefetch_prediction_hits"] += len(prediction_hits)
                scheduler_metrics["prefetch_prediction_misses"] += len(prediction_misses)
                scheduler_metrics["prefetch_prediction_wasted"] += len(prediction_wasted)
                prediction_compare_state["pids"] = set()

            current_tick = int(plan_sequence["tick"])
            for source_tick, horizon_plans in list(prediction_snapshots.items()):
                delta = current_tick - source_tick + 1
                if delta < 1:
                    continue
                if delta > len(horizon_plans):
                    prediction_snapshots.pop(source_tick, None)
                    continue
                horizon_pids = horizon_plans[delta - 1]
                bucket = scheduler_metrics["prefetch_prediction_by_horizon"].setdefault(
                    str(delta),
                    {"plans": 0, "predicted": 0, "actual": 0,
                     "hits": 0, "misses": 0, "wasted": 0},
                )
                bucket["plans"] += 1
                scheduler_metrics["prediction_snapshots_consumed"] += 1
                bucket["predicted"] += len(horizon_pids)
                bucket["actual"] += len(actual_verify_pids)
                bucket["hits"] += len(horizon_pids & actual_verify_pids)
                bucket["misses"] += len(actual_verify_pids - horizon_pids)
                bucket["wasted"] += len(horizon_pids - actual_verify_pids)
            ready_at = getattr(plan, "ready_at", time.monotonic())
            DIAG.emit("plan_submitted", plan_id=getattr(plan,"_diag_id",None), ready_at=ready_at, selected=[diag_req(r) for r in verify_batch+prefill_batch])
            reserve_admission_plan(plan)
            plan_sequence["tick"] += 1
            tickets = {}
            if residency_manager.active:
                transfers.admit()
                with cache_lock:
                    residency.reserve(reserved_pids)
                    for req in verify_batch + prefill_batch:
                        pid = req["proc_id"]
                        req["_plan_ready_at"] = ready_at
                        req["_prediction_horizon"] = prediction_horizon_by_pid.get(pid)
                        cache = kv_cache_manager._past_key_values.get(pid)
                        if cache is not None and not use_h1_prefetch:
                            ticket = residency.claim_locked(pid, cache, id(plan))
                            if ticket is not None:
                                ready = ticket.ready()
                                is_verify = req.get("task_type") == "verify"
                                detail_key = (
                                    "prefetch_verify_ready_hit" if ready else "prefetch_verify_pending_hit"
                                ) if is_verify else (
                                    "prefetch_prefill_ready_hit" if ready else "prefetch_prefill_pending_hit"
                                )
                                scheduler_metrics[detail_key] = scheduler_metrics.get(detail_key, 0) + 1
                                if is_verify:
                                    key = "prefetch_ready_hit" if ready else "prefetch_pending_hit"
                                    scheduler_metrics[key] = scheduler_metrics.get(key, 0) + 1
                                    scheduler_metrics["prefetch_claims"] = scheduler_metrics.get("prefetch_claims", 0) + 1
                                    intent_at = getattr(ticket, "intent_at", None)
                                    claimed_at = getattr(ticket, "claimed_at", None)
                                    if intent_at is not None and claimed_at is not None:
                                        lead_ms = max(0.0, (claimed_at - intent_at) * 1000.0)
                                        scheduler_metrics["prefetch_claim_lead_ms_sum"] = (
                                            scheduler_metrics.get("prefetch_claim_lead_ms_sum", 0.0) + lead_ms
                                        )
                                        scheduler_metrics["prefetch_claim_lead_ms_max"] = max(
                                            scheduler_metrics.get("prefetch_claim_lead_ms_max", 0.0), lead_ms
                                        )
                                horizon = ticket.horizon
                                if horizon is not None:
                                    state_name = "ready" if ready else "pending"
                                    if is_verify:
                                        horizon_key = (
                                            "prefetch_ready_by_horizon" if ready
                                            else "prefetch_pending_by_horizon"
                                        )
                                    else:
                                        horizon_key = (
                                            "prefetch_prefill_ready_by_horizon" if ready
                                            else "prefetch_prefill_pending_by_horizon"
                                        )
                                    bucket = scheduler_metrics[horizon_key].setdefault(
                                        str(horizon), {state_name: 0}
                                    )
                                    bucket[state_name] += 1
                                tickets[pid] = req["_cache_ticket"] = ticket
            else:
                for req in verify_batch + prefill_batch:
                    req["_plan_ready_at"] = ready_at
            job = _ExecutionJob(
                plan=plan, item_by_id=item_by_id, verify_batch=verify_batch,
                prefill_batch=prefill_batch, reserved_pids=reserved_pids,
                cache_tickets=tickets, ready_at=ready_at)
            in_flight["plan"] = plan
            in_flight["job"] = job
            scheduler_metrics["iterations"] += 1
            try:
                if PREFETCH_BEFORE_SUBMIT and not use_h1_prefetch:
                    refresh_prefetch()
                executor.submit(job)
                scheduler_metrics["actual_plans"] += 1
            except BaseException:
                in_flight["plan"] = None
                in_flight["job"] = None
                if residency_manager.active:
                    residency.release(reserved_pids)
                    transfers.release_admission()
                    for ticket in tickets.values():
                        transfers.retire(ticket)
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
            service = (getattr(item, "forward_gpu_ms_total", 0.0) if compute_stream is not None
                       else item.verify_ms_total) / 1000.0
            elapsed = max(0.0, time.monotonic() - item.request.get("server_enqueue_monotonic", time.monotonic()))
            response.update(gamma_control.finish(item.proc_id,
                accepted=item.accepted_so_far, proposed=item.total_tokens,
                request=item.request, verify_s=service, queue_s=max(0, elapsed - service)))
            response["verify_ms"] = item.verify_ms_total
            if getattr(item, "verify_gpu_ms_total", None) is not None:
                response["verify_gpu_ms"] = item.verify_gpu_ms_total
            for metric in ("forward_gpu_ms", "exposed_h2d_stall_ms", "batch_start_delay_ms"):
                response[metric] = getattr(item, metric + "_total", 0.0)
            response["request_e2e_ms"] = elapsed * 1000.0
            scheduler_metrics["gamma_rounds"] = gamma_control.rounds
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
                for metric in ("forward_gpu_ms", "exposed_h2d_stall_ms", "batch_start_delay_ms"):
                    setattr(item, metric + "_total", getattr(item, metric + "_total", 0.0) + float(response.get(metric, 0.0)))
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
                phase_response = prefill_responses.get(item.proc_id, {})
                for metric in ("forward_gpu_ms", "exposed_h2d_stall_ms", "batch_start_delay_ms"):
                    setattr(item, metric + "_total", getattr(item, metric + "_total", 0.0) + float(phase_response.get(metric, 0.0)))
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
                    for metric in ("forward_gpu_ms", "exposed_h2d_stall_ms", "batch_start_delay_ms"):
                        response[metric] = getattr(item, metric + "_total", 0.0)
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
                    lambda plan, **kwargs: commit_admission_plan(plan,
                        all_items=work_items.values(), category_for=category_for, **kwargs),
                    completed_work_ids=job.completed,
                    progress=job.progress,
                )
                committed = True

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
                in_flight["job"] = None
                residency.release(job.reserved_pids)
                raise
            finally:
                if outcome.final or outcome.error is not None:
                    if outcome.final and outcome.error is None:
                        observe_forward_duration(time.monotonic() - job.ready_at)
                    in_flight["plan"] = None
                    in_flight["job"] = None
                    residency.release(job.reserved_pids)
                    if residency_manager.active:
                        transfers.release_admission()
                        for ticket in job.cache_tickets.values():
                            transfers.retire(ticket)

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
                category="short",
                cycle=scheduler_state["current_cycle"],
                work_id=work_id,
            )
            item.category = length_classifier.observe(item)
            DIAG.emit("length_classification", work_id=work_id, task_type=item.task_type,
                      task_tokens=item.total_tokens, category=item.category,
                      thresholds=length_classifier.thresholds[item.task_type])
            update = pending_prefill_timing.pop(req["proc_id"], None)
            if update is not None:
                req = dict(req)
                req.update(update)
                item.request = req
            work_items[work_id] = item
            task_queues.enqueue(item)
            DIAG.emit("scheduler_enqueue", **diag_item(item), queue_depth=len(work_items))

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
            energy_campaign_started = self._start_energy_campaign(
                energy_service,
                metadata={
                    "worker_mode": "target_batching",
                    "scheduler_mode": getattr(self.args, "server_sched_mode", None),
                },
            )
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
                            req["_plan_ready_at"] = time.monotonic()
                            # FCFS bypasses submit_plan(), so it must advance
                            # the same recency clock and sample the same budget
                            # itself. A frozen tick protects every previously
                            # used cache forever and silently defeats eviction.
                            plan_sequence["tick"] += 1
                            ensure_cache_management()
                            observe_memory()
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
            # The KV prefetch thread is started lazily by the residency
            # manager, on the first BYPASS -> ACTIVE transition.
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
            if h1_prefetch is not None:
                h1_prefetch.close()
            transfers.close()
            if h1_prefetch is not None:
                # Include whole-ticket H2D in the same totals as demand/D2H;
                # otherwise the paired report would hide speculative traffic.
                copied = scheduler_metrics["prefetch_bytes"]
                copies = scheduler_metrics["prefetch_copy_count"]
                transfers.metrics["h2d_bytes"] += copied
                transfers.metrics["copies"] += copies
                transfers.metrics["speculative_submitted_bytes"] += copied
                transfers.metrics["prefetch_bytes_by_horizon"] = {"1": copied}
                transfers.metrics["speculative_bytes_by_horizon"] = {"1": copied}
                transfers.metrics["prefetch_reserved_bytes_peak"] = scheduler_metrics["prefetch_reserved_bytes_peak"]
                transfers.metrics["kv_transfer_prefetch_avg_s"] = (
                    scheduler_metrics["prefetch_copy_ms"] / copies / 1000 if copies else 0.0)
            scheduler_metrics["cache_transfers"] = dict(transfers.metrics)
            scheduler_metrics["prefetch_bytes_by_horizon"] = dict(transfers.metrics["prefetch_bytes_by_horizon"])
            scheduler_metrics["kv_residency"] = residency_manager.stats()
            scheduler_metrics["kv_overlap_controller"] = transfers.controller_status()
            scheduler_metrics["kv_prefetch_effective_mode"] = transfers.mode
            actual_plans = scheduler_metrics["actual_plans"]
            ready = scheduler_metrics["prefetch_ready_hit"]
            scheduler_metrics["ready_per_actual_plan"] = ready / actual_plans if actual_plans else None
            scheduler_metrics["prefetch_verify_ready_per_claim"] = (
                scheduler_metrics.get("prefetch_verify_ready_hit", 0) / claims
                if (claims := scheduler_metrics.get("prefetch_claims", 0)) else None
            )
            scheduler_metrics["prefetch_verify_pending_per_claim"] = (
                scheduler_metrics.get("prefetch_verify_pending_hit", 0) / claims
                if claims else None
            )
            scheduler_metrics["prefetch_verify_miss_per_claim"] = (
                scheduler_metrics.get("prefetch_verify_demand_miss", 0) / claims
                if claims else None
            )
            scheduler_metrics["prefetch_claim_lead_ms_avg"] = (
                scheduler_metrics.get("prefetch_claim_lead_ms_sum", 0.0) / claims
                if claims else None
            )
            scheduler_metrics["prefetch_arena_bytes"] = int(
                transfers.metrics.get("prefetch_pool_gpu_bytes", 0) or 0
            )
            pressure_samples = scheduler_metrics.get("residency_pressure_samples", 0)
            scheduler_metrics["residency_used_bytes_avg"] = (
                scheduler_metrics.get("residency_used_bytes_sum", 0) / pressure_samples
                if pressure_samples else None
            )
            scheduler_metrics["prefetch_reserved_bytes_peak"] = int(
                transfers.metrics.get("prefetch_reserved_bytes_peak", 0) or 0
            )
            scheduler_metrics["h2d_bytes_per_ready"] = transfers.metrics["h2d_bytes"] / ready if ready else None
            scheduler_metrics["copies_per_ready"] = transfers.metrics["copies"] / ready if ready else None
            for bucket in scheduler_metrics["prefetch_prediction_by_horizon"].values():
                bucket["precision"] = bucket["hits"] / bucket["predicted"] if bucket["predicted"] else None
                bucket["recall"] = bucket["hits"] / bucket["actual"] if bucket["actual"] else None
                bucket["coverage_per_actual_plan"] = bucket["plans"] / actual_plans if actual_plans else None
                bucket["hits_per_actual_plan"] = bucket["hits"] / actual_plans if actual_plans else None
            scheduler_metrics["batch_cost_profiles"] = {str(k): v for k, v in batch_cost.table.items()}
            self._stop_energy_campaign(
                energy_service,
                energy_campaign_started,
                metadata={
                    "worker_mode": "target_batching",
                    "scheduler_mode": getattr(self.args, "server_sched_mode", None),
                },
            )
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
