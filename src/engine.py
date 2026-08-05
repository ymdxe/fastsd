import os
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
from .kvcache4RC import KVCacheModel as KVCache2Model
from .cache_offload_policy import should_offload_target_cache
from .util import seed_everything, norm_logits, sample, max_fn
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
    FASTSD_DEFAULT_R1,
    FASTSD_DEFAULT_R2,
    FASTSD_DYNAMIC_WINDOW,
    OversizeVerifyError,
    UnifiedSchedulerState,
    advance_prefill_cursor,
    build_unified_schedule_plan,
    compute_priority_score as _compute_priority_score,
    drain_queue_bounded,
    length_category as _length_category,
    split_plan_microbatches,
    update_length_thresholds as _update_length_thresholds,
    uses_strict_fcfs,
)


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
    
    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}", 3)
        if self.args.eval_mode == "small":
            self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        elif self.args.eval_mode == "sd":
            # self.draft_model = AutoModelForCausalLM.from_pretrained(self.args.draft_model, device_map="cuda:0", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
            # self.target_model = AutoModelForCausalLM.from_pretrained(self.args.target_model, device_map="cuda:1", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()

            self.draft_model = AutoGPTQForCausalLM.from_quantized(
                self.args.draft_model,
                device="cuda:1",
                use_safetensors=True,
                trust_remote_code=True,
            )

            self.target_model = AutoGPTQForCausalLM.from_quantized(
                self.args.target_model,
                device="cuda:0",
                use_safetensors=True,
                trust_remote_code=True,
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
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def _service_target_device(self) -> str:
        return os.environ.get("FASTSD_TARGET_DEVICE", "cuda:0")

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.draft_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.draft_model, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        
        # for llama models
        self.tokenizer.pad_token_id = 2

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
        return x

    @torch.no_grad()
    def speculative_decoding(self, prefix):
        max_tokens = prefix.shape[1] + self.args.max_tokens
        
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        
        approx_model_cache = KVCacheModel(self.draft_model, self.args.temp, self.args.top_k, self.args.top_p)
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(self.target_model, self.args.temp, self.args.top_k, self.args.top_p)
        target_model_cache.vocab_size = self.vocab_size

        while prefix.shape[1] < max_tokens:
            prefix_len = prefix.shape[1]
            x = approx_model_cache.generate(prefix.to(draft_device), self.args.gamma)
            _ = target_model_cache.generate(x.to(target_device), 1)
            if self.accelerator.is_main_process:
                self.draft_forward_times += self.args.gamma
                self.target_forward_times += 1
            
            n = prefix_len + self.args.gamma - 1
            for i in range(self.args.gamma):
                j = x[:, prefix_len + i]
                # target 使用贪心策略得到当前步 token，与 draft token 不一致则在前一位置截断
                target_logits = target_model_cache._prob_history[
                    :, prefix_len + i - 1, :self.vocab_size
                ].to(draft_device)
                greedy_token = torch.argmax(target_logits, dim=-1)  # (1,)
                if j.item() != greedy_token.item():
                    n = prefix_len + i - 1
                    break

            self.num_acc_tokens.append(n - prefix_len + 1)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, :n + 1]
            
            approx_model_cache.rollback(n+1)

            if n < prefix_len + self.args.gamma - 1:
                # 存在拒绝：在位置 n 上，使用 target 模型的贪心 token 作为 new_token
                target_logits_next = target_model_cache._prob_history[
                    :, n, :self.vocab_size
                ].to(draft_device)
                t = torch.argmax(target_logits_next, dim=-1).unsqueeze(-1)
                target_model_cache.rollback(n + 1)
            else:
                # 所有 draft token 被接受：在最后一步的位置上，使用 target 贪心 token 作为 new_token
                target_logits_next = target_model_cache._prob_history[
                    :, -1, :self.vocab_size
                ].to(draft_device)
                t = torch.argmax(target_logits_next, dim=-1).unsqueeze(-1)
                target_model_cache.rollback(n + 2)
            prefix = torch.cat((prefix, t), dim=1)
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
            approx_model_cache.vocab_size = tokenizer.vocab_size

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
                cache.vocab_size = tokenizer.vocab_size
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

            _ = cache.generate(x_for_target, 1)

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

            n = prefix_len + self.args.gamma - 1
            for i in range(self.args.gamma):
                if tail_only:
                    tail_offset = 1 if has_bridge_token else 0
                    j = x[:, tail_offset + i]
                else:
                    j = x[:, prefix_len + i]
                # target 使用贪心策略，若与 draft token 不一致则在前一位置截断
                target_logits = cache._prob_history[:, prefix_len + i - 1, :self.vocab_size]
                greedy_token = torch.argmax(target_logits, dim=-1)
                if j.item() != greedy_token.item():
                    n = prefix_len + i - 1
                    break

            accepted_len = n + 1
            accepted_cnt = accepted_len - prefix_len
            if accepted_cnt < self.args.gamma:
                # 存在拒绝：在位置 n 上使用 target 贪心 token 作为 new_token
                target_logits_next = cache._prob_history[:, n, :self.vocab_size]
                new_token = torch.argmax(target_logits_next, dim=-1).unsqueeze(-1)
            else:
                # 所有 draft token 被接受：在最后一步位置上使用 target 贪心 token
                target_logits_next = cache._prob_history[:, -1, :self.vocab_size]
                new_token = torch.argmax(target_logits_next, dim=-1).unsqueeze(-1)

            # Keep only accepted prefix in target cache.
            # final_token will be consumed in the next verify call together with new draft tokens.
            cache.rollback(accepted_len)
            if should_offload_target_cache("verify", pid=proc_id):
                cache._past_key_values = move_dynamic_cache_to(cache._past_key_values, "cpu")

            response_queues[proc_id].put({
                "accepted": accepted_len,
                "final_token": new_token,
            })

            # Update accept statistics
            accept_stats[proc_id][0] += accepted_len - prefix_len
            accept_stats[proc_id][1] += self.args.gamma

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
        kv_cache_manager.vocab_size = tokenizer.vocab_size

        # --------------------------- 队列与统计 ---------------------------
        task_queues = {
            "prefill": {"short": deque(), "mid": deque(), "long": deque()},
            "verify": {"short": deque(), "mid": deque(), "long": deque()},
        }
        accept_stats = defaultdict(lambda: [0, 1])  # {proc_id: [accepted_sum, total_sum]}
        committed_prefix_tokens = {}  # {proc_id: List[int]} for debug context display
        verify_time_ema = {}
        edge_draft_per_token_ema = {}
        edge_rtt_ema = {}
        recent_prefix_lens = deque(maxlen=FASTSD_DYNAMIC_WINDOW)
        len_r1 = FASTSD_DEFAULT_R1
        len_r2 = FASTSD_DEFAULT_R2
        cache_lock = threading.Lock()
        preloaded_gpu_pids = set()
        scheduler_state = UnifiedSchedulerState(
            prefill_max_wait_cycles=getattr(self.args, "prefill_max_wait_cycles", 2),
        )

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

        # 将 DynamicCache 中的每一层的 key/value 移动到 CPU 上
        def move_dynamic_cache_to(cache, device):
            legacy_cache = cache.to_legacy_cache()
            current_device = legacy_cache[0][0].device  # 任意一层 key 的 device 作为参考

            if str(current_device) == device:
                return cache  # 已在目标设备上，无需迁移

            new_cache = []
            for k, v in legacy_cache:
                new_cache.append((k.to(device), v.to(device)))
            return DynamicCache.from_legacy_cache(new_cache)

        # --------------------- 批量处理（Prefill / Verify） ----------------
        def handle_request_batch(batch: list[dict]):
            """
            batch : 同一种 task_type (全部 prefill or 全部 verify)
            """
            proc_ids = [req["proc_id"] for req in batch]
            prefix_len = [req["prefix_len"] for req in batch]

            if batch[0]["task_type"] == "prefill":
                dispatch_monotonic = time.monotonic()
                for req in batch:
                    req.setdefault(
                        "prefill_first_dispatch_monotonic", dispatch_monotonic
                    )
                chunk_ranges = [
                    (
                        int(req.get("_prefill_chunk_start", req.get("prefill_cursor", 0))),
                        int(req.get("_prefill_chunk_end", req["prefix_len"])),
                    )
                    for req in batch
                ]
                seqs = [
                    req["draft_output"][:, chunk_start:chunk_end].to(target_model.device)
                    for req, (chunk_start, chunk_end) in zip(batch, chunk_ranges)
                ]
                input_lens = [x.shape[1] for x in seqs]
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

                # Fresh Prefill creates state; continuation Prefill reuses exactly
                # the cache accumulated up to its persistent cursor.
                with cache_lock:
                    for pid, (chunk_start, _) in zip(proc_ids, chunk_ranges):
                        preloaded_gpu_pids.discard(pid)
                        if chunk_start == 0:
                            kv_cache_manager.reset(pid)
                        else:
                            cache = kv_cache_manager._past_key_values.get(pid)
                            if cache is None:
                                raise ValueError(
                                    f"missing Prefill continuation cache for proc_id={pid!r}"
                                )
                            kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(
                                cache, target_model.device
                            )

                prefill_metadata = kv_cache_manager.prefill_chunks(
                    x_batch,
                    proc_ids=proc_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    input_lens=input_lens,
                    cursors=[chunk_start for chunk_start, _ in chunk_ranges],
                )

                with cache_lock:
                    for pid in proc_ids:
                        cache = kv_cache_manager._past_key_values[pid]
                        if should_offload_target_cache("prefill", pid=pid):
                            kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(cache, "cpu")

                unfinished_prefills = []
                for req, metadata, (_, chunk_end) in zip(
                    batch, prefill_metadata, chunk_ranges
                ):
                    pid = req["proc_id"]
                    req["prefill_chunks_completed"] = int(
                        req.get("prefill_chunks_completed", 0)
                    ) + 1
                    if int(metadata["cursor_end"]) != chunk_end:
                        raise AssertionError(
                            f"Prefill cursor mismatch for proc_id={pid!r}: "
                            f"expected {chunk_end}, got {metadata['cursor_end']}"
                        )
                    prefill_complete = advance_prefill_cursor(req, chunk_end)
                    if not prefill_complete:
                        unfinished_prefills.append(
                            (req.get("_fastsd_category"), req)
                        )
                        continue

                    kv_cache_manager.rollback(pid, int(req["prefix_len"]))
                    committed_prefix_tokens[pid] = req["draft_output"][
                        0, :int(req["prefix_len"])
                    ].tolist()
                    completed_monotonic = time.monotonic()
                    enqueue_monotonic = float(
                        req.get("server_enqueue_monotonic", completed_monotonic)
                    )
                    first_dispatch = float(
                        req.get(
                            "prefill_first_dispatch_monotonic", completed_monotonic
                        )
                    )
                    response_queues[pid].put({
                        "status": "prefill_ok",
                        "prefill_chunks": req["prefill_chunks_completed"],
                        "prefill_queue_ms": max(
                            0.0, (first_dispatch - enqueue_monotonic) * 1000.0
                        ),
                        "prefill_service_ms": max(
                            0.0, (completed_monotonic - enqueue_monotonic) * 1000.0
                        ),
                    })

                self.color_print(
                    f"process prefill chunks from: {proc_ids} ranges={chunk_ranges}", 3
                )
                return unfinished_prefills
            else:  # verify 批量
                seqs = [req["draft_output"].to(target_model.device) for req in batch]
                x_batch = []
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
                    else:
                        x_batch.append(x)
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

                verify_compute_start = time.time()
                _ = kv_cache_manager.generate(x_batch, 1, proc_ids=proc_ids, pad_token_id=tokenizer.pad_token_id, is_prefill=False)
                verify_elapsed = time.time() - verify_compute_start

                # verify之后，将KV cache移到CPU节省显存
                with cache_lock:
                    for pid in proc_ids:
                        if not should_offload_target_cache("verify", pid=pid, pinned_gpu_pids=preloaded_gpu_pids):
                            continue
                        cache = kv_cache_manager._past_key_values[pid]
                        kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(cache, "cpu")

                # self.color_print(f"process verify tasks from: {proc_ids}", 3)

            # ---------- 逐样本接受 / 采样 ---------------------------------
            probs_full = [kv_cache_manager._prob_history[pid] for pid in proc_ids]  # (1, *, V) each
            for idx, req in enumerate(batch):
                pid = req["proc_id"]
                prefix_len = req["prefix_len"]
                probs = probs_full[idx]  # (1, L, V)
                x = req["draft_output"].to(target_model.device)
                tail_only = req.get("tail_only", False)
                has_bridge_token = req.get("has_bridge_token", False)
                req_gamma = max(1, int(req.get("gamma", self.args.gamma) or self.args.gamma))

                # 验证 γ 个 token：target 贪心 token 与 draft token 不一致则在前一位置截断
                n = prefix_len + req_gamma - 1
                mismatch_pos = None
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
                for i in range(req_gamma):
                    if tail_only:
                        tail_offset = 1 if has_bridge_token else 0
                        j = x[:, tail_offset + i]
                    else:
                        j = x[:, prefix_len + i]
                    target_logits = probs[:, prefix_len + i - 1, :self.vocab_size]
                    greedy_token = torch.argmax(target_logits, dim=-1)  # (1,)
                    draft_token_id = int(j.item())
                    target_token_id = int(greedy_token.item())

                    if debug_enabled and i < debug_max_steps:
                        match_flag = "MATCH" if draft_token_id == target_token_id else "MISMATCH"
                        self.color_print(
                            f"[VERIFY][pid={pid}] step={i} pos={prefix_len + i} "
                            f"draft={token_repr(draft_token_id)} "
                            f"target={token_repr(target_token_id)} {match_flag}",
                            6,
                        )

                    if j.item() != greedy_token.item():
                        n = prefix_len + i - 1
                        mismatch_pos = prefix_len + i
                        break

                accepted_len = n + 1
                accepted_cnt = accepted_len - prefix_len
                if accepted_cnt < req_gamma:
                    # 存在拒绝：在位置 n 上使用 target 贪心 token 作为 new_token
                    target_logits_next = probs[:, n, :self.vocab_size]
                    new_token = torch.argmax(target_logits_next, dim=-1).unsqueeze(-1)
                else:
                    # 所有 draft token 被接受：在最后一步位置上使用 target 贪心 token
                    target_logits_next = probs[:, -1, :self.vocab_size]
                    new_token = torch.argmax(target_logits_next, dim=-1).unsqueeze(-1)

                # Keep only accepted prefix in target cache.
                # final_token will be consumed in the next verify call together with new draft tokens.
                kv_cache_manager.rollback(pid, accepted_len)

                # 更新统计并回传
                accept_stats[pid][0] += accepted_len - prefix_len
                accept_stats[pid][1] += req_gamma
                if pid in committed_prefix_tokens:
                    if tail_only:
                        req_tokens = x[0].tolist()
                        draft_start_idx = 1 if has_bridge_token else 0
                        accepted_draft_tokens = req_tokens[draft_start_idx:draft_start_idx + accepted_cnt]
                        committed_prefix_tokens[pid].extend(accepted_draft_tokens)
                        committed_prefix_tokens[pid].append(int(new_token.item()))
                    else:
                        committed_prefix_tokens[pid] = x[0, :accepted_len].tolist()
                        committed_prefix_tokens[pid].append(int(new_token.item()))
                if debug_enabled:
                    accepted_cnt = accepted_len - prefix_len
                    mismatch_info = mismatch_pos if mismatch_pos is not None else "none"
                    self.color_print(
                        f"[VERIFY][pid={pid}] prefix_len={prefix_len} accepted={accepted_cnt}/{req_gamma} "
                        f"accepted_len={accepted_len} mismatch_pos={mismatch_info} "
                        f"new_token={token_repr(int(new_token.item()))}",
                        2,
                    )

                response_payload = {
                    "accepted": accepted_len,
                    "final_token": new_token,
                    "verify_ms": verify_elapsed * 1000.0,
                }
                if getattr(self.args, "enable_pipeline", True) and getattr(self.args, "pipeline_gamma_adapt", True):
                    # Target: T_verify ~= T_edge_draft + RTT.
                    # Use online EMA to suggest per-session gamma for next round.
                    drafted_tokens = max(1, req_gamma)
                    edge_per_tok = float(req.get("lag", 0.0)) / drafted_tokens
                    avg_cloud_total_ms = max(
                        0.0, float(req.get("avg_cloud_total_ms", 0.0))
                    )
                    # Use global average cloud_total_ms across all tasks when available.
                    # Fallback to local verify elapsed for warmup.
                    verify_budget = (
                        avg_cloud_total_ms / 1000.0 if avg_cloud_total_ms > 0.0 else verify_elapsed
                    )
                    transport_rtt = max(
                        0.0, float(req.get("transport_rtt", req.get("edge_rtt", 0.0)))
                    )
                    v_ema = update_ema(verify_time_ema, pid, max(1e-6, verify_budget))
                    d_ema = update_ema(edge_draft_per_token_ema, pid, max(1e-6, edge_per_tok))
                    rtt_ema = update_ema(edge_rtt_ema, pid, transport_rtt)
                    target_draft_time = max(1e-6, v_ema - rtt_ema)
                    suggested = int(round(target_draft_time / d_ema))
                    min_gamma = int(getattr(self.args, "pipeline_gamma_min", 1))
                    max_gamma = int(getattr(self.args, "pipeline_gamma_max", 16))
                    suggested = max(min_gamma, min(max_gamma, suggested))
                    response_payload["suggested_gamma"] = suggested
                    if getattr(self.args, "debug_pipeline", False):
                        self.color_print(
                            f"[PIPELINE-CLOUD][pid={pid}] req_gamma={req_gamma} "
                            f"accepted={accepted_cnt}/{req_gamma} verify={verify_elapsed*1000:.2f}ms "
                            f"avg_cloud_total={avg_cloud_total_ms:.2f}ms "
                            f"transport_rtt={transport_rtt*1000:.2f}ms v_ema={v_ema*1000:.2f}ms "
                            f"d_ema={d_ema*1000:.4f}ms/tok rtt_ema={rtt_ema*1000:.2f}ms "
                            f"target_draft={target_draft_time*1000:.2f}ms suggested_gamma={suggested}",
                            3,
                        )

                response_queues[pid].put(response_payload)

            return []

        def sort_task_queues():
            now = time.time()
            for ttype in ["verify", "prefill"]:
                for cat in task_queues[ttype]:
                    items = sorted(
                        task_queues[ttype][cat],
                        key=lambda r: _compute_priority_score(r, accept_stats, now=now),
                        reverse=True,
                    )
                    task_queues[ttype][cat] = deque(items)

        def preload_kvcache(proc_ids):
            if not proc_ids:
                return
            with cache_lock:
                for pid in proc_ids:
                    cache = kv_cache_manager._past_key_values.get(pid)
                    if cache is None:
                        continue
                    kv_cache_manager._past_key_values[pid] = move_dynamic_cache_to(cache, target_model.device)
                    preloaded_gpu_pids.add(pid)

        def run_active_batch(batch):
            if not batch:
                return []
            if energy_service is not None:
                energy_service.enter_active()
            try:
                return handle_request_batch(batch)
            finally:
                if energy_service is not None:
                    energy_service.exit_active()

        def materialize_entry(entry):
            req = entry.request
            req["_fastsd_category"] = entry.category
            if entry.task_type == "prefill":
                req["_prefill_chunk_start"] = entry.chunk_start
                req["_prefill_chunk_end"] = entry.chunk_end
            return req

        def execute_plan(plan):
            verify_microbatches = split_plan_microbatches(
                plan.verify_entries, getattr(self.args, "max_num_seqs", 4)
            )
            prefill_microbatches = split_plan_microbatches(
                plan.prefill_entries, getattr(self.args, "max_num_seqs", 4)
            )

            if verify_microbatches:
                preload_kvcache(
                    [entry.request_id for entry in verify_microbatches[0]]
                )
            for index, entries in enumerate(verify_microbatches):
                preload_thread = None
                if index + 1 < len(verify_microbatches):
                    next_proc_ids = [
                        entry.request_id for entry in verify_microbatches[index + 1]
                    ]
                    preload_thread = threading.Thread(
                        target=preload_kvcache,
                        args=(next_proc_ids,),
                        daemon=True,
                    )
                    preload_thread.start()
                run_active_batch([materialize_entry(entry) for entry in entries])
                if preload_thread is not None:
                    preload_thread.join()

            for entries in prefill_microbatches:
                unfinished = run_active_batch(
                    [materialize_entry(entry) for entry in entries]
                )
                for category, req in unfinished:
                    if category not in task_queues["prefill"]:
                        raise ValueError(
                            f"invalid Prefill category for requeue: {category!r}"
                        )
                    task_queues["prefill"][category].append(req)

            if getattr(self.args, "debug_pipeline", False):
                self.color_print(
                    f"[FASTSD] plan tokens={plan.used_tokens}/{plan.token_budget} "
                    f"verify={len(plan.verify_entries)} prefill={len(plan.prefill_entries)} "
                    f"cursor={plan.start_cursor}->{plan.end_cursor} "
                    f"cycles={plan.completed_cycles}",
                    3,
                )

        def reject_oversize_verify(exc):
            req = exc.request
            removed = False
            for category, pending in task_queues["verify"].items():
                retained = deque()
                while pending:
                    item = pending.popleft()
                    if not removed and item is req:
                        removed = True
                        continue
                    retained.append(item)
                task_queues["verify"][category] = retained
            response_queues[req["proc_id"]].put({
                "error": "verify_exceeds_token_budget",
                "detail": str(exc),
                "effective_tokens": exc.cost,
                "token_budget": exc.token_budget,
            })

        # ---------------------- 主循环 -------------------------------
        sched_mode = getattr(self.args, "server_sched_mode", "fastsd")
        use_strict_fcfs = uses_strict_fcfs(sched_mode)

        # Pipeline baseline: strict FCFS, single-request handling only.
        # No queue categorization, no batching, no preload, no priority scheduling.
        try:
            if use_strict_fcfs:
                while True:
                    try:
                        req = request_queue.get(timeout=0.01)
                        if req is None:
                            return
                        if energy_service is not None:
                            energy_service.enter_active()
                        try:
                            handle_request_batch([req])
                        finally:
                            if energy_service is not None:
                                energy_service.exit_active()
                    except queue.Empty:
                        time.sleep(0.01)
                return

            while True:
                ingested, stop_received = drain_queue_bounded(
                    request_queue,
                    max_items=6 * getattr(self.args, "max_num_seqs", 4),
                )
                if stop_received:
                    return
                for req in ingested:
                    task_type = req.get("task_type")
                    if task_type not in task_queues:
                        response_queues[req["proc_id"]].put({
                            "error": "unsupported_task_type",
                            "detail": f"unsupported task_type: {task_type!r}",
                        })
                        continue
                    if task_type == "prefill":
                        req.setdefault("prefill_cursor", 0)
                    recent_prefix_lens.append(int(req["prefix_len"]))
                    len_r1, len_r2 = _update_length_thresholds(recent_prefix_lens)
                    cat = _length_category(req["prefix_len"], len_r1, len_r2)
                    task_queues[task_type][cat].append(req)

                # fastsd path: keep all cloud-side scheduling optimizations.
                sort_task_queues()

                has_verify = any(task_queues["verify"].values())
                has_prefill = any(task_queues["prefill"].values())

                if getattr(self.args, "debug_pipeline", False):
                    self.color_print(
                        f"[FASTSD] thresholds prefix_len: short<={len_r1}, mid<={len_r2}",
                        3,
                    )

                if has_verify or has_prefill:
                    try:
                        plan = build_unified_schedule_plan(
                            task_queues["verify"],
                            task_queues["prefill"],
                            scheduler_state,
                            token_budget=self.args.token_budget,
                        )
                    except OversizeVerifyError as exc:
                        reject_oversize_verify(exc)
                        continue
                    if plan.entries:
                        execute_plan(plan)
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.01)
        finally:
            if energy_service is not None:
                energy_service.shutdown()


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
