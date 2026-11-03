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
import requests

from .energy_meter import EnergyControlService, EnergyServiceConfig
from .fastsd_scheduler import (
    FASTSD_DYNAMIC_WINDOW, Task, Allocation, BatchPlan,
    rebuild_queues, plan_batch, update_length_thresholds,
)
from .kv_preload import KVPreloader


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
    def run_target_process_batching(self, tokenizer, request_queue, response_queues):
        """Paper Algorithm 1: queue update, read-only planning, execution, commit."""
        target_model = self._load_target_model_for_service(
            self.args.target_model, self._service_target_device())
        config = target_model.config
        if (getattr(config, "_attn_implementation", "eager") == "flash_attention_2"
                or "sliding_attention" in (getattr(config, "layer_types", None) or [])
                or getattr(config, "use_sliding_window", False)):
            raise ValueError("Packed FastSD requires full attention with an eager/SDPA 4D mask")
        rope = getattr(config, "rope_scaling", None) or {}
        if rope.get("rope_type", rope.get("type", "default")) not in {"default", "linear"}:
            raise ValueError("Packed FastSD requires default or linear position scaling")
        kv = KVCacheModel_batching(target_model, self.args.temp, self.args.top_k, self.args.top_p)
        kv.vocab_size = tokenizer.vocab_size
        preloader = KVPreloader(kv._past_key_values, target_model.device)
        energy_service = self._maybe_start_energy_service()
        pending = {}
        accept_stats = defaultdict(lambda: [0, 0])
        history = {kind: deque(maxlen=FASTSD_DYNAMIC_WINDOW) for kind in ("prefill", "verify")}
        defaults = {"prefill": (128, 512), "verify": (20, 100)}
        thresholds = dict(defaults)
        cursor = cycle = 0
        stopping = False
        fastsd = self.args.server_sched_mode == "fastsd"
        budget, max_devices = int(self.args.token_budget), int(self.args.batch_size)
        feedback_ema = {}

        def admit_request(req):
            pid = req["proc_id"]
            if req.get("task_type") == "release":
                if pid in pending:
                    raise ValueError("cannot release a device with pending work")
                kv.reset(pid)
                accept_stats.pop(pid, None)
                feedback_ema.pop(pid, None)
                response_queues[pid].put({"status": "released"})
                return
            # The IPC queue is an external input boundary. Canonicalize once.
            req = dict(req)
            tokens = torch.as_tensor(req["draft_output"], dtype=torch.long).reshape(1, -1)
            req["draft_output"] = tokens
            req.setdefault("arrival_time", time.monotonic())
            req.setdefault("current_time", time.time())
            if pid in pending:
                raise ValueError("only one unfinished task per device is allowed")
            kind = req["task_type"]
            if kind not in {"prefill", "verify"} or tokens.numel() == 0:
                raise ValueError("expected a nonempty prefill/verify request")
            if tokens.min().item() < 0 or tokens.max().item() >= config.vocab_size:
                raise ValueError("token ID is outside the target vocabulary")
            if kind == "prefill":
                if int(req["prefix_len"]) != tokens.numel():
                    raise ValueError("prefill must contain exactly the prompt")
                kv.reset(pid)
                accept_stats[pid] = [0, 0]
                length, bridge = tokens.numel(), 0
            else:
                length = int(req.get("gamma", 0) or self.args.gamma)
                req["gamma"] = length
                if pid not in kv._past_key_values:
                    raise ValueError("verify requires a completed prefill")
                bridge = int(req["prefix_len"]) - kv._past_key_values[pid].get_seq_length()
                expected = length + (bridge if req.get("tail_only", False) else int(req["prefix_len"]))
                if length <= 0 or bridge < 0 or tokens.numel() != expected:
                    raise ValueError("verify payload does not match its prefix, cache and gamma")
                if bridge >= budget and fastsd:
                    raise ValueError("token_budget must fit the bridge and one draft token")
            task = Task(req, length, bridge=bridge, last_served_cycle=cycle)
            pending[pid] = task
            history[kind].append(length)
            thresholds[kind] = update_length_thresholds(history[kind], *defaults[kind])

        def receive(req):
            # Report malformed external requests to their caller. Inference
            # errors remain fail-fast in execute(), outside this boundary.
            try:
                admit_request(req)
            except ValueError as exc:
                response_queues[req["proc_id"]].put({"error": str(exc)})

        def execute(allocations, kind):
            tokens = []
            for allocation in allocations:
                task, req = allocation.task, allocation.task.request
                if task.pid in kv._past_key_values:
                    preloader.ensure(task.pid)
                if kind == "prefill":
                    start = allocation.offset
                elif req.get("tail_only", False):
                    start = task.bridge + allocation.offset - allocation.bridge
                else:
                    start = int(req["prefix_len"]) + allocation.offset - allocation.bridge
                tokens.append(req["draft_output"][:, start:start + allocation.cost])
            started = time.perf_counter()
            kv.forward(tokens, [a.task.pid for a in allocations])
            if target_model.device.type == "cuda":
                torch.cuda.current_stream(target_model.device).synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            for allocation in allocations:
                task, req = allocation.task, allocation.task.request
                pid = task.pid
                if kind == "prefill":
                    task.offset += allocation.tokens
                    if task.remaining == 0:
                        response_queues[pid].put({"status": "prefill_ok"})
                        del pending[pid]
                    continue
                task.verify_ms += elapsed_ms
                absolute = int(req["prefix_len"]) + allocation.offset
                draft_start = (task.bridge + allocation.offset if req.get("tail_only", False)
                               else absolute)
                draft = req["draft_output"][0, draft_start:draft_start + allocation.tokens]
                target = kv._prob_history[pid][0, absolute - 1:absolute + allocation.tokens - 1].argmax(-1)
                mismatches = (draft.to(target.device) != target).nonzero()
                accepted = int(mismatches[0].item()) if mismatches.numel() else allocation.tokens
                task.offset += accepted
                finished = accepted < allocation.tokens or task.remaining == 0
                if finished:
                    accepted_length = int(req["prefix_len"]) + task.offset
                    final_token = kv._prob_history[pid][0, accepted_length - 1].argmax().item()
                    kv.rollback(pid, accepted_length)
                    accept_stats[pid][0] += task.offset
                    accept_stats[pid][1] += task.length
                    result = {"accepted": accepted_length, "final_token": int(final_token),
                              "verify_ms": task.verify_ms}
                    # Keep the original feedback rule for the existing pipeline baseline.
                    # Paper FastSD uses the device's declared draft length m_n.
                    if (not fastsd and self.args.enable_pipeline and self.args.pipeline_gamma_adapt):
                        values = (max(1e-6, req.get("avg_cloud_total_ms", 0) / 1000 or task.verify_ms / 1000),
                                  max(1e-6, req["lag"] / task.length), req.get("transport_rtt", 0))
                        previous = feedback_ema.get(pid, values)
                        alpha = self.args.pipeline_ema_alpha
                        v, d, rtt = feedback_ema[pid] = tuple((1 - alpha) * old + alpha * new
                                                            for old, new in zip(previous, values))
                        result["suggested_gamma"] = max(self.args.pipeline_gamma_min,
                            min(self.args.pipeline_gamma_max, round(max(1e-6, v - rtt) / d)))
                    response_queues[pid].put(result)
                    del pending[pid]

        try:
            while pending or not stopping:
                if not pending and not stopping:
                    req = request_queue.get()
                    if req is None:
                        stopping = True
                    else:
                        receive(req)
                # Bounded ingress prevents a continuous producer from starving execution.
                if not stopping:
                    for _ in range(1024):
                        try:
                            req = request_queue.get_nowait()
                        except queue.Empty:
                            break
                        if req is None:
                            stopping = True
                            break
                        receive(req)
                if not pending:
                    continue
                queues = rebuild_queues(pending.values(), thresholds, accept_stats, time.monotonic())
                if fastsd:
                    plan = plan_batch(queues, cursor, cycle, budget, max_devices)
                else:
                    task = next(iter(pending.values()))
                    plan = BatchPlan((Allocation(task, task.offset, task.remaining, task.bridge),), cursor, cycle)
                active = {a.task.pid for a in plan.allocations}
                if fastsd:
                    # Paper Eq. (8): same Q_t and h_t, excluding current devices.
                    prediction = plan_batch(queues, cursor, cycle, budget, max_devices, excluded=active)
                    predicted = [a.task.pid for a in prediction.allocations if a.task.kind == "verify"]
                    preloader.predict(predicted, active=active)
                if energy_service is not None:
                    energy_service.enter_active()
                try:
                    for kind in ("verify", "prefill"):
                        allocations = [a for a in plan.allocations if a.task.kind == kind]
                        if allocations:
                            execute(allocations, kind)
                finally:
                    if energy_service is not None:
                        energy_service.exit_active()
                # Commit the actual plan only after both phases succeed.
                for allocation in plan.allocations:
                    allocation.task.last_served_cycle = allocation.cycle
                for task in pending.values():
                    if task.kind == "prefill":
                        # Count only complete cycles without service, excluding
                        # the cycle of arrival or most recent execution.
                        task.waited_cycles = max(0, plan.cycle - task.last_served_cycle - 1)
                cursor, cycle = plan.cursor, plan.cycle
                for allocation in plan.allocations:
                    if should_offload_target_cache(allocation.task.kind, pid=allocation.task.pid):
                        preloader.offload(allocation.task.pid)
                preloader.collect()
        finally:
            preloader.close()
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
