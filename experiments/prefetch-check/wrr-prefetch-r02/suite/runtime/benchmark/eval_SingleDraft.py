# eval_humaneval.py
import os
import sys
sys.path.append(os.path.join(sys.path[0], "../"))
import time
import torch
import json
import random
import tqdm
import argparse
from multiprocessing import Queue, Process, Value, Lock
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

from src.util import seed_everything, parse_arguments
from src.engine import Decoding

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"

class MultiDraftEval(Decoding):
    def __init__(self, args):
        super().__init__(args)

        # load relative resources
        self.load_tokenizer()
        # self.load_model()

    def load_data(self):
        pass  # not used

    def preprocess(self, input_text):
        return input_text.strip()

    def postprocess(self, input_text, output_text):
        if output_text.startswith(self.tokenizer.bos_token):
            generation = output_text[len(input_text)+len(self.tokenizer.bos_token)+1:]
        else:
            generation = output_text[len(input_text):]

        stop_words = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", self.tokenizer.eos_token]
        for stop_word in stop_words:
            if stop_word in generation:
                generation = generation[:generation.index(stop_word)].strip()

        return input_text + "\n    " + generation.replace("\t", "    ")

    @torch.no_grad()
    def eval(self):
        torch.cuda.init()
        torch.multiprocessing.set_start_method("spawn", force=True)

        # 创建共享队列
        request_queue = Queue()
        # response_queue = Queue()
        response_queues = {i: Queue() for i in range(self.args.num_drafts)}
        measure_started = Value("b", 0)
        completed_tasks = Value("i", 0)
        control_lock = Lock()
        with open(self.args.data_path, "r") as f:
            dataset_tasks = len(f.readlines())
        tasks_per_draft = min(self.args.max_tasks_per_draft, dataset_tasks)
        total_tasks = self.args.num_drafts * tasks_per_draft

        # KV Cache Preload标识符
        # is_preload = Value('b', False)

        # 两组队列
        # task_queues = {
        #     "prefill": {"short": Queue(), "mid": Queue(), "long": Queue()},
        #     "verify": {"short": Queue(), "mid": Queue(), "long": Queue()},
        # }

        # 启动 target_model 中心验证者进程
        target_proc = Process(target=self.run_target_process, args=(self.tokenizer, request_queue, response_queues))
        target_proc.start()
        print("target_proc build success")

        # # 启动 target_model 中心 KV Cache Preload进程
        # Preload_kvcache_proc = Process(target=self.Preload_kvcache, args=is_preload)
        # Preload_kvcache_proc.start()
        # print("Preload_kvcache_proc build success")

        # 启动多个 draft 模型进程
        draft_procs = []
        for i in range(self.args.num_drafts):
            proc = Process(
                target=self.run_draft_process,
                args=(
                    self.tokenizer,
                    request_queue,
                    response_queues,
                    i,
                    measure_started,
                    completed_tasks,
                    control_lock,
                    total_tasks,
                ),
            )
            proc.start()
            draft_procs.append(proc)
            print("draft_proc build success:", i+1)

        for proc in draft_procs:
            proc.join()
            print("draft_proc is closed")

        # proc = Process(
        #     target=self.run_draft_process,
        #     args=(self.tokenizer, request_queue, response_queues, 0)
        # )
        # proc.start()
        # print("draft_proc build success")

        target_proc.terminate()
        print("target_proc is closed")
        # Preload_kvcache_proc.terminate()
        # print("Preload_kvcache_proc is closed")


if __name__ == "__main__":
    args = parse_arguments()
    alg = MultiDraftEval(args)
    alg.eval()
