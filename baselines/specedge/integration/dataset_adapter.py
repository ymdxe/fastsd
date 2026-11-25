"""Prompt construction and offline quality evaluation for canonical records."""

import re
import subprocess
from decimal import Decimal, InvalidOperation


NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?")
CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.S | re.I)


class DatasetAdapter:
    stop_markers = ("<|im_end|>", "<|eot_id|>", "</s>")

    def build_prompt(self, record, tokenizer, *, chat, history=()):
        messages = self.messages(record, history)
        if not chat:
            if len(messages) != 1:
                raise ValueError("multi-turn evaluation requires a chat template")
            return messages[0]["content"]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    def messages(self, record, history):
        return [{"role": "user", "content": record["prompt"]}]

    def turn_count(self, record):
        return 1

    def extract_answer(self, text):
        for marker in self.stop_markers:
            text = text.split(marker, 1)[0]
        return text.strip()

    def evaluate(self, record, output, *, executor=None, judge=None):
        raise NotImplementedError


class NumericAdapter(DatasetAdapter):
    def extract_answer(self, text):
        text = super().extract_answer(text)
        answer = text.rsplit("####", 1)[-1]
        boxed = re.findall(r"\\boxed\{([^{}]+)\}", answer)
        matches = NUMBER.findall(boxed[-1] if boxed else answer)
        try:
            return Decimal(matches[-1].replace(",", "")) if matches else None
        except InvalidOperation:
            return None

    def evaluate(self, record, output, **_kwargs):
        answer = self.extract_answer(output)
        expected = self.extract_answer(str(record["reference"]))
        return {"metric": "numeric_exact_match", "score": int(answer == expected) if answer is not None and expected is not None else (0 if expected is not None else None)}


class CodeAdapter(DatasetAdapter):
    def extract_answer(self, text):
        for marker in self.stop_markers:
            text = text.split(marker, 1)[0]
        fenced = CODE_BLOCK.search(text)
        return (fenced[1] if fenced else text).strip("\r\n")

    def evaluate(self, record, output, *, executor=None, judge=None):
        if executor is None:
            return {"metric": "pass@1", "score": None, "reason": "isolated executor required"}
        return {"metric": "pass@1", "score": int(executor(self.program(record, self.extract_answer(output))))}


class HumanEvalAdapter(CodeAdapter):
    def program(self, record, code):
        reference = record["reference"]
        entry = reference["entry_point"]
        if not entry or not reference["test"]:
            raise ValueError("HumanEval reference requires entry_point and test")
        if not re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry)}\s*\(", code, re.M):
            code = f"{record['prompt']}\n{code}"
        return f"{code}\n\n{reference['test']}\n\ncheck({entry})\n"


class MBPPAdapter(CodeAdapter):
    def program(self, record, code):
        reference = record["reference"]
        tests = reference["test_list"]
        if not tests:
            raise ValueError("MBPP reference requires test_list")
        return "\n".join([reference.get("test_setup_code") or "", code, *tests, ""])


class MTBenchAdapter(DatasetAdapter):
    def turn_count(self, record):
        if "fixed_messages" in record["reference"]:
            return 1
        return len(record["reference"]["turns"])

    def messages(self, record, history):
        if "fixed_messages" in record["reference"]:
            return record["reference"]["fixed_messages"]
        turns = record["reference"]["turns"]
        if len(history) >= len(turns):
            raise ValueError("MT-Bench history already contains all turns")
        messages = []
        for index, response in enumerate(history):
            messages.extend([{"role": "user", "content": turns[index]}, {"role": "assistant", "content": response}])
        return [*messages, {"role": "user", "content": turns[len(history)]}]

    def evaluate(self, record, output, *, executor=None, judge=None):
        if not isinstance(output, list) or len(output) != self.turn_count(record):
            return {"metric": "mt_bench_judge", "score": None, "reason": "missing turns"}
        return {"metric": "mt_bench_judge", "score": judge(record, output) if judge else None, "completed_turns": len(output)}


ADAPTERS = {"humaneval": HumanEvalAdapter(), "gsm8k": NumericAdapter(), "mgsm": NumericAdapter(),
            "mt_bench": MTBenchAdapter(), "mbpp": MBPPAdapter()}


def container_executor(image, *, timeout=10):
    def execute(program):
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
                 "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
                 "--memory=256m", "--pids-limit=32", "--cpus=1", "--log-driver=none", "-i", "--entrypoint=timeout", image,
                 str(timeout), "python", "-I", "-B", "-"],
                input=program, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as error:
            raise RuntimeError("pass@1 requires Docker or an injected isolated executor") from error
        if result.returncode in {125, 126, 127}:
            raise RuntimeError(f"isolated code runner failed to start (status={result.returncode})")
        return result.returncode == 0
    return execute
