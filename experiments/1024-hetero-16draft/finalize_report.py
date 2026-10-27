"""Archive the completed run and a Chinese overview without rerunning inference."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / "results.json").read_text())
assert report["status"] == "completed" and not report["issues"]
methods = {"standard": "Standard SD", "fastsd": "FastSD", "specedge": "SpecEdge 单链"}
for method in methods:
    assert report["methods"][method]["overall"]["completed_requests"] == 16
    assert report["methods"][method]["overall"]["rpc_error_count"] == 0

def requests(method, node):
    return {row["task_id"]: row for path in (ROOT / method / node / "run").glob("edge_metrics_proc*.jsonl")
            for row in [json.loads(line) for line in path.read_text().splitlines()]}

reviews = []
for mismatch in report["output_comparisons"]["specedge"]["mismatched_outputs"]:
    node, task = mismatch["node"], mismatch["task_id"]
    a, b = requests("standard", node)[task], requests("specedge", node)[task]
    reviews.append({**mismatch, "standard_text": a["output_text"], "specedge_text": b["output_text"],
                    "reference": a.get("reference"), "root_cause": "not_established",
                    "manual_review": "gsm8k-0003: wording differs; both final answers are 540. This is not token-exact parity."})
(ROOT / "output_review.json").write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n")

manifest = json.loads((ROOT / "manifest.json").read_text())
stages = [json.loads((ROOT / (name + "_status.json")).read_text()) for name in ["native", "specedge"]]
manifest.update(status="completed", completed_epoch=max(x["epoch"] for x in stages),
                finalized_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                completed_requests={method: 16 for method in methods},
                rpc_errors={method: 0 for method in methods},
                output_identical_to_standard={method: comparison["identical_output_requests"]
                    for method, comparison in report["output_comparisons"].items()},
                output_review="output_review.json",
                method_overrides={"specedge": {"method": "specedge_cpu_adapted_single_chain", "transport": "gRPC",
                    "token_budget": None, "max_len": 768, "batch_size": 8, "main_branch_width": 1,
                    "proactive_branch_width": 1, "prefill_cache": False}},
                async_target_kv_prefetch_exercised=False)
manifest["limitations"] += [item for item in [
    "SpecEdge differs from Standard SD in one output sequence (gsm8k-0003); root cause not established",
    "FastSD asynchronous KV prefetch submitted zero tasks in this run",
    "SpecEdge has no equivalent to the native token_budget=512 scheduling limit"] if item not in manifest["limitations"]]
(ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

lines = ["# 16-draft 异构实验结果", "", "三种方法均完成 **16/16** 请求，合计48条正式请求，无RPC错误。", "",
    "node1、node3各8个独立Qwen3-1.7B FP16 CPU draft，每个绑定一个物理核、一个计算线程；node1固定γ=100，node3固定γ=20。",
    "Target为gpunode2的Qwen3-32B FP16，物理GPU3（48GB A6000）+CPU权重卸载，GPU权重预算34GiB。三种方法的权重设备映射相同。",
    "同一批GSM8K前16题，偶数索引分配node1，奇数索引分配node3；non-thinking模板、greedy、EOS、256-token上限。每方法一次，正式请求不含加载和冒烟。", "",
    "## 核心结果", "", "以下均为逐请求均值；完整中位数、P95、E2E见[RESULTS.md](RESULTS.md)及[CSV](results.csv)。", "",
    "| 方法 | 分组 | TTFT均值（秒） | TPOT均值（毫秒/token） | 输出tokens |",
    "| --- | --- | ---: | ---: | ---: |"]
for method, label in methods.items():
    data = report["methods"][method]
    for group, summary in [("整体", data["overall"]), ("node1 / γ100", data["nodes"]["node1"]), ("node3 / γ20", data["nodes"]["node3"])]:
        lines.append(f"| {label} | {group} | {summary['ttft_seconds']['mean']:.2f} | {summary['tpot_ms']['mean']:.2f} | {summary['generated_tokens']} |")
lines += ["", "| 方法 | 整体吞吐（tokens/s） | 统一计时窗口（秒） | 候选接受率 |",
          "| --- | ---: | ---: | ---: |"]
for method, label in methods.items():
    value = report["methods"][method]["overall"]
    lines.append(f"| {label} | {value['aggregate_tokens_per_second']:.3f} | {value['wall_seconds']:.2f} | {value['acceptance_rate_weighted']:.2%} |")
lines += ["", "TTFT为请求开始至首个已验证输出块可见；TPOT=(E2E−TTFT)/(输出tokens−1)，是块输出的摊销间隔，包含剩余proactive计算的收尾。整体吞吐使用协调器单一时钟，从模型就绪屏障至全部worker完成，包含后续启动/校准，不跨主机相减monotonic。", "",
    "## 输出核验", "", "- FastSD与Standard SD：**16/16**逐token完全相同。",
    "- SpecEdge与Standard SD：**15/16**逐token完全相同。node3的`gsm8k-0003`从第25个输出token起不同，输出152/143 tokens；两者都算出540，但措辞不同。根因尚未定位，不能声称全部输出完全对齐。",
    "- 该题完整输出和参考答案保存在[output_review.json](output_review.json)。本实验并未进行全量GSM8K准确率评测。", "",
    "## 解释范围", "", "node3三轮的平均load1分别为：Standard SD 4.26、FastSD 111.55、SpecEdge 33.68。负载包含实验与其他任务，三轮顺序运行且负载变化明显，**本轮耗时差不能单独归因于算法**。",
    "node1负载较接近：FastSD平均TTFT为93.64秒，低于Standard SD的110.77秒；平均TPOT为5349.17毫秒，略高于5284.69毫秒。该组同样只有8个请求、一次运行。",
    "FastSD记录了308个调度计划，但异步KV预取提交/完成/失败均为0，本轮没有触发该机制。Native调度计数涵盖Target整个生命周期（含独立冒烟），正式延迟和RPC统计已排除冒烟。",
    "SpecEdge使用官方固定版本的CPU单链适配，保留主草拟、提前草拟、gRPC传输、Target验证、接受/回滚全流程；主链及proactive分支宽度均为1。实际服务端验证γ100共168次、γ20共258次。两端FP16、相同输入模板、相同输出截断；Target batch=8，但SpecEdge没有与Native token_budget=512对应的参数。详见[SpecEdge适配说明](specedge/README.md)。", "",
    "## 图与数据", "", "- [TTFT/TPOT分布图](latency_comparison.png) / [PDF](latency_comparison.pdf)",
    "- [各节点完整负载轨迹](host_load.png)", "- [完整统计JSON](results.json) / [CSV](results.csv)",
    "- [冻结配置](manifest.json)、[tokenizer核验](tokenizer_check.json)、[进程退出核验](postflight.json)", "",
    "原始记录位于`standard/`、`fastsd/`、`specedge/`的各node目录，保留逐请求、逐RPC、CPU核映射、连续负载、完整输出、服务端日志和启动命令。", ""]
(ROOT / "RESULTS.zh.md").write_text("\n".join(lines))
print(json.dumps({"status": "completed", "report": str(ROOT / "RESULTS.zh.md"), "output_review": str(ROOT / "output_review.json")}, ensure_ascii=False))
