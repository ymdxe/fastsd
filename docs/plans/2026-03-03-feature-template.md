# Feature Template Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 明确并实现一个具体功能（待你补充）。

**Architecture:**
基于现有 FastSD 代码结构，在 `src/` 中做最小改动实现功能；
通过测试或最小可复现实验脚本验证行为；
避免引入与推理主链路无关依赖，保持 DRY / YAGNI。

**Tech Stack:** Python 3, PyTorch 2.1.2, Transformers, FastSD (`src/engine.py`, `src/util.py`, KVCache 模块)

---

### Task 1: 明确需求与验收标准

**Files:**
- Modify: `docs/plans/2026-03-03-feature-template.md`
- Test: `（按需求补充，例如 tests/... 或 benchmark/...）`

**Step 1: 写失败用例（先定义预期行为）**

```python
# TODO: 按需求补充真实测试

def test_target_behavior():
    assert False, "replace with real failing assertion"
```

**Step 2: 运行测试并确认失败**

Run: `pytest <test-path>::<test-name> -v`
Expected: FAIL（与目标行为相关，而不是语法错误）

**Step 3: 最小实现**

```python
# TODO: 在 src/ 对应模块补充最小实现
```

**Step 4: 运行测试并确认通过**

Run: `pytest <test-path>::<test-name> -v`
Expected: PASS

**Step 5: Commit**

```bash
git add <modified-files>
git commit -m "feat: implement <feature-name>"
```

### Task 2: 端到端验证（与 FastSD 运行方式一致）

**Files:**
- Modify: `scripts/<script>.sh`（如需）
- Test: `benchmark/<benchmark-script>.py`（如需）

**Step 1: 增加最小可复现实验参数**

```bash
# TODO: 补充真实命令
bash scripts/run_sd.sh
```

**Step 2: 运行并记录关键指标**

Run: `bash scripts/<script>.sh`
Expected: 输出目标指标（latency / throughput / accept rate 等）

**Step 3: 校验无回归**

Run: `pytest -q`（若无完整测试集，至少跑受影响模块测试）
Expected: PASS

**Step 4: Commit**

```bash
git add <modified-files>
git commit -m "test: add validation for <feature-name>"
```

### Task 3: 文档与可复现说明

**Files:**
- Modify: `README.md` 或 `docs/<feature>.md`

**Step 1: 记录参数与默认值**

```markdown
- 参数名
- 默认值
- 作用
- 对性能/精度影响
```

**Step 2: 提供复现命令**

```bash
# TODO: 补充真实复现命令
```

**Step 3: Commit**

```bash
git add README.md docs/
git commit -m "docs: document <feature-name> usage and reproduction"
```
