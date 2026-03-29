# EthAuditor 架构设计文档（Step 1）

> 版本：v0.1（2026-03-28）
> 范围：离线预处理 + Phase 1 + Phase 2 的系统蓝图
> 目标：作为 Step 2～5 的唯一实现基线

---

## 1. 总体目标与边界

EthAuditor 通过 **离线静态分析预处理** + **两阶段 LangGraph 多智能体迭代**，自动完成：

1. 跨客户端 Guard/Action 词汇发现与收敛（Phase 1）
2. 7 个核心 Workflow 的 LSG 抽取、横向比对与逻辑差异报告（Phase 2）

### 1.1 系统边界

- 输入（只读）：
  - `./docs/LSG_Schema_Spec.md`
  - `./code/{prysm,lighthouse,grandine,teku,lodestar}/`
- 输出（可写）：
  - `./output/preprocess/*`
  - `./output/checkpoints/*`
  - `./output/iterations/*`
  - `./output/audit_logs/*`
  - `./output/Global_LSG_Spec_Enriched.yaml`
  - `./output/LSG_<ClientName>_final.yaml`
  - `./output/Audit_Diff_Report.md`

---

## 2. GlobalState 设计

采用 **Pydantic BaseModel**（推荐）定义，兼顾强校验与可序列化。

## 2.1 类型定义草案（实现参考）

```python
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Literal

ClientName = Literal["prysm", "lighthouse", "grandine", "teku", "lodestar"]
Phase = Literal[0, 1, 2]
WorkflowId = Literal[
    "initial_sync",
    "regular_sync",
    "checkpoint_sync",
    "attestation_generate",
    "block_generate",
    "aggregate",
    "execute_layer_relation",
]

class Evidence(BaseModel):
    file: str
    function: str
    lines: tuple[int, int]

class VocabEntry(BaseModel):
    name: str
    category: str
    description: str
    aliases: list[str] = []
    evidence: list[Evidence] = []

class Transition(BaseModel):
    guard: str
    actions: list[str]
    next_state: str
    evidence: Evidence | None = None

class LSGStateNode(BaseModel):
    id: str
    label: str
    category: str
    transitions: list[Transition]

class LSGWorkflow(BaseModel):
    id: WorkflowId
    name: str
    description: str
    mode: str
    initial_state: str
    states: list[LSGStateNode]

class LSGFile(BaseModel):
    version: int = 1
    client: str
    generated_at: str
    guards: list[VocabEntry]
    actions: list[VocabEntry]
    workflows: list[LSGWorkflow]

class PreprocessArtifactStatus(BaseModel):
    symbols_json: bool = False
    callgraph_json: bool = False
    bm25_pkl: bool = False
    chroma_dir: bool = False

class PreprocessStatus(BaseModel):
    done: bool = False
    force_rebuild: bool = False
    per_client: dict[ClientName, PreprocessArtifactStatus] = {}
    started_at: str | None = None
    finished_at: str | None = None
    skipped_clients: list[ClientName] = []

class DiffItem(BaseModel):
    diff_id: str
    diff_class: Literal["A", "B"]
    workflow_id: WorkflowId
    state_id: str
    transition_guard: str
    involved_clients: list[ClientName]
    summary: str
    expected_behavior: str | None = None
    actual_behavior: str | None = None
    evidence: dict[ClientName, list[Evidence]] = {}

class IterationMetrics(BaseModel):
    phase: Literal[1, 2]
    iteration: int
    diff_rate: float | None = None
    logic_diff_rate: float | None = None
    compared_items: int = 0
    new_vocab_items: int = 0
    a_diff_count: int = 0
    b_diff_count: int = 0

class GlobalState(BaseModel):
    # 运行控制
    phase: Phase = 0
    iteration_phase1: int = 0
    iteration_phase2: int = 0
    converged_phase1: bool = False
    converged_phase2: bool = False
    force_stopped: bool = False

    # 预处理状态
    preprocess: PreprocessStatus = Field(default_factory=PreprocessStatus)

    # 词汇表（全局）
    vocab_version: int = 0
    guards_vocab: list[VocabEntry] = []
    actions_vocab: list[VocabEntry] = []

    # 各客户端中间与最终 LSG
    lsg_current_iter: dict[ClientName, LSGFile] = {}
    lsg_final: dict[ClientName, LSGFile] = {}

    # 差异结果
    diff_rate: float = 1.0
    logic_diff_rate: float = 1.0
    diff_items_a: list[DiffItem] = []
    diff_items_b: list[DiffItem] = []

    # 审计与快照
    checkpoint_paths: list[str] = []
    audit_log_paths: list[str] = []

    # 节点间通信缓存
    phase1_sub_reports: dict[ClientName, dict] = {}
    phase2_sub_reports: dict[ClientName, LSGFile] = {}

    # 可观察性
    metrics_history: list[IterationMetrics] = []
    warnings: list[str] = []
    errors: list[str] = []
```

## 2.2 字段读写责任矩阵（节点级）

| 字段 | 写入节点 | 读取节点 | 说明 |
|---|---|---|---|
| `phase` | `init_node`, `phase1_finalize_node`, `phase2_finalize_node` | 全部 router / agent 节点 | 当前阶段标识 0/1/2 |
| `iteration_phase1` | `phase1_main_node`, `router_phase1` | `phase1_dispatch_node`, `phase1_subagent_node` | 每轮+1 |
| `iteration_phase2` | `phase2_main_node`, `router_phase2` | `phase2_dispatch_node`, `phase2_subagent_node` | 每轮+1 |
| `converged_phase1` | `router_phase1` | `phase1_finalize_node` | `diff_rate < 0.05` |
| `converged_phase2` | `router_phase2` | `phase2_finalize_node` | `logic_diff_rate < 0.05` |
| `force_stopped` | `router_phase1/router_phase2` | `phase1_finalize_node/phase2_finalize_node` | 超过 `MAX_ITER` |
| `preprocess` | `preprocess_node` | `phase1_dispatch_node`, `search tools` | 离线预处理产物状态 |
| `vocab_version` | `phase1_main_node` | `phase1_subagent_node`, `phase2_dispatch_node` | 每次词汇合并后 +1 |
| `guards_vocab/actions_vocab` | `phase1_main_node`, `phase1_finalize_node` | `phase1_subagent_node`, `phase2_subagent_node` | 统一词汇表 |
| `lsg_current_iter` | `phase2_subagent_collect_node` | `phase2_main_node` | 当轮 LSG 输入主比较器 |
| `lsg_final` | `phase2_finalize_node` | `writer` | 最终落盘 |
| `diff_rate` | `phase1_main_node` | `router_phase1` | 词汇收敛指标 |
| `logic_diff_rate` | `phase2_main_node` | `router_phase2` | 逻辑差异收敛指标 |
| `diff_items_a` | `phase2_main_node` | `phase2_feedback_node` | A 类差异用于反馈重抽象 |
| `diff_items_b` | `phase2_main_node` | `phase2_finalize_node`, `writer` | B 类差异写最终报告 |
| `phase1_sub_reports` | `phase1_subagent_node` | `phase1_main_node` | 每客户端候选词汇 |
| `phase2_sub_reports` | `phase2_subagent_node` | `phase2_main_node` | 每客户端当轮 LSG |
| `checkpoint_paths` | `checkpoint_node` | `resume_node` | 断点续跑索引 |
| `audit_log_paths` | `audit_callback` | `finalize nodes` | 审计追踪 |
| `metrics_history` | `phase1_main_node`, `phase2_main_node` | 监控/报告节点 | 迭代轨迹 |
| `warnings/errors` | 任意节点 | `finalize nodes` | 故障与降级说明 |

---

## 3. LangGraph 节点拓扑（含条件路由与并行）

```mermaid
flowchart TD
    A([START]) --> B[init_node]
    B --> C[preprocess_node\n(offline AST/callgraph/index)]
    C --> D{preprocess done?}
    D -- no --> Z1[fail_fast_node]
    D -- yes --> E[phase1_dispatch_node\nprepare vocab + iter]

    E --> F[phase1_fanout_node\nSend(client) x5]
    F --> F1[phase1_subagent_prysm]
    F --> F2[phase1_subagent_lighthouse]
    F --> F3[phase1_subagent_grandine]
    F --> F4[phase1_subagent_teku]
    F --> F5[phase1_subagent_lodestar]

    F1 --> G[phase1_collect_node]
    F2 --> G
    F3 --> G
    F4 --> G
    F5 --> G

    G --> H[phase1_main_node\nmerge+dedupe+diff_rate]
    H --> I[checkpoint_node]
    I --> J{router_phase1}

    J -- diff_rate < 0.05 --> K[phase1_finalize_node\nwrite Global_LSG_Spec_Enriched.yaml]
    J -- iter >= MAX_ITER_PHASE1 --> K
    J -- else continue --> E

    K --> L[phase2_dispatch_node\nseed 7 workflows]
    L --> M[phase2_fanout_node\nSend(client) x5]
    M --> M1[phase2_subagent_prysm]
    M --> M2[phase2_subagent_lighthouse]
    M --> M3[phase2_subagent_grandine]
    M --> M4[phase2_subagent_teku]
    M --> M5[phase2_subagent_lodestar]

    M1 --> N[phase2_collect_node]
    M2 --> N
    M3 --> N
    M4 --> N
    M5 --> N

    N --> O[phase2_main_node\ntri-key compare + classify A/B]
    O --> P[phase2_feedback_node\nfeedback A diffs]
    P --> Q[checkpoint_node]
    Q --> R{router_phase2}

    R -- logic_diff_rate < 0.05 --> S[phase2_finalize_node\nwrite finals + report]
    R -- iter >= MAX_ITER_PHASE2 --> S
    R -- else continue --> L

    S --> T([END])

    Z1 --> T
```

### 3.1 条件边（必须由 `add_conditional_edges` 实现）

- `router_phase1(state)`：
  - `if state.diff_rate < CONVERGENCE_THRESHOLD`: `"phase1_finalize"`
  - `elif state.iteration_phase1 >= MAX_ITER_PHASE1`: `"phase1_finalize_force_stop"`
  - `else`: `"phase1_continue"`
- `router_phase2(state)`：
  - `if state.logic_diff_rate < CONVERGENCE_THRESHOLD`: `"phase2_finalize"`
  - `elif state.iteration_phase2 >= MAX_ITER_PHASE2`: `"phase2_finalize_force_stop"`
  - `else`: `"phase2_continue"`

### 3.2 并行 fan-out 约定（Send API）

- `phase1_fanout_node`：对 `CLIENT_NAMES` 逐个 `Send("phase1_subagent_node", payload)`
- `phase2_fanout_node`：对 `CLIENT_NAMES` 逐个 `Send("phase2_subagent_node", payload)`
- 汇聚节点负责以 `client_name` 作为 key 合并结果，必须保证幂等（重复上报可覆盖）

---

## 4. 文件产出清单与命名规范

## 4.1 离线预处理产物（每客户端）

目录：`./output/preprocess/`

- `<client>_symbols.json`
- `<client>_callgraph.json`
- `<client>_bm25.pkl`
- `<client>_chroma/`（持久化向量库目录）

跳过重建条件：上述四类产物均存在且 `force_rebuild=False`。

## 4.2 迭代期产物

目录：`./output/iterations/`

- `LSG_<ClientName>_iter<N>.yaml`（Phase 2 每轮每客户端）

目录：`./output/checkpoints/`

- `checkpoint_phase<P>_iter<N>.json`

目录：`./output/audit_logs/`

- `audit_phase<P>_iter<N>_<agent_type>_<timestamp>.json`

## 4.3 最终产物

目录：`./output/`

- `Global_LSG_Spec_Enriched.yaml`
- `LSG_<ClientName>_final.yaml`（×5）
- `Audit_Diff_Report.md`

---

## 5. Step 2～5 任务拆解（函数签名粒度）

## 5.1 Step 2 — Graph Infrastructure

### `config.py`

```python
MAX_ITER_PHASE1: int = 20
MAX_ITER_PHASE2: int = 20
CONVERGENCE_THRESHOLD: float = 0.05
CLIENT_NAMES: list[str] = ["prysm", "lighthouse", "grandine", "teku", "lodestar"]
CODE_BASE_PATH: str = "./code"
OUTPUT_PATH: str = "./output"
PREPROCESS_PATH: str = "./output/preprocess"
SPEC_PATH: str = "./docs/LSG_Schema_Spec.md"
BM25_VECTOR_WEIGHT: tuple[float, float] = (0.4, 0.6)
ENTRY_POINT_OVERRIDES: dict[str, dict[str, list[str]]] = {}
```

### `state.py`

- 暴露：`GlobalState`, `VocabEntry`, `LSGWorkflow`, `PreprocessStatus`, `DiffItem`, `IterationMetrics`
- 接口：
  - `def make_initial_state() -> GlobalState`

### `graph.py`

- 构建入口：
  - `def build_graph() -> "CompiledStateGraph"`
- 路由：
  - `def router_phase1(state: GlobalState) -> str`
  - `def router_phase2(state: GlobalState) -> str`
- 节点（Mock）：
  - `def preprocess_node(state: GlobalState) -> GlobalState`
  - `def phase1_dispatch_node(state: GlobalState) -> GlobalState`
  - `def phase1_subagent_node(state: GlobalState, client_name: str) -> dict`
  - `def phase1_main_node(state: GlobalState) -> GlobalState`
  - `def phase2_dispatch_node(state: GlobalState) -> GlobalState`
  - `def phase2_subagent_node(state: GlobalState, client_name: str) -> dict`
  - `def phase2_main_node(state: GlobalState) -> GlobalState`
  - `def checkpoint_node(state: GlobalState) -> GlobalState`

关键依赖：`langgraph`, `pydantic`, `typing_extensions`

## 5.2 Step 3 — Tooling & RAG

### `tools/preprocessor.py`

- `def run_preprocessing(client_name: str, force_rebuild: bool = False) -> PreprocessArtifactStatus`
- `def _extract_symbols(client_name: str) -> list[SymbolInfo]`
- `def _build_callgraph(client_name: str, symbols: list[SymbolInfo]) -> CallGraph`
- `def _build_vector_index(client_name: str, symbols: list[SymbolInfo], callgraph: CallGraph) -> None`
- `def _build_bm25_index(client_name: str, symbols: list[SymbolInfo]) -> None`

### `tools/search.py`

- `@tool def search_codebase(query: str, client_name: str, top_k: int = 5) -> list[Document]`
- `@tool def search_codebase_by_workflow(workflow_id: str, query: str, client_name: str, max_call_depth: int = 5, top_k: int = 10) -> list[Document]`

关键依赖：`tree-sitter`, `tree-sitter-go/rust/java/typescript`, `langchain`, `langchain-community`, `langchain-chroma`, `rank-bm25`, `networkx`

## 5.3 Step 4 — Agents + Prompt

### `agents/prompts/*.jinja2`

- `phase1_sub.jinja2`
- `phase1_main.jinja2`
- `phase2_sub.jinja2`
- `phase2_main.jinja2`

### Agent 模块

- `def run_phase1_sub_agent(client_name: str, vocab: dict, iteration: int, llm) -> VocabDiscoveryReport`
- `def run_phase1_main_agent(sub_reports: dict, current_vocab: dict, llm) -> EnrichedSpec`
- `def run_phase2_sub_agent(client_name: str, enriched_spec: dict, iteration: int, llm) -> LSGFile`
- `def run_phase2_main_agent(client_lsgs: dict[str, LSGFile], llm) -> DiffReport`

关键依赖：`langchain`, `anthropic`（Claude 模型）, `jinja2`, `pydantic`

## 5.4 Step 5 — I/O & 收尾

### `io/checkpoint.py`

- `def save_checkpoint(state: GlobalState, phase: int, iteration: int) -> "Path"`
- `def load_checkpoint(phase: int, iteration: int) -> GlobalState`

### `io/writer.py`

- `def write_enriched_spec(spec: dict) -> "Path"`
- `def write_iteration_lsg(client_name: str, iteration: int, lsg: LSGFile) -> "Path"`
- `def write_final_lsgs(lsg_final: dict[str, LSGFile]) -> list["Path"]`
- `def write_diff_report(diff_items_b: list[DiffItem], summary: dict) -> "Path"`

### `io/audit_logger.py`

- `class AuditCallbackHandler(BaseCallbackHandler): ...`
- `def make_audit_callback(state_getter: callable) -> AuditCallbackHandler`

关键依赖：`pyyaml`, `orjson`(可选), `langchain-core`

---

## 6. 关键设计决策（便于 Review）

1. **GlobalState 使用 Pydantic 而非 TypedDict**：便于 checkpoint 反序列化与字段校验。
2. **离线预处理作为图起点强制门禁**：无预处理则 Phase 1/2 不允许启动。
3. **三元组主键比较**：`(workflow_id, state_id, transition_guard)` 固化为横向比对主键，确保差异归档可重复。
4. **A/B 差异分流**：A 进入反馈闭环，B 进入最终报告，保证收敛目标聚焦逻辑差异。
5. **Checkpoint 每轮必写**：支持硬中断恢复与可审计回放。

---

## 7. Step 1 验收自检

- [x] `GlobalState` 覆盖阶段号、迭代计数、词汇表版本、各客户端 LSG、差异率、收敛标志、`force_stopped`、预处理状态、审计路径
- [x] 提供字段级读写节点说明
- [x] Mermaid 图包含离线预处理、Phase 1、Phase 2、并行 fan-out 与全部条件边
- [x] 文件产出规范覆盖预处理/迭代/最终产物
- [x] Step 2～5 细化到文件与函数签名
