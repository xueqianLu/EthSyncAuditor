# 角色设定 (Role)

你是一位资深 AI 工程师和分布式系统架构师，精通大模型多智能体协作、LangGraph 框架高级特性，以及 RAG（检索增强生成）系统设计。

我们正在共同开发 **EthAuditor**——一个自动化提取并横向比对以太坊共识层客户端**逻辑同步图（LSG, Logic Synchronization Graph）**的审计系统。目标客户端覆盖以下实现：

| 客户端 | 语言 |
|---|---|
| Prysm | Go |
| Lighthouse | Rust |
| Grandine | Rust |
| Teku | Java |
| Lodestar | JavaScript / TypeScript |

---

# 背景知识 (Background)

## 1. LSG 核心概念

LSG（逻辑同步图）是对以太坊共识客户端内部**业务工作流的状态机建模**，由三类要素构成：

### Guard（守卫条件）
Guard 是触发状态迁移的**布尔前置条件**（即状态机中的 Σ）。只有当 Guard 评估为 `true` 时，对应的状态迁移才被允许执行。每个 Guard 有全局唯一的名称和语义分类。

规范中的预定义 Guard 示例（`./docs/LSG_Schema_Spec.md` Section 2 完整定义）：

| Guard 名称 | 分类 | 语义 |
|---|---|---|
| `RespRecv` | network | 从对等节点收到响应 |
| `TimeoutExpired` | time | 请求超时触发 |
| `MissingParent` | validation | 区块引用的父块尚未已知 |
| `HasProposerDuty` | validator | 本地验证者在本 Slot 有出块职责 |
| `ExecutionValidationFailed` | exec | 执行层验证负载失败 |
| `ModeIsInitialSync` | mode | 节点处于初始同步模式 |
| ... | ... | （共 23 个预定义 Guard，详见规范）|

### Action（动作）
Action 是状态迁移发生时**伴随执行的副作用**（即状态机中的 Γ），包括发送消息、更新本地状态、广播数据等。Action 名称全局唯一。

规范中的预定义 Action 示例（`./docs/LSG_Schema_Spec.md` Section 3 完整定义）：

| Action 名称 | 分类 | 语义 |
|---|---|---|
| `SendRangeRequest` | network | 向对等节点发送区块范围请求 |
| `ValidateBatch` | block | 验证一批区块 |
| `ApplyBatch` | block | 应用一批区块并推进分叉选择 |
| `BuildBlock` | validator | 构建区块（含执行负载） |
| `SignAttestation` | validator | 对见证投票签名 |
| `RollbackToSafeHead` | exec | 回滚到最后安全检查点 |
| ... | ... | （共 33 个预定义 Action，详见规范）|

### Workflow（工作流）
Workflow 是由若干**状态节点（States）**、**Guard 条件**和 **Action 动作**组成的有向状态机，描述共识客户端在某一核心业务场景下的完整处理流程，最终表示为一张 LSG 图。

规范中保留了以下 **7 个核心 Workflow ID**，所有客户端 Agent **必须**实现：

| Workflow ID | 说明 |
|---|---|
| `initial_sync` | 初始同步：从空状态启动到追上链头 |
| `regular_sync` | 常规同步：通过 Gossip 持续跟随链头 |
| `checkpoint_sync` | 检查点同步：从最终化检查点快速启动 |
| `attestation_generate` | 见证投票生成：验证者执行 Attestation 职责 |
| `block_generate` | 区块生成：验证者执行 Block Proposal 职责 |
| `aggregate` | 聚合：聚合见证/同步委员会消息 |
| `execute_layer_relation` | 执行层交互：与执行客户端（EL）的协作流程 |

### LSG YAML 文件格式（`LSG_Schema_Spec.md` 定义）

每个客户端每次迭代输出一个 YAML 文件，顶层结构如下：

```yaml
version: 1
client: <string>           # e.g. "prysm"
generated_at: <RFC3339>    # 比对时忽略此字段

guards:                    # 全局 Guard 词汇表（本客户端用到的所有 Guard）
  - name: <GuardName>
    category: <string>
    description: <string>

actions:                   # 全局 Action 词汇表（本客户端用到的所有 Action）
  - name: <ActionName>
    category: <string>
    description: <string>

workflows:
  - id: <one-of-7-reserved-ids>
    name: <string>
    description: <string>
    mode: <string>
    initial_state: <StateId>
    states:
      - id: <StateId>      # 推荐格式：<workflow_prefix>.<phase>，如 initial.peer_select
        label: <string>
        category: <string> # init|peer_select|request|wait|validate|import|recover|progress|terminal 等
        transitions:
          - guard: <GuardName | TRUE>
            actions: [<ActionName>, ...]
            next_state: <StateId>
            evidence:          # 【Phase 2 必填】
              file: <string>   # 相对仓库根目录的文件路径
              function: <string>   # 函数或方法名
              lines: [<int>, <int>]  # [起始行, 结束行]，1-based
```

> **关键规则**（来自规范 Section 6）：
> - 比对工具将 `guards[*].name`、`actions[*].name`、`workflows[*].id`、`states`、`transitions` 视为**结构性比对字段**
> - `generated_at` 和 `evidence` 为**信息性字段**，结构比对时忽略
> - Guard 取值必须是已声明的 `GuardName` 或特殊字面量 `TRUE`（无条件迁移）
> - `actions` 列表可为空（`[]`）

---

## 2. 本地资源

| 资源 | 路径 |
|---|---|
| LSG 规范文件 | `./docs/LSG_Schema_Spec.md` |
| Prysm 源码 | `./code/prysm/` |
| Lighthouse 源码 | `./code/lighthouse/` |
| Grandine 源码 | `./code/grandine/` |
| Teku 源码 | `./code/teku/` |
| Lodestar 源码 | `./code/lodestar/` |

---

# 系统架构要求 (System Requirements)

## 整体流程

系统分为 **2 个串行阶段（Phase）**，每个阶段内部通过多轮迭代驱动收敛，在正式进入任何阶段之前，需先完成一次**离线静态分析预处理**：

```
[离线预处理] tree-sitter AST 解析 → 函数调用图 → 增强向量索引
      ↓
Phase 1: 规范提炼（Spec Enrichment）
    输入: LSG_Schema_Spec.md（基础词汇表）
    输出: Global_LSG_Spec_Enriched.yaml（富化后的统一词汇表）
      ↓
Phase 2: LSG 提取与比对（LSG Extraction & Diff）
    输入: Global_LSG_Spec_Enriched.yaml
    输出: LSG_<ClientName>_final.yaml × 5 个客户端
          Audit_Diff_Report.md（标注 B 类逻辑差异，供人工审查）
```

---

## 离线预处理阶段（Offline Preprocessing）

> **执行时机**：在 Phase 1 启动前一次性完成，结果持久化到磁盘，后续所有 RAG 检索均基于此预处理结果。不随迭代重复执行。

共识客户端（如 Prysm/Lighthouse/Teku/LoadStar/Grandine）的代码量在 **50 万～100 万行**级别，纯语义向量检索面临三大挑战：
1. **同名函数干扰**：`validate`、`process` 等词在源码中出现数千次，语义相近但语境完全不同
2. **跨文件逻辑分散**：一个 Workflow 的实现往往跨越 10～20 个文件，单次 RAG 召回无法覆盖完整链路
3. **状态机隐式表达**：Go/Rust 的状态机极少有显式 `state` 枚举，更多通过 channel、goroutine、event loop 隐式表达

为此，在 RAG 检索之前，必须通过静态分析构建**结构化代码导航层**，将"模糊语义匹配"升级为"有结构的代码路径追踪"。

### 预处理任务清单

**任务 1：基于 tree-sitter 的 AST 解析**

使用 `tree-sitter` Python 绑定（`pip install tree-sitter`）对各客户端源码进行全量 AST 解析：

```python
# 各语言对应的 tree-sitter grammar
LANGUAGE_GRAMMARS = {
    "prysm":      ("go",         "tree-sitter-go"),
    "lighthouse": ("rust",       "tree-sitter-rust"),
    "grandine":   ("rust",       "tree-sitter-rust"),
    "teku":       ("java",       "tree-sitter-java"),
    "lodestar":   ("typescript", "tree-sitter-typescript"),
}
```

从 AST 中提取每个**函数/方法**的以下信息，序列化为 `./output/preprocess/<client>_symbols.json`：

```json
{
  "file": "beacon-chain/sync/service.go",
  "function_name": "runInitialSync",
  "qualified_name": "(*Service).runInitialSync",
  "start_line": 142,
  "end_line": 218,
  "calls": ["fetchBatch", "validateResponse", "applyBlocks"],
  "called_by": ["Start", "restartSyncIfNeeded"]
}
```

**任务 2：函数调用图（Call Graph）构建**

基于 Task 1 提取的调用关系，构建有向调用图，持久化为 `./output/preprocess/<client>_callgraph.json`：

```json
{
  "nodes": ["(*Service).Start", "(*Service).runInitialSync", "fetchBatch", ...],
  "edges": [
    {"caller": "(*Service).Start", "callee": "(*Service).runInitialSync"},
    {"caller": "(*Service).runInitialSync", "callee": "fetchBatch"}
  ],
  "entry_points": {
    "initial_sync":    ["(*Service).Start", "(*Service).runInitialSync"],
    "regular_sync":    ["(*Service).runRegularSync"],
    "block_generate":  ["(*ValidatorService).ProposeBlock"],
    "attestation_generate": ["(*ValidatorService).SubmitAttestation"]
  }
}
```

> **entry_points 说明**：通过正则匹配关键方法名（如含 `Sync`、`Propose`、`Attest` 关键字的顶层 Service 方法）自动识别 7 个 Workflow 的代码入口函数，作为调用图遍历的起点。

**任务 3：构建增强向量索引**

在标准 `RecursiveCharacterTextSplitter.from_language` 分割的基础上，为每个 Document chunk 注入调用图 metadata，存入向量数据库：

```python
# 标准分割后，注入增强 metadata
chunk.metadata = {
    "client_name":     "prysm",
    "language":        "go",
    "file_path":       "beacon-chain/sync/service.go",
    "function_name":   "runInitialSync",
    "qualified_name":  "(*Service).runInitialSync",
    "start_line":      142,
    "end_line":        218,
    "call_depth":      2,           # 距离 entry_point 的调用深度
    "workflow_hints":  ["initial_sync"],  # 该函数所属 Workflow 的推断标签
    "callers":         ["(*Service).Start"],
    "callees":         ["fetchBatch", "validateResponse"],
}
```

向量数据库配置要求：
- 按 `client_name` 进行 **collection 隔离**，避免跨客户端检索污染
- 支持**持久化**：首次构建后可跨进程复用，源码未变更时跳过重建
- 推荐使用针对代码语义训练的 Embedding 模型（如 `text-embedding-3-large` 或开源的 `nomic-embed-code`）

**任务 4：构建 BM25 精确匹配索引**

使用 `rank-bm25` 对同一批代码 chunk 构建 BM25 索引，用于精确匹配函数名、Guard/Action 名称等关键词：

```python
from rank_bm25 import BM25Okapi

# 以 token 列表（标识符分词）为单位建立索引
# 持久化为 ./output/preprocess/<client>_bm25.pkl
```

---

## Phase 1 — 规范提炼（Spec Enrichment）

**目标**：在规范预定义的 Guard/Action 词汇基础上，从各客户端源码中发现新词汇，迭代收敛后输出 `Global_LSG_Spec_Enriched.yaml`。

### 协作模型

```
Main Agent (协调者)
    ├─ Sub-Agent: prysm
    ├─ Sub-Agent: lighthouse
    ├─ Sub-Agent: grandine
    ├─ Sub-Agent: teku
    └─ Sub-Agent: lodestar
```

每个客户端对应一个独立的 Sub-Agent。

### Sub-Agent 职责
1. 接收 Main Agent 下发的当前词汇表版本
2. 使用 **混合检索工具**（见 Step 3）检索本客户端源码，发现**规范词汇表中尚未收录的** Guard/Action 候选词汇
3. 每个候选词汇必须附带：名称、分类、语义描述、来源 evidence（文件路径 + 函数名 + 行号）
4. 将发现结果返回给 Main Agent

### Main Agent 职责
1. **汇总与去重**：合并各 Sub-Agent 返回的候选词汇，识别同义词和重复词，执行去重与归一化（例如：将语义相同的 `SendBlockRequest` 和 `SendRangeRequest` 合并为规范名称）
2. **计算差异率**：与上一轮词汇表进行比较
   ```
   diff_rate = 本轮新增词汇数 / 当前词汇总量
   ```
3. **版本控制**：将合并后的词汇表保存为新版本，生成 Checkpoint 快照
4. **反馈驱动**：将更新后的词汇表下发给各 Sub-Agent，驱动下一轮迭代
5. **收敛判断**：当 `diff_rate < 5%` 时，退出迭代，输出 `Global_LSG_Spec_Enriched.yaml`

### 迭代控制
- 最大迭代次数：`MAX_ITER_PHASE1 = 20`
- 收敛条件：`diff_rate < 0.05`
- 路由实现：必须使用 LangGraph `Conditional Edges`

---

## Phase 2 — LSG 提取与比对（LSG Extraction & Diff）

**目标**：基于 `Global_LSG_Spec_Enriched.yaml`，各客户端 Sub-Agent 提取 7 个核心 Workflow 的完整 LSG，Main Agent 横向比对并分类差异。

### Sub-Agent 职责
1. 接收 Main Agent 下发的统一词汇表
2. 使用**调用图导向的混合检索**（见 Step 3），以 7 个 Workflow 的入口函数为起点，沿调用链路检索实现代码
3. 每条状态迁移的 `evidence` 字段**必须填写**（文件路径 + 函数名 + 行号）——Phase 2 的强制要求
4. 每轮生成 `LSG_<ClientName>_iter<N>.yaml`，格式严格遵守 `LSG_Schema_Spec.md`

### Main Agent 职责
1. **横向比对**：以 `(workflow_id, state_id, transition_guard)` 三元组为 key，比对各客户端的 LSG 结构
2. **差异分类**：
   - **A 类（实现差异）**：同一 Guard/Action 在不同客户端有不同的代码实现路径，但业务语义等价。例如：同为 `ApplyBatch`，Prysm 通过 `processBatch()` 实现，Lighthouse 通过 `process_chain_segment()` 实现 → 属于多态实现，**不计入最终差异**
   - **B 类（逻辑差异）**：在状态机结构、Guard 条件组合或 Action 序列上存在实质性业务逻辑差异 → **保留至最终报告**，标注涉及的客户端，供人工审查
3. **反馈 A 类差异**：将 A 类差异反馈给相关 Sub-Agent，要求对齐抽象层描述
4. **计算逻辑差异率**：
   ```
   logic_diff_rate = B 类差异项数 / 所有比对项总数
   ```
5. **收敛判断**：当 `logic_diff_rate < 5%` 时，输出最终报告

### 迭代控制
- 最大迭代次数：`MAX_ITER_PHASE2 = 20`
- 收敛条件：`logic_diff_rate < 0.05`

---

## 横切关注点（Cross-Cutting Concerns）

| 关注点 | 具体要求 |
|---|---|
| **RAG 检索策略** | 必须使用**混合检索**（Hybrid Search）：BM25 精确匹配 + 向量语义检索，通过 `EnsembleRetriever` 加权融合（建议权重 BM25:Vector = 4:6）；Phase 2 额外叠加**调用图导向检索**（见 Step 3）|
| **Evidence 完整性** | Phase 1 的词汇候选需附带 evidence；Phase 2 的每条状态迁移**必须**包含 evidence（`file` + `function` + `lines`） |
| **审计日志** | 每次 LLM 调用前后，将 Prompt、Chain-of-Thought、LLM 原始回复结构化保存为 JSON，文件名格式：`audit_phase<P>_iter<N>_<agent_type>_<timestamp>.json` |
| **快照与版本控制** | 每轮迭代后 Main Agent 将完整 State 序列化为 Checkpoint，文件名：`checkpoint_phase<P>_iter<N>.json`，支持断点续跑 |
| **防死循环** | 每个阶段设置 `MAX_ITER` 上限，超限后强制退出并在 State 写入 `force_stopped: true` 警告 |
| **条件路由** | 所有收敛判断和阶段跳转通过 LangGraph `add_conditional_edges` 实现，不得使用轮询或 sleep |

---

# 开发执行计划 (Execution Plan)

> **协作约定**：严格按照以下步骤顺序开发。**每完成一步并输出结果后，必须暂停，等待我的 Review 和明确 "Approve" 指令后，再进入下一步。禁止跳步或提前实现后续步骤的功能。**

---

## Step 1 — 系统设计文档（Architecture Document）

**任务**：基于上述需求，输出一份完整的系统架构设计文档，作为后续所有开发步骤的蓝图。

**产出要求**：

1. **全局状态字典 `GlobalState`**：
   - 使用 `TypedDict` 或 `Pydantic BaseModel` 定义
   - 必须覆盖：当前阶段号、当前迭代计数器、Guard/Action 词汇表（含版本号）、各客户端当前迭代的 LSG 内容、差异率、收敛标志、`force_stopped` 标志、预处理完成标志、审计日志路径列表等
   - 说明每个字段在哪个节点被写入、在哪个节点被读取

2. **LangGraph 节点拓扑图**（使用 Mermaid 语法绘制）：
   - 包含**离线预处理节点**（Offline Preprocessing）作为图的起始节点
   - 涵盖 Phase 1 和 Phase 2 的完整节点和边
   - 明确标注所有 Conditional Edges 及其判断条件
   - 明确标注 Sub-Agent 节点的并行执行关系（`Send` API / fan-out 模式）

3. **文件产出清单**：每个阶段、每次迭代产出的文件名规范及目录结构

4. **后续开发任务清单**：按 Step 2～5 拆解，每个模块注明文件名、主要类/函数、对外接口签名、关键技术依赖

**验收标准**：`GlobalState` 字段覆盖两个阶段所有需求；Mermaid 图能正确渲染，完整表达预处理→Phase1→Phase2 的流转逻辑；任务清单具体到函数/方法签名粒度。

---

## Step 2 — 核心图结构框架（Graph Infrastructure）

**任务**：搭建 LangGraph 图骨架，暂不实现真实 LLM 调用，重点验证图拓扑的正确性与状态流转。

**产出要求**：

1. **`config.py`**：集中管理所有配置常量
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
   ```

2. **`state.py`**：定义 `GlobalState` 及必要的子结构（如 `VocabEntry`、`LSGWorkflow`、`PreprocessStatus` 等），与 Step 1 设计文档保持一致

3. **`graph.py`**：
   - 实例化 `StateGraph(GlobalState)`
   - 包含**离线预处理 Mock 节点**（`preprocess_node`），作为图的起始节点，检查预处理产物是否已存在，存在则跳过
   - 所有 Agent 节点使用 Mock 函数（返回固定占位数据，打印当前节点名、阶段、迭代计数）
   - 实现 `router_phase1` 和 `router_phase2`，包含收敛判断和 `MAX_ITER` 防死循环逻辑
   - Sub-Agent 节点使用 LangGraph `Send` API 实现 fan-out 并行调度

**验收标准**：`graph.compile()` 无报错；Mock 数据能完整跑通预处理 → Phase 1 → Phase 2 并正常退出；`MAX_ITER` 能正确触发强制退出。

---

## Step 3 — 代码解析、静态分析与 RAG 工具链（Tooling & RAG）

**任务**：实现离线预处理管道和 Sub-Agent 混合检索工具，这是整个系统召回质量的核心保障。

### 3.1 离线预处理模块

**`tools/preprocessor.py`**：实现完整的静态分析预处理管道，提供入口函数 `run_preprocessing(client_name: str, force_rebuild: bool = False)`，内部按顺序执行以下四个任务：

**Task A — AST 符号提取**（`_extract_symbols`）：

```python
def _extract_symbols(client_name: str) -> list[SymbolInfo]:
    """
    使用 tree-sitter 解析对应语言源码，提取所有函数/方法的：
    - file_path（相对路径）
    - function_name（短名）
    - qualified_name（含 receiver/class 的全限定名）
    - start_line / end_line
    - 函数体 source_code（用于后续 chunk 分割）
    - 直接调用的函数名列表 calls（通过 AST call_expression 节点提取）
    输出持久化至 ./output/preprocess/<client>_symbols.json
    """
```

各语言的 tree-sitter AST 节点类型映射：

| 语言 | 函数定义节点类型 | 调用表达式节点类型 | 函数名字段 |
|---|---|---|---|
| Go | `function_declaration` / `method_declaration` | `call_expression` | `name` / `field_name` |
| Rust | `function_item` | `call_expression` / `method_call_expression` | `name` |
| Java | `method_declaration` | `method_invocation` | `name` |
| TypeScript | `function_declaration` / `method_definition` | `call_expression` | `name` |

**Task B — 调用图构建**（`_build_callgraph`）：

```python
def _build_callgraph(client_name: str, symbols: list[SymbolInfo]) -> CallGraph:
    """
    基于 symbols 中的 calls 关系构建有向调用图。
    额外任务：自动识别 7 个 Workflow 的入口函数（entry_points），
    通过以下规则匹配：
      - 方法名包含关键词（大小写不敏感）：
        initial_sync    → ["initialsync", "runinitial", "startinitial"]
        regular_sync    → ["regularsync", "runregular", "gossipsync"]
        checkpoint_sync → ["checkpointsync", "runcheckpoint"]
        block_generate  → ["proposeblock", "buildblock", "produceblock"]
        attestation_generate → ["submitattestation", "createattestation"]
        aggregate       → ["aggregate", "computeaggregate"]
        execute_layer_relation → ["engineapi", "executionengine", "forkchoiceupdate"]
      - 匹配结果需人工可覆盖（通过 config 中的 ENTRY_POINT_OVERRIDES 字典）
    输出持久化至 ./output/preprocess/<client>_callgraph.json
    """
```

**Task C — 增强向量索引构建**（`_build_vector_index`）：

```python
def _build_vector_index(client_name: str, symbols: list[SymbolInfo], callgraph: CallGraph):
    """
    基于 symbols 的 source_code 字段，使用 RecursiveCharacterTextSplitter.from_language
    进行语言感知分割，并为每个 chunk 注入调用图增强 metadata：
      - client_name, language, file_path
      - function_name, qualified_name
      - start_line, end_line（chunk 在文件中的实际行号）
      - call_depth: 该函数距最近 entry_point 的最短调用路径长度
        （通过 BFS 遍历调用图计算，若不可达则设为 999）
      - workflow_hints: 根据 call_depth 和调用链路推断该 chunk 属于哪些 Workflow
      - callers, callees: 直接调用者和被调用者列表
    存入 Chroma collection，collection 名称为 client_name，持久化目录为
    ./output/preprocess/<client>_chroma/
    """
```

Embedding 模型优先级（按代码语义理解能力排序）：
1. `nomic-embed-code`（开源，专为代码设计）
2. `text-embedding-3-large`（OpenAI，通用但效果好）
3. `all-MiniLM-L6-v2`（轻量备选，精度较低）

**Task D — BM25 精确匹配索引构建**（`_build_bm25_index`）：

```python
def _build_bm25_index(client_name: str, symbols: list[SymbolInfo]):
    """
    对所有函数体进行标识符级别的 tokenization（按驼峰/蛇形命名拆分，
    保留原始 token 和拆分后的子词），使用 BM25Okapi 建立索引。
    持久化为 ./output/preprocess/<client>_bm25.pkl。
    tokenization 示例：
      "runInitialSync" → ["runInitialSync", "run", "Initial", "Sync"]
      "process_chain_segment" → ["process_chain_segment", "process", "chain", "segment"]
    """
```

---

### 3.2 混合检索接口

**`tools/search.py`**：封装统一的 Tool 接口，提供两种检索模式：

**模式 A — 语义混合检索**（Phase 1 使用，词汇发现场景）：

```python
@tool
def search_codebase(
    query: str,
    client_name: str,
    top_k: int = 5
) -> list[Document]:
    """
    执行混合检索（Hybrid Search）：
    1. BM25 检索：对 query 进行标识符 tokenization 后检索，返回 top_k 结果
    2. 向量检索：对 query 进行 embedding 后检索，返回 top_k 结果
    3. EnsembleRetriever 加权融合（BM25:Vector = 4:6）去重后返回 top_k 结果
    返回的 Document.metadata 包含完整 evidence 信息。
    """
```

**模式 B — 调用图导向检索**（Phase 2 使用，LSG 提取场景）：

```python
@tool
def search_codebase_by_workflow(
    workflow_id: str,
    query: str,
    client_name: str,
    max_call_depth: int = 5,
    top_k: int = 10
) -> list[Document]:
    """
    调用图导向的混合检索，专为 LSG 提取设计：
    1. 从 callgraph 中获取 workflow_id 对应的 entry_points
    2. BFS 遍历调用图，收集 call_depth <= max_call_depth 的所有函数节点
    3. 在上述函数集合内执行混合检索（BM25 + 向量，同模式 A）
    4. 结果按 call_depth 升序排列（越接近入口函数越靠前）
    这样确保检索范围锚定在 Workflow 相关的代码子图上，
    避免全库检索带来的噪声。
    返回的 Document.metadata 额外包含 call_depth 和 workflow_hints 字段。
    """
```

**验收标准**：
- 对每个客户端的 `initial_sync` Workflow 执行一次 `search_codebase_by_workflow`，返回结果中 `call_depth` 字段正确，最浅层结果应为 `Start` / `Run` 等顶层 Service 方法
- BM25 索引对精确函数名（如 `runInitialSync`）的召回率应为 100%
- Chroma 持久化后，第二次启动无需重建索引
- 预处理产物（`_symbols.json`、`_callgraph.json`、`_bm25.pkl`、`_chroma/`）均存在时，`run_preprocessing` 应在 < 1 秒内完成（跳过重建）

---

## Step 4 — Agent 逻辑与 Prompt 集成（Agents Implementation）

**任务**：将 Mock 节点替换为真实的 LangChain Agent 执行逻辑，集成 Prompt 模板和结构化输出, 模型选择 Claude 模型。

**产出要求**：

1. **`agents/prompts/`**：存放所有 Prompt 模板（`.jinja2` 格式），与代码解耦

2. **`agents/phase1_sub_agent.py`**：
   - 绑定 `search_codebase` 工具（模式 A）的 ReAct Agent
   - Prompt 要求 Agent 对照当前词汇表，通过多次检索发现未收录的 Guard/Action 候选词汇
   - 使用 `with_structured_output` 强制输出 `VocabDiscoveryReport` Pydantic 模型

3. **`agents/phase1_main_agent.py`**：
   - 接收各 Sub-Agent 的 `VocabDiscoveryReport`，执行去重/同义词合并
   - 使用 `with_structured_output` 强制输出 `EnrichedSpec` Pydantic 模型
   - 在 State 中写入本轮 `diff_rate`

4. **`agents/phase2_sub_agent.py`**：
   - 绑定 `search_codebase_by_workflow` 工具（模式 B）的 ReAct Agent
   - Prompt 要求 Agent 针对 7 个 Workflow ID，从入口函数出发沿调用链路提取完整状态机
   - 每条 transition **必须**包含 evidence
   - 使用 `with_structured_output` 强制输出 `LSGFile` Pydantic 模型

5. **`agents/phase2_main_agent.py`**：
   - 横向比对各客户端 LSG，以 `(workflow_id, state_id, transition_guard)` 为 key 进行三元组比对
   - 分类输出 A 类和 B 类差异，使用 `with_structured_output` 输出 `DiffReport` Pydantic 模型
   - 在 State 中写入本轮 `logic_diff_rate`

**验收标准**：每个 Agent 节点可独立单元测试（提供 Mock LLM 测试用例）；所有结构化输出通过 Pydantic 校验；Prompt 变量与代码输入字段一一对应。

---

## Step 5 — 文件持久化与系统收尾（File I/O & Output）

**任务**：完善所有阶段的 Exit Node，实现产出物落盘与审计日志记录。

**产出要求**：

1. **`io/checkpoint.py`**：
   - `save_checkpoint(state: GlobalState, phase: int, iteration: int) -> Path`
   - `load_checkpoint(phase: int, iteration: int) -> GlobalState`
   - 文件：`./output/checkpoints/checkpoint_phase<P>_iter<N>.json`

2. **`io/writer.py`**：
   - Phase 1 退出：`./output/Global_LSG_Spec_Enriched.yaml`（严格符合 `LSG_Schema_Spec.md` 格式）
   - Phase 2 退出：
     - `./output/LSG_<ClientName>_final.yaml` × 5
     - `./output/Audit_Diff_Report.md`（含差异摘要表、每条 B 类差异的详细描述：��及客户端、Workflow、状态节点、预期行为 vs 实际行为、关联 evidence）
   - 中间产物：`./output/iterations/LSG_<ClientName>_iter<N>.yaml`

3. **`io/audit_logger.py`**：
   - 通过 LangChain Callback 自动触发，无需业务代码手动调用
   - 结构化写入 `./output/audit_logs/audit_phase<P>_iter<N>_<agent_type>_<timestamp>.json`

**验收标准**：全流程端到端运行后所有文件内容完整、格式合法；`Global_LSG_Spec_Enriched.yaml` 与 `LSG_Schema_Spec.md` 格式完全兼容；Checkpoint 支持断点续跑。

---

# 预期项目目录结构

```
EthAuditor/
├── config.py
├── state.py
├── graph.py
│
├── agents/
│   ├── prompts/
│   │   ├── phase1_sub.j2
│   │   ├── phase1_main.j2
│   │   ├── phase2_sub.j2
│   │   └── phase2_main.j2
│   ├── phase1_sub_agent.py
│   ├── phase1_main_agent.py
│   ├── phase2_sub_agent.py
│   └── phase2_main_agent.py
│
├── tools/
│   ├── preprocessor.py      # 离线预处理管道（AST + 调用图 + 索引构建）
│   └── search.py            # search_codebase / search_codebase_by_workflow
│
├── io/
│   ├── checkpoint.py
│   ├── writer.py
│   └── audit_logger.py
│
├── docs/
│   └── LSG_Schema_Spec.md
│
├── code/                    # 客户端源码（只读）
│   ├── prysm/
│   ├── lighthouse/
│   ├── grandine/
│   ├── teku/
│   └── lodestar/
│
└── output/
    ├── preprocess/          # 离线预处理产物（只生成一次）
    │   ├── <client>_symbols.json
    │   ├── <client>_callgraph.json
    │   ├── <client>_bm25.pkl
    │   └── <client>_chroma/
    ├── checkpoints/
    ├── iterations/
    ├── audit_logs/
    ├── Global_LSG_Spec_Enriched.yaml
    ├── LSG_<ClientName>_final.yaml  (×5)
    └── Audit_Diff_Report.md
```