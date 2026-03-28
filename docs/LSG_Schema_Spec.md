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
| `ModeIsInitialSync` | mode | 节点处于���始同步模式 |
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

系统分为 **2 个串行阶段（Phase）**，每个阶段内部通过多轮迭代驱动收敛：

```
Phase 1: 规范提炼（Spec Enrichment）
    输入: LSG_Schema_Spec.md（基础词汇表）
    输出: Global_LSG_Spec_Enriched.yaml（富化后的统一词汇表）

Phase 2: LSG 提取与比对（LSG Extraction & Diff）
    输入: Global_LSG_Spec_Enriched.yaml
    输出: LSG_<ClientName>_final.yaml × 5 个客户端
          Audit_Diff_Report.md（标注 B 类逻辑差异，供人工审查）
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
2. 使用 RAG 工具检索本客户端源码，发现**规范词汇表中尚未收录的** Guard/Action 候选词汇
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
5. **收敛判断**：当 `diff_rate < 5%` 时，退出迭代，将最终词汇表输出为 `Global_LSG_Spec_Enriched.yaml`

### 迭代控制
- 最大迭代次数：`MAX_ITER_PHASE1 = 20`（超限强制退出并记录警告）
- 收敛条件：`diff_rate < 0.05`
- 路由实现：必须使用 LangGraph `Conditional Edges`，不得使用轮询或 sleep

---

## Phase 2 — LSG 提取与比对（LSG Extraction & Diff）

**目标**：基于 `Global_LSG_Spec_Enriched.yaml`，各客户端 Sub-Agent 提取 7 个核心 Workflow 的完整 LSG，Main Agent 横向比对并分类差异。

### Sub-Agent 职责
1. 接收 Main Agent 下发的统一词汇表
2. 使用 RAG 工具检索本客户端源码，提取 7 个 Workflow 的完整状态机
3. 每条状态迁移的 `evidence` 字段**必须填写**（文件路径 + 函数名 + 行号），这是 Phase 2 的强制要求
4. 每轮生成 `LSG_<ClientName>_iter<N>.yaml`，格式严格遵守 `LSG_Schema_Spec.md`，并返回给 Main Agent

### Main Agent 职责
1. **横向比对**：以 `workflow.id` → `state.id` → `transition` 为粒度，比对各客户端的 LSG 结构
2. **差异分类**（关键）：
   - **A 类（实现差异）**：同一 Guard/Action 在不同客户端有不同的代码实现路径，但业务语义等价。例如：同为 `ApplyBatch`，Prysm 通过 `processBatch()` 实现，Lighthouse 通过 `process_chain_segment()` 实现 → 属于多态实现，**不计入最终差异**
   - **B 类（逻辑差异）**：在状态机结构、Guard 条件组合或 Action 序列上存在实质性业务逻辑差异 → **保留至最终报告**，标注涉及的客户端，供人工审查
3. **反馈 A 类差异**：将 A 类差异的描述反馈给相关 Sub-Agent，要求对齐抽象层描述
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
| **代码解析与 RAG** | 使用 `RecursiveCharacterTextSplitter.from_language` 按语言（Go / Rust / Java / JS）分割代码；向量数据库使用 Chroma 或 FAISS，按 `client_name` 进行 namespace 隔离，支持持久化复用 |
| **Evidence 完整性** | Phase 1 的词汇候选需附带 evidence；Phase 2 的每条状态迁移 **必须**包含 evidence（`file` + `function` + `lines`） |
| **审计日志** | 每次 LLM 调用前后，将 Prompt、Chain-of-Thought 思考过程、LLM 原始回复结构化保存为 JSON，文件名格式：`audit_<phase>_iter<N>_<agent_type>_<timestamp>.json` |
| **快照与版本控制** | 每轮迭代后，Main Agent 将当前完整 State 序列化为 Checkpoint，文件名格式：`checkpoint_phase<P>_iter<N>.json`，支持断点续跑 |
| **防死循环** | 每个阶段均设置 `MAX_ITER` 上限，超限后强制退出并在 State 中写入 `force_stopped: true` 警告标记 |
| **条件路由** | 所有收敛判断和阶段跳转通过 LangGraph `add_conditional_edges` 实现 |

---

# 开发执行计划 (Execution Plan)

> **协作约定**：严格按照以下 5 个步骤顺序开发。**每完成一步并输出结果后，必须暂停，等待我的 Review 和明确 "Approve" 指令后，再进入下一步。禁止跳步或提前实现后续步骤的功能。**

---

## Step 1 — 系统设计文档（Architecture Document）

**任务**：基于上述需求，输出一份完整的系统架构设计文档，作为后续所有开发步骤的蓝图。

**产出要求**：

1. **全局状态字典 `GlobalState`**：
   - 使用 `TypedDict` 或 `Pydantic BaseModel` 定义
   - 必须覆盖：当前阶段号、当前迭代计数器、Guard/Action 词汇表（含版本号）、各客户端当前迭代的 LSG 内容、差异率、收敛标志、`force_stopped` 标志、审计日志路径列表等
   - 说明每个字段在哪个节点被写入、在哪个节点被读取

2. **LangGraph 节点拓扑图**（使用 Mermaid 语法绘制）：
   - 涵盖 Phase 1 和 Phase 2 的完整节点和边
   - 明确标注所有 Conditional Edges 及其判断条件（`diff_rate < 5%`、`logic_diff_rate < 5%`、`iter >= MAX_ITER`）
   - 明确标注 Sub-Agent 节点的并行执行关系（使用 `Send` API 或 `fan-out` 模式）

3. **文件产出清单**：
   - 每个阶段、每次迭代产出的文件名规范（含占位符说明）
   - 最终产出物清单及其目录位置

4. **后续开发任务清单**：
   - 按 Step 2～5 拆解，每个模块注明：文件名、主要类/函数、对外接口签名、关键技术依赖

**验收标准**：
- `GlobalState` 字段覆盖两个阶段的所有需求，无遗漏
- Mermaid 图能正确渲染，完整表达循环迭代与条件跳转逻辑
- 任务清单具体到函数/方法签名粒度

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
   SPEC_PATH: str = "./docs/LSG_Schema_Spec.md"
   ```

2. **`state.py`**：定义 `GlobalState` 及必要的子结构（如 `VocabEntry`, `LSGWorkflow` 等），与 Step 1 设计文档保持一致

3. **`graph.py`**：
   - 实例化 `StateGraph(GlobalState)`
   - 所有 Agent 节点使用 **Mock 函数**（返回固定占位数据，并打印当前节点名、阶段、迭代计数）
   - 实现 `router_phase1(state) -> str` 和 `router_phase2(state) -> str`，包含收敛判断和 `MAX_ITER` 防死循环逻辑
   - 使用 `add_conditional_edges` 连接所有条件边
   - Sub-Agent 节点使用 LangGraph `Send` API 实现 fan-out 并行调度

**验收标准**：
- `graph.compile()` 无报错
- 以 Mock 数据驱动图运行，能完整跑通两个阶段并正常退出
- 日志能清晰打印每轮迭代的阶段号、迭代次数、当前差异率
- 当 Mock 差异率永远不收敛时，`MAX_ITER` 能正确触发强制退出

---

## Step 3 — 代码解析与 RAG 工具链（Tooling & RAG）

**任务**：实现 Sub-Agent 与本地源码库交互的检索工具，为 Phase 1 词汇发现和 Phase 2 证据提取提供支撑。

**产出要求**：

1. **`tools/splitter.py`**：
   - 基于 `RecursiveCharacterTextSplitter.from_language` 为 Go、Rust、Java、JavaScript 实现语言特定分割器
   - 分割后每个 `Document` 的 `metadata` 必须包含：
     ```python
     {
       "client_name": str,   # 客户端名称
       "language": str,       # 编程语言
       "file_path": str,      # 相对仓库根目录的路径
       "start_line": int,     # 代码块起始行（1-based）
       "end_line": int        # 代码块结束行（1-based）
     }
     ```

2. **`tools/indexer.py`**：
   - 构建本地向量检索管道（Chroma 或 FAISS）
   - 按 `client_name` 进行 collection/namespace 隔离，避免跨客户端检索污染
   - 支持持久化：首次构建后可跨进程复用，源码未变更时不重建
   - 提供 `build_index(client_name: str)` 和 `load_index(client_name: str)` 接口

3. **`tools/search.py`**：
   - 封装统一 Tool 接口：
     ```python
     def search_codebase(
         query: str,
         client_name: str,
         top_k: int = 5
     ) -> list[Document]:
         """
         返回的 Document.metadata 必须包含完整 evidence 信息：
         file_path, start_line, end_line, function_name（尽力提取）
         """
     ```
   - 将此函数封装为 LangChain `@tool`，供 Agent 直接调用

**验收标准**：
- 对每个客户端执行一次查询（如："block proposal duty check"），返回结果包含正确的 `metadata`
- Chroma/FAISS 持久化后，第二次启动无需重建索引，查询结果一致
- `function_name` 字段能通过正则或 AST 从代码块中提取（允许近似匹配，精度 ≥ 80%）

---

## Step 4 — Agent 逻辑与 Prompt 集成（Agents Implementation）

**任务**：将 Mock 节点替换为真实的 LangChain Agent 执行逻辑，集成 Prompt 模板和结构化输出。

**产出要求**：

1. **`agents/prompts/`**：存放所有 Prompt 模板（`.jinja2` 格式），与代码解耦，便于独立迭代

2. **`agents/phase1_sub_agent.py`**：
   - 绑定 `search_codebase` 工具的 ReAct Agent
   - Prompt 要求 Agent 对照当前词汇表，检索源码中未收录的 Guard/Action，并输出结构化候选词汇列表（含 evidence）
   - 使用 `with_structured_output` 强制输出 `VocabDiscoveryReport` Pydantic 模型

3. **`agents/phase1_main_agent.py`**：
   - 接收各 Sub-Agent 的 `VocabDiscoveryReport`，执行去重/同义词合并
   - 使用 `with_structured_output` 强制输出 `EnrichedSpec` Pydantic 模型
   - 在 State 中写入本轮 `diff_rate`

4. **`agents/phase2_sub_agent.py`**：
   - 绑定 `search_codebase` 工具的 ReAct Agent
   - Prompt 要求 Agent 针对 7 个 Workflow ID，逐一检索源码，提取完整状态机，**每条 transition 必须填写 evidence**
   - 使用 `with_structured_output` 强制输出 `LSGFile` Pydantic 模型（严格对应 `LSG_Schema_Spec.md` 定义的 YAML 结构）

5. **`agents/phase2_main_agent.py`**：
   - 横向比对各客户端 LSG，以 `(workflow_id, state_id, transition_guard)` 为 key 进行三元组比对
   - 分类输出 A 类（实现差异）和 B 类（逻辑差异），使用 `with_structured_output` 输出 `DiffReport` Pydantic 模型
   - 在 State 中写入本轮 `logic_diff_rate`

**验收标准**：
- 每个 Agent 节点可独立单元测试（提供 Mock LLM 测试用例）
- 所有结构化输出通过对应的 Pydantic 模型校验，无字段缺失
- Prompt 模板中的变量与代码中的输入字段一一对应，无悬空变量

---

## Step 5 — 文件持久化与系统收尾（File I/O & Output）

**任务**：完善所有阶段的 Exit Node，实现产出物落盘、Checkpoint 持久化与审计日志记录。

**产出要求**：

1. **`io/checkpoint.py`**：
   - `save_checkpoint(state: GlobalState, phase: int, iteration: int) -> Path`
   - `load_checkpoint(phase: int, iteration: int) -> GlobalState`
   - 文件存储于 `./output/checkpoints/checkpoint_phase<P>_iter<N>.json`

2. **`io/writer.py`**：
   - Phase 1 Exit Node 调用：将 `EnrichedSpec` 序列化为 `./output/Global_LSG_Spec_Enriched.yaml`，严格符合 `LSG_Schema_Spec.md` 的 YAML 格式
   - Phase 2 Exit Node 调用：
     - 将各客户端最终 LSG 写出为 `./output/LSG_<ClientName>_final.yaml`
     - 将 B 类差异写出为 `./output/Audit_Diff_Report.md`，格式包含：差异摘要表、每条 B 类差异的详细描述（涉及客户端、Workflow、状态节点、预期行为 vs 实际行为、关联 evidence）
   - 每轮迭代的中间 LSG 文件写出为 `./output/iterations/LSG_<ClientName>_iter<N>.yaml`

3. **`io/audit_logger.py`**：
   - 提供 `log_llm_call(phase, iteration, agent_type, prompt, chain_of_thought, llm_response)` 接口
   - 结构化写入 `./output/audit_logs/audit_phase<P>_iter<N>_<agent_type>_<timestamp>.json`
   - 支持在 LangChain Callback 中自动触发，无需在业务代码中手动调用

**验收标准**：
- 全流程端到端运行后，`output/` 目录下所有文件内容完整、格式合法（YAML 可被标准解析器加载，Markdown 格式正确）
- `Global_LSG_Spec_Enriched.yaml` 的结构与 `LSG_Schema_Spec.md` 定义完全兼容
- 使用 Checkpoint 文件可从任意迭代断点续跑，最终产出物一致

---

# 预期项目目录结构

```
EthAuditor/
├── config.py                  # 全局配置常量
├── state.py                   # GlobalState 及子结构定义
├── graph.py                   # LangGraph 图实例化与路由
│
├── agents/
│   ├── prompts/               # Jinja2 Prompt 模板
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
│   ├── splitter.py            # 多语言代码分割器
│   ├── indexer.py             # 向量索引构建与加载
│   └── search.py             # search_codebase Tool
│
├── io/
│   ├── checkpoint.py          # Checkpoint 序列化/反序列化
│   ├── writer.py              # YAML / Markdown 文件落盘
│   └── audit_logger.py        # LLM 调用审计日志
│
├── docs/
│   └── LSG_Schema_Spec.md     # 原始规范文件（只读）
│
├── code/                      # 客户端源码（只读）
│   ├── prysm/
│   ├── lighthouse/
│   ├── grandine/
│   ├── teku/
│   └── lodestar/
│
└── output/                    # 所有生成文件
    ├── checkpoints/           # checkpoint_phase<P>_iter<N>.json
    ├── iterations/            # LSG_<ClientName>_iter<N>.yaml
    ├── audit_logs/            # audit_phase<P>_iter<N>_<type>_<ts>.json
    ├── Global_LSG_Spec_Enriched.yaml
    ├── LSG_<ClientName>_final.yaml  (×5)
    └── Audit_Diff_Report.md
```