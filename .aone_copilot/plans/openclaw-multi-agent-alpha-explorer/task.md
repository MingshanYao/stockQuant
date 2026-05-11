### openclaw-multi-agent-alpha-explorer ###
# OpenClaw Multi-Agent Alpha 探索系统 — 任务清单

## Phase 1: 核心基础设施

- [ ] 实现自定义表达式因子引擎 `stockquant/research/expr_engine.py`
  - [ ] AST 安全解析器（白名单算子）
  - [ ] 内置时序算子：`ts_mean`, `ts_std`, `ts_corr`, `delta`, `returns`, `rank`, `delay`
  - [ ] 面板数据输入适配（open/high/low/close/volume → 因子面板 DataFrame）
  - [ ] 单元测试 `tests/test_expr_engine.py`

- [ ] 实现因子注册表 `stockquant/research/factor_registry.py`
  - [ ] FactorRecord 数据结构定义
  - [ ] JSON 持久化存储（CRUD 操作）
  - [ ] 按绩效指标排序检索
  - [ ] 单元测试 `tests/test_factor_registry.py`

- [ ] 实现探索轮次管理器 `stockquant/research/exploration_manager.py`
  - [ ] ExplorationRound 生命周期管理（create/pause/resume/complete）
  - [ ] Markdown 探索报告生成
  - [ ] 轮次持久化与恢复
  - [ ] 单元测试 `tests/test_exploration_manager.py`

## Phase 2: MCP Server 开发

- [ ] 创建 MCP 模块 `stockquant/mcp/__init__.py`
- [ ] 实现 MCP 工具函数 `stockquant/mcp/tools.py`
  - [ ] `compute_alpha101` — Alpha101 因子计算
  - [ ] `compute_custom_factor` — 自定义表达式因子计算
  - [ ] `run_backtest` — 单因子回测
  - [ ] `analyze_factor` — IC/IR 分析
  - [ ] `get_performance_report` — 绩效报告
  - [ ] `list_factors` / `save_factor` — 因子注册表操作
- [ ] 实现 MCP Server 入口 `stockquant/mcp/factor_lab_server.py`
- [ ] MCP 工具函数单元测试 `tests/test_mcp_tools.py`

## Phase 3: OpenClaw Agent 配置

- [ ] 创建 OpenClaw 网关配置 `openclaw.json`
- [ ] 编写 Researcher Agent workspace (`agents/researcher/SOUL.md`, `IDENTITY.md`)
- [ ] 编写 Coder Agent workspace (`agents/coder/SOUL.md`, `IDENTITY.md`)
- [ ] 编写 Backtester Agent workspace (`agents/backtester/SOUL.md`, `IDENTITY.md`)
- [ ] 编写 Analyst Agent workspace (`agents/analyst/SOUL.md`, `IDENTITY.md`)

## Phase 4: 配置与集成

- [ ] 更新 `pyproject.toml` 添加 mcp 依赖组
- [ ] 更新 `config/default.yaml` 添加探索配置项
- [ ] 端到端集成测试 `tests/test_e2e_exploration.py`

## Phase 5: 人工验证

- [ ] 用户安装 OpenClaw 并配置 Agent 团队
- [ ] 启动 MCP Server 验证 Agent 调用链路
- [ ] 执行首轮因子探索并验证完整流程


updateAtTime: 2026/5/11 14:20:21

planId: 7caae312-7faf-4912-af1b-e45a804785ee