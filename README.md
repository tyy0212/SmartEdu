# SmartEdu
## 项目概述

这是一个**智能教育 MCP 代理平台**，实现了一个具有多模态功能的 AI 教育助手。该项目基于真实框架构建，包含一个符合模型上下文协议（MCP）的服务器，提供教育任务工具，并结合了一个使用反思架构（包含规划器和审核器节点）的代理系统。

## 架构

### 核心组件

1. **MCP 服务器** (`src/mcp_server.py`, `src/server.py`):
   - 基于 FastAPI 实现的自定义 MCP 服务器，符合 MCP 协议规范，提供标准端点：`/tools`, `/health`, `/openapi.json`
   - 包含四个教育工具（定义在 `src/server.py`）：
     - `check_typos_logic`: NLP 工具，使用 `pylanguagetool` 进行中文语法和错别字检查，支持自动纠正和错误报告（真实实现）
     - `search_knowledge_base`: RAG 工具，集成向量数据库（ChromaDB + SentenceTransformer）进行语义搜索，包含默认教学大纲数据（真实实现）
     - `analyze_classroom_video`: 多模态工具，用于分析课堂视频（当前为模拟实现，待升级）
     - `analyze_image`: VLM 工具，使用 LLaVA 模型进行具有教育上下文的图像分析（支持真实 LLaVA 模型，依赖缺失时优雅回退到模拟结果）
   - 包含模拟知识库：`SYLLABUS_DB` 和 `TYPO_MAPPING`（位于 `src/server.py`），作为回退数据源
   - 支持工具发现、参数验证、标准化错误处理和优雅回退机制

2. **向量数据库模块** (`src/vector_db.py`):
   - 实现 ChromaDB 向量知识库，支持语义搜索和默认教学大纲数据初始化
   - 使用 `sentence-transformers` 模型 (`paraphrase-multilingual-MiniLM-L12-v2`) 生成中文嵌入
   - 提供单例访问模式，失败时回退到模拟数据
   - 数据持久化在 `./data/chroma_db` 目录

3. **代理系统** (`src/agent.py`):
   - 基于真实 `langgraph` 库实现的状态图，采用反思架构
   - **状态**: `AgentState` 包含消息、计划、审核反馈、重试计数和图像分析结果
   - **节点**:
     - `planner_node`: 调用 MCP 工具生成教学计划，当引用图像时整合图像分析，使用 LLM 生成最终方案
     - `reviewer_node`: 审核计划质量，提供改进反馈
   - **条件边**: 实现了一个"重新规划"循环，用于细化直到批准或达到重试限制（最多3次）
   - **持久化**: 使用 `SqliteSaver` 进行检查点保存（SQLite 数据库），支持跨会话状态恢复
   - **LLM 客户端**: 集成 `LLMClient` 统一客户端，支持 OpenAI 和 Claude API，含配置管理和错误处理
   - **入口点**: `src/main.py` 演示系统，`src/api/main.py` 提供 RESTful API 接口

4. **结构化日志系统** (`src/utils/logger.py`):
   - 基于 `structlog` 实现，支持 JSON 和控制台两种输出格式
   - 可配置日志级别、输出文件和格式
   - 集成到所有模块，提供统一的日志记录体验

5. **RESTful API** (`src/api/main.py`):
   - 基于 FastAPI 实现的外部调用接口，提供智能教育代理的 HTTP 访问
   - 支持用户查询和图像分析请求
   - 包含健康检查端点和会话管理

6. **配置管理系统** (`config/settings.py`):
   - 基于 Pydantic Settings 实现，支持环境变量和 `.env` 文件加载
   - 包含服务器配置、AI 服务配置（OpenAI/Claude API 密钥）、日志配置、数据库配置等
   - 自动创建必要目录（data/, images/）

7. **自适应阈值管理器** (`src/adaptive_threshold.py`):
   - 基于用户反馈动态调整语义相似度阈值，支持个性化阈值配置
   - 采用滑动窗口平均更新阈值，适应不同学科和用户的使用习惯
   - 集成到上下文依赖图谱中，提升代理对历史信息引用检测的准确性

8. **上下文依赖图谱模块** (`src/context_graph.py`):
   - 基于图结构管理对话上下文依赖关系，解决长对话中的上下文丢失问题
   - 借鉴代码图谱(Code Graph)思想，将对话元素构建为有向无环图(DAG)
   - 支持节点类型：用户输入、工具调用、工具结果、计划、审核反馈、图像分析、知识检索等
   - 提供语义相似度计算和依赖关系分析，增强代理对上下文的理解能力

9. **主入口点** (`src/main.py`):
   - 使用多进程设置演示系统（当前为模拟启动）
   - 模拟三轮对话，展示跨"断开连接"的状态持久化
   - 展示 VLM 图像分析功能的集成
   - 注意：当前 `src/main.py` 中的服务器启动为模拟实现，真实 MCP 服务器需要单独启动（`python src/mcp_server.py`）

### 数据流

1. 用户输入 → 代理状态
2. 规划器节点使用 MCP 工具（错别字检查、知识检索、图像分析）→ 生成计划
3. 审核器节点评估计划 → 提供反馈
4. 条件逻辑：如果反馈为 "Approve" → 结束，否则 → 重新规划（最多 3 次重试）
5. 状态通过基于 thread_id 的检查点保存持久化

### 关键设计模式

- **真实框架集成**: 项目使用真实的 `langgraph`、`FastAPI`、`Pydantic`、`ChromaDB`、`sentence-transformers`、`structlog` 等框架构建
- **模块化 MCP 服务器**: 自定义 MCP 服务器实现，支持工具装饰器模式和标准化端点
- **优雅回退机制**: 关键组件（LLaVA 图像分析、向量数据库、LLM API）在依赖缺失或调用失败时优雅回退到模拟结果或备用方案
- **数据库持久化**: 代理状态通过 `SqliteSaver` 按 `user_id` (thread_id) 保存到 SQLite 数据库，支持跨会话连续性
- **多模态集成**: 将文本处理（错别字检查、知识检索）与视觉内容分析（图像分析）相结合，用于教育场景
- **配置化管理**: 使用 Pydantic Settings 实现统一配置管理，支持环境变量和 `.env` 文件，涵盖服务器、AI 服务、日志、数据库等
- **结构化日志**: 集成 `structlog` 提供 JSON 和控制台格式的结构化日志，便于监控和调试
- **统一 LLM 客户端**: 实现 `LLMClient` 统一客户端，支持多模型（OpenAI/Claude）切换和配置管理
- **RESTful API 暴露**: 提供 FastAPI RESTful 接口，支持外部系统集成和调用
- **高级上下文管理**: 实现自适应阈值管理和上下文依赖图谱，提升代理对历史信息的引用检测能力，区分真正的引用与相似话题

## 开发命令

### 运行系统

```bash
# 方式1: 运行完整演示（模拟启动）
python src/main.py

# 方式2: 单独启动 MCP 服务器（真实实现）
python src/mcp_server.py
# 或
python src/server.py

# 方式3: 启动 RESTful API 服务器（提供外部调用接口）
python src/api/main.py
# 或使用 uvicorn 直接启动
uvicorn src.api.main:app --host 0.0.0.0 --port 8002 --reload

# 方式4: 直接运行代理系统（需确保 MCP 服务器已启动）
python -c "from src.agent import run_agent_with_persistence; run_agent_with_persistence('test_user', '测试输入')"

# 注意：当前 `src/main.py` 中的服务器启动为模拟实现，真实使用时需要先启动 MCP 服务器（方式2），然后再运行代理或 API 服务。
# RESTful API 服务器（方式3）依赖 MCP 服务器运行，确保 MCP 服务器在运行或配置正确的 MCP 服务器地址。
```

### 依赖管理

项目使用 `pyproject.toml` 和 `requirements.txt` 进行依赖管理，要求 Python 3.8+。

#### 核心依赖安装：

```bash
# 使用 pip 安装
pip install -r requirements.txt

# 或使用 pip 安装 pyproject.toml 中定义的依赖
pip install .

# 安装开发依赖
pip install ".[dev]"

# 安装 AI 相关依赖（可选，用于真实 LLaVA 等功能）
pip install ".[ai]"

# 安装 Web 界面依赖（可选）
pip install ".[web]"
```

#### 主要依赖包：
- **核心框架**: `langgraph`, `fastapi`, `uvicorn`, `pydantic`
- **向量数据库和 NLP**: `chromadb`, `sentence-transformers`, `pylanguagetool`
- **AI/ML 服务**: `openai`, `anthropic`, `torch`, `transformers`, `Pillow`（可选）
- **日志和配置**: `structlog`, `python-dotenv`
- **开发工具**: `pytest`, `black`, `mypy`, `pre-commit`
- **Web 界面**: `streamlit`（可选）

#### 可选依赖分组（通过 pyproject.toml）：
- `pip install ".[dev]"` - 开发工具（测试、代码格式化、类型检查）
- `pip install ".[ai]"` - AI 相关依赖（LLaVA 模型所需额外包）
- `pip install ".[web]"` - Web 界面依赖（Streamlit 管理界面）

#### LLaVA 集成注意事项：
- `analyze_image` 工具尝试使用 LLaVA 模型，但如果依赖缺失则回退到模拟结果
- LLaVA 安装复杂，可能需要特定设置和模型下载

### 测试

包含基础测试文件 `test_agent.py`，用于验证代理工作流和 MCP 工具调用，以及新增的 `test_context_graph.py` 等测试文件，覆盖上下文依赖图谱和自适应阈值管理模块。

```bash
# 运行测试
pytest test_agent.py

# 或运行所有测试
pytest tests/
```

`src/main.py` 中的演示作为主要功能验证。

### 文件约定

- **项目结构**:
  - `src/`: 主要源代码目录
    - `mcp_server.py`: 自定义 MCP 服务器实现
    - `server.py`: MCP 工具定义（集成真实 AI 服务）
    - `agent.py`: LangGraph 代理实现（含 LLMClient 和 SQLite 持久化）
    - `main.py`: 主入口点（演示）
    - `vector_db.py`: 向量数据库模块（ChromaDB 知识库）
    - `api/main.py`: RESTful API 接口（FastAPI 外部调用）
    - `utils/logger.py`: 结构化日志配置模块（structlog）
    - `adaptive_threshold.py`: 自适应阈值管理器，支持动态阈值调整
    - `context_graph.py`: 上下文依赖图谱模块，管理对话上下文依赖关系
  - `config/`: 配置管理
    - `settings.py`: Pydantic 配置类（环境变量支持）
  - `tests/`: 测试文件
  - `docs/`: 文档
  - `images/`: 图像资源，由图像分析工具引用
  - `data/`: 数据目录（自动创建，存放向量数据库和 SQLite 数据库）
- `.pyc` 文件是编译后的 Python 字节码，应忽略
- 代码使用 UTF-8 编码，包含中文注释和演示文本
