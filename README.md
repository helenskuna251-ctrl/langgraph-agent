# 🤖 LangGraph Agent · 汽车智能客服系统

基于 **LangGraph** 构建的多Agent汽车智能客服系统，采用图结构编排4个专业Agent节点，实现意图识别、RAG知识检索、智能问答与答案质量审核的完整工作流。

## 📋 项目概述

传统客服系统依赖关键词匹配，无法理解用户真实意图。本项目利用 LangGraph 的状态图机制，将复杂的客服流程拆解为多个Agent节点协作完成，实现汽车专业问题的精准检索与高质量回答。

## 🏗️ 系统架构

```
用户输入
  │
  ▼
┌─────────────┐
│  意图识别节点  │  ← Agent 1: 判断问题类型
└──────┬──────┘
       │
       ▼
┌──────────────┐
│  RAG检索节点   │  ← Agent 2: FAISS向量检索
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  答案生成节点   │  ← Agent 3: GLM-4生成回答
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   审核节点     │  ← Agent 4: 质量控制
└──────┬───────┘
    ┌──┴──┐
    │     │
   通过  不通过 → 重新生成
    │
    ▼
┌──────────────┐
│    输出回答    │
└──────────────┘
```

## ✨ 核心特性

**多Agent协作架构**
- 4个功能节点：意图识别 → RAG检索 → 答案生成 → 质量审核
- 基于 LangGraph StateGraph 构建有状态工作流
- 所有问题经意图识别后进入RAG检索链路，确保专业知识覆盖

**RAG知识检索**
- 使用 BAAI/bge-small-zh 中文向量模型进行文本嵌入
- FAISS 向量数据库实现高效相似度检索
- 支持汽车售后领域专业知识问答

**质量控制机制**
- 审核节点对生成答案进行质量评估
- 不合格答案自动触发重新生成
- 确保输出内容的准确性与专业性

**工程化设计**
- 多轮对话记忆（MemorySaver），保持上下文连贯
- 指数退避重试机制，应对API调用异常
- 结构化日志系统，便于问题排查与监控
- 环境变量管理（.env），安全存储敏感信息

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| Agent框架 | LangChain + LangGraph |
| 大语言模型 | 智谱AI GLM-4-Flash |
| 向量模型 | BAAI/bge-small-zh |
| 向量数据库 | FAISS |
| 开发语言 | Python |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/helenskunda251-ctrl/langgraph-agent.git
cd langgraph-agent
```

### 2. 安装依赖

```bash
pip install langchain langgraph langchain-community faiss-cpu
pip install zhipuai
pip install sentence-transformers
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
ZHIPUAI_API_KEY=your_api_key_here
```

### 4. 运行

```bash
python langgraph_workflow.py
```

## 💡 设计思路

**为什么选择 LangGraph？**

相比传统的 LangChain Chain 线性调用，LangGraph 支持：
- **条件分支**：根据意图识别结果动态调整检索策略
- **循环控制**：审核不通过时可以回到生成节点重新生成
- **状态管理**：通过 StateGraph 统一管理对话状态，天然支持多轮对话

**为什么需要审核节点？**

在实际业务场景中，大模型可能生成不准确或偏离主题的回答。审核节点作为"质量守门人"，确保每次输出都符合业务要求，体现了生产级AI应用的工程化思维。

## 📂 项目结构

```
langgraph-agent/
├── langgraph_workflow.py   # 主程序：Agent工作流定义与运行
├── .env                    # 环境变量配置（需自行创建）
└── README.md               # 项目说明
```

## 🔗 相关项目

- [AutoRAG · 汽车售后智能问答系统](https://github.com/helenskunda251-ctrl/autorag-rag) — 基于 FastAPI + FAISS + Reranker 的RAG系统，支持PDF解析、父子Chunk分块、多车型索引隔离、Docker部署

## 📄 License

MIT License
