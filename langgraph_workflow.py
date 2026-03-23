import os
import time
import logging
from typing import TypedDict
from zhipuai import ZhipuAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import sys
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== 加载 PDF 建索引 ====================
# 加载问界M8使用手册PDF
loader = PyMuPDFLoader("D:/python/ai_rag_training/pythonProject/data/uploads/问界m8.pdf")
docs = loader.load()

# 分块：每块500字，相邻块重叠50字
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 向量化：使用中文BGE模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh")

# 建FAISS索引
vectorstore = FAISS.from_documents(chunks, embeddings)

# 检索器：MMR模式，先召回10个再取3个，保证多样性
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10})
logger.info("索引加载完成")

# ==================== 初始化LLM ====================
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("ZHIPU_API_KEY")
zhipu_client = ZhipuAI(api_key=api_key)

def call_llm(prompt: str, max_retries: int = 3) -> str:
    """
    调用智谱AI，带重试机制
    失败后指数退避重试，最多3次
    """
    for attempt in range(max_retries):
        try:
            response = zhipu_client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"第{attempt + 1}次调用失败：{e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1秒、2秒、4秒
                logger.warning(f"等待{wait_time}秒后重试...")
                time.sleep(wait_time)
            else:
                return "服务暂时不可用，请稍后再试"

# ==================== 定义状态 ====================
class AgentState(TypedDict):
    messages: list       # 对话历史，实现多轮记忆
    question: str        # 用户当前问题
    intent: str          # 意图识别结果：car_question / chat
    context: str         # RAG检索到的内容
    answer: str          # 生成的答案
    review: str          # 审核结果：pass / retry
    retry_count: int     # 重试次数，防止无限循环

# ==================== 节点定义 ====================

# 节点1：意图识别
# 判断用户问的是汽车相关问题还是闲聊
def intent_node(state: AgentState) -> AgentState:
    question = state["question"]
    prompt = f"""判断以下问题的类型，只回答 car_question 或 chat，不要说其他内容。
问题：{question}
汽车相关问题回答：car_question
其他问题回答：chat"""
    intent = call_llm(prompt).strip()
    logger.info(f"意图识别结果：{intent}")
    return {"intent": intent}

# 节点2：RAG检索
# 从知识库里检索和问题相关的内容
def rag_node(state: AgentState) -> AgentState:
    question = state["question"]
    logger.info(f"开始检索：{question}")
    docs = retriever.invoke(question)
    # 把检索到的多个chunk拼成一段文本
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

# 节点3：生成答案
# 根据检索内容和历史对话生成答案
def answer_node(state: AgentState) -> AgentState:
    question = state["question"]
    intent = state["intent"]
    context = state["context"]
    messages = state["messages"]  # 历史对话

    if intent == "car_question":
        # 汽车问题：带上检索内容和历史
        prompt = f"""你是专业汽车售后工程师，根据以下资料回答问题，使用中文。
历史对话：{messages}
资料：{context}
问题：{question}
回答："""
    else:
        # 闲聊：只带历史，直接回答
        prompt = f"历史对话：{messages}\n请用中文回答：{question}"

    response_text = call_llm(prompt)
    logger.info("生成答案完成")

    # 把这轮对话追加到历史，下一轮能看到
    new_messages = messages + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text}
    ]
    return {"answer": response_text, "messages": new_messages}

# 节点4：审核节点
# 判断答案质量，不好就重新检索生成
def review_node(state: AgentState) -> AgentState:
    question = state["question"]
    answer = state["answer"]
    retry_count = state["retry_count"]

    # 已重试2次，强制通过，防止无限循环
    if retry_count >= 2:
        logger.info("已达最大重试次数，强制通过")
        return {"review": "pass"}

    prompt = f"""你是一个答案质量审核员。
判断以下答案是否真正回答了用户的问题。
只回答 pass 或 retry，不要说其他内容。

用户问题：{question}
答案：{answer}

判断标准：
- 如果答案有具体内容，回答：pass
- 如果答案说"无法找到"、"建议查阅手册"、"无法提供"等，回答：retry
"""

    result = call_llm(prompt).strip().lower()
    logger.info(f"审核结果：{result}")
    # 重试次数+1，写回状态
    return {"review": result, "retry_count": retry_count + 1}

# ==================== 路由函数 ====================

def route_by_intent(state: AgentState) -> str:
    """意图路由：汽车问题去RAG，闲聊直接生成答案"""
    if state["intent"] == "car_question":
        return "rag"
    else:
        return "answer"

def route_by_review(state: AgentState) -> str:
    """审核路由：pass结束，retry重新检索"""
    if state["review"] == "pass":
        return "end"
    else:
        return "rag"

# ==================== 构建工作流 ====================
workflow = StateGraph(AgentState)

# 添加四个节点
workflow.add_node("intent", intent_node)
workflow.add_node("rag", rag_node)
workflow.add_node("answer", answer_node)
workflow.add_node("review", review_node)

# 设置入口：从意图识别开始
workflow.set_entry_point("intent")

# 意图识别后根据结果路由
workflow.add_conditional_edges(
    "intent",
    route_by_intent,
    {"rag": "rag", "answer": "answer"}
)

# RAG检索完去生成答案
workflow.add_edge("rag", "answer")

# 答案生成完去审核
workflow.add_edge("answer", "review")

# 审核后根据结果路由
workflow.add_conditional_edges(
    "review",
    route_by_review,
    {"end": END, "rag": "rag"}
)

# 编译，绑定记忆存储器
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
logger.info("工作流构建完成")

# ==================== 测试 ====================
# thread_id 标识用户，同一个id的对话历史会被记住
config = {"configurable": {"thread_id": "user_001"}}

# 第一轮：问续航
print("\n--- 第一轮 ---")
result1 = app.invoke({
    "messages": [],
    "question": "问界M8的续航是多少",
    "intent": "",
    "context": "",
    "answer": "",
    "review": "",
    "retry_count": 0
}, config=config)
print(f"答案：{result1['answer']}")

# 第二轮：问充电，依赖上一轮的上下文
print("\n--- 第二轮 ---")
result2 = app.invoke({
    "messages": result1["messages"],
    "question": "那它的充电时间呢",
    "intent": "",
    "context": "",
    "answer": "",
    "review": "",
    "retry_count": 0
}, config=config)
print(f"答案：{result2['answer']}")