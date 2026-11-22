import json
import os
from typing import TypedDict
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, START, END
from langgraph.errors import GraphRecursionError

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

# ===== 工具的真实实现（代码负责执行） =====
# 方案C：检索方式从"关键词精确匹配"升级为"字符级 TF-IDF 相似度"
# 生产级方案 = embedding 模型 + 向量数据库；本方案检索管线一致，向量化的实现不同
FOOTBALL_DB = {
    "世界杯冠军": "2022年世界杯冠军是阿根廷（决赛点球战胜法国），2018年是法国。",
    "金球奖": "2023年金球奖得主是梅西（第8次），2022年是本泽马。",
    "欧冠冠军": "2023-24赛季欧冠冠军是皇家马德里（第15次夺冠）。",
}

_keys = list(FOOTBALL_DB.keys())
_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
_matrix = _vectorizer.fit_transform(_keys)          # 启动时把知识库向量化一次

SIM_THRESHOLD = 0.15                                # 相似度低于它 = 检索失败

def lookup_db(query: str) -> str:
    """工具函数：语义检索知识库，取相似度最高的一条；低于阈值视为查不到"""
    sims = cosine_similarity(_vectorizer.transform([query]), _matrix)[0]
    best = int(sims.argmax())
    if sims[best] < SIM_THRESHOLD:
        return "资料库里没找到"
    return FOOTBALL_DB[_keys[best]]

# ===== 方案B：Function Calling 的工具清单（给模型看的说明书） =====
TOOLS = [{
    "type": "function",
    "function": {
        "name": "lookup_football",
        "description": "查询足球精确数据（世界杯冠军、金球奖得主、欧冠冠军等）。常识性问题不需要调用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要查询的数据关键词，如：2022世界杯冠军"},
            },
            "required": ["query"],
        },
    },
}]

# ===== 方案A：State 新增 history（多轮记忆） =====
class State(TypedDict):
    question: str
    answer: str
    tool_result: str
    tool_call: dict
    history: list

SYSTEM = (
    "你是一位足球知识专家，回答不超过 5 句话。"
    "涉及精确数据时用工具查询；工具查不到就直说不知道，绝不编造。"
)

# ===== agent 节点：思考——直接答，还是发起工具调用 =====
def agent_node(state: State) -> dict:
    messages = [{"role": "system", "content": SYSTEM}] + state["history"]
    content = state["question"]
    if state["tool_result"]:                     # 工具刚返回结果 → 观察后再生成
        content += f"\n\n[工具查询结果] {state['tool_result']}"
    messages.append({"role": "user", "content": content})

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=TOOLS,                             # 方案B：把工具清单交给模型
    )
    msg = resp.choices[0].message

    if msg.tool_calls:                           # 模型返回的是"调用指令"
        call = msg.tool_calls[0]
        args = json.loads(call.function.arguments)
        return {"tool_call": {"name": call.function.name, "query": args["query"]}}
    return {"answer": msg.content, "tool_call": {}}   # 模型直接给出最终回答

# ===== tool 节点：执行——真正干活的是代码 =====
def tool_node(state: State) -> dict:
    result = lookup_db(state["tool_call"]["query"])
    return {"tool_result": f"查到的资料：{result}", "tool_call": {}}

# ===== 条件边：函数管判断，图管跳转 =====
def route(state: State) -> str:
    if state["answer"]:
        return "finish"
    if state["tool_call"]:
        return "call_tool"
    return "finish"

def build_app():
    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tool", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"call_tool": "tool", "finish": END})
    graph.add_edge("tool", "agent")              # 真循环：查完资料回来继续思考
    return graph.compile()

app = build_app()

if __name__ == "__main__":
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("错误：请先设置环境变量 DEEPSEEK_API_KEY")
        raise SystemExit(1)
    print("足球知识专家启动（多轮记忆 + Function Calling），输入 exit 退出。")
    history = []                                 # 方案A：跨轮次的历史
    while True:
        q = input("你: ")
        if q.strip().lower() in ("exit", "quit", "退出"):
            break
        try:
            result = app.invoke(
                {"question": q, "answer": "", "tool_result": "", "tool_call": {}, "history": history},
                config={"recursion_limit": 10},  # 防死循环：最多转 10 步
            )
            print(f"助手: {result['answer']}\n")
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": result["answer"]})
            if len(history) > 20:                # 记忆裁剪：只留最近 10 轮，控制 token 开销
                history = history[-20:]
        except GraphRecursionError:
            print("助手: 循环步数超限，已强制终止本轮。\n")
        except Exception as e:
            print(f"本轮出错，已跳过：{e}\n")
