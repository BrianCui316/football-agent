import os
import streamlit as st
from langgraph.errors import GraphRecursionError

# Streamlit Cloud 的密钥放在 st.secrets；本地用环境变量
try:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
except Exception:
    pass  # 本地运行时走已有的环境变量

# 必须先设好 key 再 import agent（agent 在导入时就创建 client）
from agent import build_app

st.set_page_config(page_title="Football Agent", page_icon="⚽")
st.title("⚽ Football Agent — 足球知识专家")
st.caption("LangGraph + Function Calling + 多轮记忆 · 问它足球问题试试")

@st.cache_resource
def load_app():
    return build_app()

app = load_app()

if "history" not in st.session_state:
    st.session_state.history = []

# 渲染历史消息
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

q = st.chat_input("问我足球问题…")
if q:
    st.session_state.history.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        with st.spinner("思考中…"):
            try:
                result = app.invoke(
                    {"question": q, "answer": "", "tool_result": "", "tool_call": {},
                     "history": st.session_state.history[:-1]},  # 不含刚这条 user 消息
                    config={"recursion_limit": 10},
                )
                reply = result["answer"]
            except GraphRecursionError:
                reply = "循环步数超限，本轮终止。"
            except Exception as e:
                reply = f"出错了：{e}"
        st.write(reply)
    st.session_state.history.append({"role": "assistant", "content": reply})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
