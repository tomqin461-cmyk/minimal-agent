import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"


def backend_is_available() -> bool:
    try:
        response = requests.get(HEALTH_URL, timeout=2)
        return response.status_code == 200 and response.json() == {
            "status": "ok",
        }
    except requests.RequestException:
        return False


st.set_page_config(
    page_title="能源 RAG Agent",
    page_icon="E",
)

st.session_state.setdefault("messages", [])

with st.sidebar:
    st.subheader("服务状态")

    if backend_is_available():
        st.success("后端服务正常")
    else:
        st.error("后端服务未连接")

    st.divider()

    if st.button("清空本次对话"):
        st.session_state.messages = []

    st.caption("本次对话只保存在当前浏览器页面中。")

st.title("能源 RAG Agent")
st.caption("基于本地知识库、向量检索和 Reranker 的学习助手")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            st.caption(
                "检索到的资料：" + "、".join(message["sources"])
            )

question = st.chat_input(
    "输入天然气、四川盆地或能源相关问题",
    submit_mode="disable",
)

if question:
    history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in st.session_state.messages
    ]
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并生成回答..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "question": question,
                        "history": history[-8:],
                    },
                    timeout=180,
                )
                response.raise_for_status()

                result = response.json()
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    st.caption(
                        "检索到的资料：" + "、".join(sources)
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except requests.RequestException as error:
                error_message = f"无法连接后端服务：{error}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )