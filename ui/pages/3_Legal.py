import asyncio
import streamlit as st
from agno.tools.streamlit.components import check_password

from ui.css import CUSTOM_CSS
from ui.utils import about_agno, footer, get_agent_response

st.set_page_config(
    page_title="Legal Assistant",
    page_icon=":balance_scale:",
    layout="wide",
)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


async def header():
    st.markdown("<h1 class='heading'>Legal Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subheading'>A specialized legal assistant focusing on Italian law and legal documents.</p>",
        unsafe_allow_html=True,
    )


async def chat_interface():
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about Italian law..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            response = await get_agent_response(
                prompt,
                agent_id="legal",
                stream=True,
            )
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


async def main():
    await header()
    await chat_interface()
    await footer()
    await about_agno()


if __name__ == "__main__":
    if check_password():
        asyncio.run(main()) 