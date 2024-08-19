import streamlit as st
from streamlit_chat import message
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from streamlit_extras.add_vertical_space import add_vertical_space
import requests  # For making API calls to your database
import os

st.subheader("ChatLlama")

st.markdown("Built by [Ujjwal Khadka](https://bento.me/khadkaujjwal)")

with st.sidebar:
    st.title("Llama-3.1 Chatbot by Ujjwal Khadka")
    st.subheader("This app lets you chat with Llama 3.1 405B! [👉]")
    api_key = st.text_input("Enter your Fireworks API Key", type="password")
    add_vertical_space(2)
    st.markdown("""
    Want to learn how to build this? 
   
    Join [Medium](https://medium.com/@khadkaujjwal47)
    """)
    add_vertical_space(3)
    st.write("Reach out to me on [LinkedIn](https://www.linkedin.com/in/ujjwal-khadka-94393816b/)")

# Initialize session state variables
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Function to fetch data from your database
def fetch_data_from_database(query):
    # Example API endpoint; replace with your actual endpoint
    endpoint = "https://noveltytechnology.com/api/data"
    response = requests.get(endpoint, params={"query": query})
    if response.status_code == 200:
        return response.json()  # Assuming the response is in JSON format
    else:
        return {"error": "Could not fetch data"}

# Only initialize ChatOpenAI and ConversationChain if API key is provided
if api_key:
    if st.session_state.conversation is None:
        llm = ChatOpenAI(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",
            openai_api_key=api_key,
            openai_api_base="https://api.fireworks.ai/inference/v1"
        )
        st.session_state.conversation = ConversationChain(
            memory=st.session_state.buffer_memory, 
            llm=llm
        )

    # Rest of your chat interface code
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Fetch data from the database based on the user's prompt
        db_response = fetch_data_from_database(prompt)

        # Process the database response if needed
        if "error" not in db_response:
            db_content = db_response.get("data", "No relevant data found.")
            prompt = f"{prompt}\n\nContext from database: {db_content}"

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation.predict(input=prompt)
                    st.write(response)
                    message = {"role": "assistant", "content": response}
                    st.session_state.messages.append(message)
else:
    st.warning("Please enter your Fireworks API Key in the sidebar to start the chat.")
