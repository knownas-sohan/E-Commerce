import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

def build_chain(vstore):
    model = ChatGroq(
        groq_api_key=os.getenv("GROQ_API"),
        model="deepseek-r1-distill-llama-70b",
        temperature=0.5
    )

    retriever_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", retriever_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """Your ecommercebot bot is an expert in product recommendations and customer queries.
It analyzes product titles and reviews to provide accurate and helpful responses.
Ensure your answers are relevant to the product context and refrain from straying off-topic.
Your responses should be concise and informative.\n\nCONTEXT:\n{context}\n\nQUESTION: {input}\n\nYOUR ANSWER:"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
