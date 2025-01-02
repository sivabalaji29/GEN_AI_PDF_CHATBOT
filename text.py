
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


os.environ["AZURE_OPENAI_API_KEY"] = "OPEN_AI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "AZURE_ENDPOINT"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"] = "gstgpt35t"


embeddings = AzureOpenAIEmbeddings(
    azure_deployment="gsttextemb002",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="AZURE_ENDPOINT",
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)

vectorstore = FAISS.load_local("./db", embeddings, allow_dangerous_deserialization=True)

# Initialize LLM for Chat
llm = AzureChatOpenAI(
    azure_deployment="gstgpt35t",
    openai_api_version="2023-12-01-preview",
    azure_endpoint="AZURE_ENDPOINT",
    api_key=os.environ["AZURE_OPENAI_API_KEY"]
)


system_prompt = (
    "You are an assistant that provides accurate and complete answers based on the provided context. "
    "Return the full answer extracted from the document along with the page number where it was found. "
    "Ensure that the answer is coherent and addresses the user's query in a meaningful way."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def get_chain():

    retriever = vectorstore.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


st.title("Pdf Chatbot")


user_query = st.text_input("Ask a question:")

if user_query:
    
    rag_chain = get_chain()
    response = rag_chain.invoke({"input": user_query})

    
    if response and response["context"]:
        answer_text = response["context"][0].page_content.strip() 
        page_number = response["context"][0].metadata.get("page", "N/A") 

        
        st.write("Answer:", answer_text)
        st.write("Found on page:", page_number)
    else:
        st.write("No results found.")


