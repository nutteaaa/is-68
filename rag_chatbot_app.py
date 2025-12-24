"""
RAG Chatbot Application with Streamlit UI
Support for OpenSearch vector store and OpenAI GPT models
"""

import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chain' not in st.session_state:
    st.session_state.chain = None


@st.cache_resource
def initialize_opensearch():
    """Initialize OpenSearch connection"""
    try:
        # AWS credentials for OpenSearch
        region = os.getenv('AWS_REGION', 'us-east-1')
        service = 'es'
        
        credentials = boto3.Session().get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            service,
            session_token=credentials.token
        )
        
        # OpenSearch client
        opensearch_client = OpenSearch(
            hosts=[{
                'host': os.getenv('OPENSEARCH_HOST'),
                'port': int(os.getenv('OPENSEARCH_PORT', 443))
            }],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        
        return opensearch_client
    except Exception as e:
        st.error(f"Failed to initialize OpenSearch: {str(e)}")
        return None


@st.cache_resource
def initialize_vectorstore(_opensearch_client):
    """Initialize vector store with embeddings"""
    try:
        embeddings = OpenAIEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        vectorstore = OpenSearchVectorSearch(
            opensearch_url=f"https://{os.getenv('OPENSEARCH_HOST')}:{os.getenv('OPENSEARCH_PORT', 443)}",
            index_name=os.getenv('OPENSEARCH_INDEX', 'rag_documents'),
            embedding_function=embeddings,
            http_auth=_opensearch_client.transport.get_connection().session.auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Failed to initialize vector store: {str(e)}")
        return None


def initialize_chain(vectorstore, model_name, temperature, k_documents):
    """Initialize conversational retrieval chain"""
    try:
        # LLM
        llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        # Custom prompt template
        prompt_template = """คุณเป็นผู้ช่วยที่ให้ข้อมูลที่ถูกต้องและเป็นประโยชน์ โดยใช้เอกสารที่ให้มาเป็นหลัก

Context จากเอกสาร:
{context}

ประวัติการสนทนา:
{chat_history}

คำถาม: {question}

คำแนะนำในการตอบ:
1. ตอบโดยอ้างอิงจากเอกสารที่ให้มาเป็นหลัก
2. ถ้าไม่มีข้อมูลในเอกสาร ให้บอกว่าไม่มีข้อมูล
3. ตอบเป็นภาษาไทยที่เข้าใจง่าย
4. ให้คำตอบที่กระชับและตรงประเด็น

คำตอบ:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Retrieval chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": k_documents}
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            verbose=True
        )
        
        return chain
    except Exception as e:
        st.error(f"Failed to initialize chain: {str(e)}")
        return None


# Sidebar configuration
with st.sidebar:
    st.title("⚙️ การตั้งค่า")
    
    st.subheader("🤖 Model Settings")
    model_name = st.selectbox(
        "เลือก Model",
        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="ค่าต่ำ = คำตอบที่แน่นอน, ค่าสูง = คำตอบที่หลากหลาย"
    )
    
    k_documents = st.slider(
        "จำนวนเอกสารอ้างอิง",
        min_value=1,
        max_value=10,
        value=3,
        help="จำนวนเอกสารที่จะดึงมาใช้ในการตอบคำถาม"
    )
    
    st.divider()
    
    # Initialize system
    if st.button("🔄 เริ่มต้นระบบ", type="primary"):
        with st.spinner("กำลังเริ่มต้นระบบ..."):
            opensearch_client = initialize_opensearch()
            if opensearch_client:
                vectorstore = initialize_vectorstore(opensearch_client)
                if vectorstore:
                    st.session_state.chain = initialize_chain(
                        vectorstore, 
                        model_name, 
                        temperature, 
                        k_documents
                    )
                    if st.session_state.chain:
                        st.success("✅ ระบบพร้อมใช้งาน!")
                    else:
                        st.error("❌ ไม่สามารถสร้าง Chain ได้")
                else:
                    st.error("❌ ไม่สามารถเชื่อมต่อ Vector Store ได้")
            else:
                st.error("❌ ไม่สามารถเชื่อมต่อ OpenSearch ได้")
    
    st.divider()
    
    # Clear chat history
    if st.button("🗑️ ล้างประวัติการสนทนา"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        if st.session_state.chain:
            st.session_state.chain.memory.clear()
        st.success("ล้างประวัติเรียบร้อย!")
        st.rerun()
    
    st.divider()
    
    # System status
    st.subheader("📊 สถานะระบบ")
    if st.session_state.chain:
        st.success("🟢 พร้อมใช้งาน")
    else:
        st.warning("🟡 กรุณาเริ่มต้นระบบ")
    
    st.info(f"💬 ข้อความในประวัติ: {len(st.session_state.messages)}")


# Main chat interface
st.title("🤖 RAG Chatbot")
st.caption("Powered by LangChain + OpenAI + OpenSearch")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display source documents if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**แหล่งที่ {i}:**")
                    st.text(source[:300] + "..." if len(source) > 300 else source)
                    st.divider()

# Chat input
if prompt := st.chat_input("พิมพ์คำถามของคุณที่นี่..."):
    # Check if chain is initialized
    if not st.session_state.chain:
        st.error("⚠️ กรุณาเริ่มต้นระบบก่อนใช้งาน (กดปุ่มที่ Sidebar)")
        st.stop()
    
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("กำลังคิด..."):
            try:
                # Query the chain
                response = st.session_state.chain({
                    "question": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                answer = response["answer"]
                source_docs = response.get("source_documents", [])
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                if source_docs:
                    with st.expander("📚 แหล่งข้อมูลอ้างอิง"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**แหล่งที่ {i}:**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            if hasattr(doc, 'metadata') and doc.metadata:
                                st.json(doc.metadata)
                            st.divider()
                
                # Save to session state
                sources = [doc.page_content for doc in source_docs]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Update chat history
                st.session_state.chat_history.append((prompt, answer))
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาด: {str(e)}")
                st.error("กรุณาลองใหม่อีกครั้งหรือเริ่มต้นระบบใหม่")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
