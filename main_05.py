# main_05_LCEL_UI.py - 챗봇기능
# -----------------------------------------------------------
# ChatPDF (Streamlit + LangChain + Chroma + LCEL)
# main_04_LCEL.py를 Streamlit UI로 변환한 버전
# - PDF 업로드 기능
# - LCEL 방식 RAG 체인 사용
# - 실시간 질의응답
# -----------------------------------------------------------
# 실행 방법: streamlit run main_05_LCEL_UI.py
# 사전 설치: pip install streamlit langchain langchain-openai langchain-chroma python-dotenv pypdf

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# LangChain 관련 import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# =========================
# 1. 페이지 설정
# =========================
st.set_page_config(
    page_title="ChatPDF - RAG with LCEL",
    layout="centered"
)

st.title("📄 ChatPDF - RAG with LCEL")
st.markdown("### PDF 문서를 업로드하고 질문해보세요!")
st.write("---")

# =========================
# 2. 환경변수 로드 및 확인
# =========================
# load_dotenv()
# api_key = os.getenv('OPENAI_API_KEY')

# if not api_key: # 예외처리
#     st.error("⚠️ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
#     st.stop()

# =========================
# 3. 세션 상태 초기화
# =========================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = "./db/chromadb_streamlit"
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# 4. PDF 문서 처리 함수
# =========================
@st.cache_data
def process_pdf(uploaded_file):
    """
    업로드된 PDF를 처리하여 문서 청크로 변환
    
    Args:
        uploaded_file: Streamlit file uploader로부터 받은 파일
    
    Returns:
        list: 분할된 문서 청크 리스트
    """
    # 4-1. 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # 4-2. PDF 로드
    loader = PyPDFLoader(tmp_path)
    pages = loader.load_and_split()
    
    # 4-3. 텍스트 청크 분할 (main_04_LCEL.py와 동일한 설정)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 청크 크기
        chunk_overlap=50,    # 청크 간 중첩
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    
    # 4-4. 임시 파일 삭제
    os.unlink(tmp_path)
    
    return texts

# =========================
# 5. 벡터스토어 생성 함수
# =========================
def create_vectorstore(documents, persist_dir):
    """
    문서들로부터 Chroma 벡터스토어 생성
    
    Args:
        documents: 임베딩할 문서 리스트
        persist_dir: 벡터스토어 저장 경로
    
    Returns:
        Chroma: 생성된 벡터스토어 객체
    """
    import chromadb
    
    # 5-1. ChromaDB 클라이언트 생성
    client = chromadb.PersistentClient(path=persist_dir)
    
    # 5-2. 기존 컬렉션이 있으면 삭제 (새로운 PDF 업로드 시 초기화)
    try:
        client.delete_collection("esg")
    except Exception:
        pass
    
    # 5-3. 임베딩 모델 초기화
    embeddings_model = OpenAIEmbeddings()
    
    # 5-4. 벡터스토어 생성
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings_model,
        collection_name="esg",
        client=client,
        persist_directory=persist_dir
    )
    
    return vectorstore

# =========================
# 6. 문서 포맷 함수
# =========================
def format_docs(docs):
    """
    검색된 문서들을 하나의 문자열로 합치는 함수
    
    Args:
        docs: retriever에서 반환된 문서 리스트
    
    Returns:
        str: 각 문서의 내용을 '\n\n'로 구분하여 합친 문자열
    """
    return "\n\n".join(doc.page_content for doc in docs)

# =========================
# 7. LCEL RAG 체인 구성 함수
# =========================
def create_rag_chain(vectorstore, model_name="gpt-4o-mini"):
    """
    LCEL 방식의 RAG 체인 생성
    
    Args:
        vectorstore: 검색에 사용할 벡터스토어
        model_name: 사용할 LLM 모델명
    
    Returns:
        LCEL RAG 체인 객체
    """
    # 7-1. LLM 설정 (main_04_LCEL.py와 동일)
    llm = ChatOpenAI(
        model_name=model_name,
        temperature=0
    )
    
    # 7-2. RAG 프롬프트 템플릿 생성
    template = """다음 문서를 참고하여 질문에 답변해주세요.

문서:
{context}

질문: {question}

답변:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 7-3. LCEL 체인 구성 (파이프 연산자 | 사용)
    rag_chain = (
        {
            # context: retriever로 문서 검색 → format_docs로 포맷팅
            "context": vectorstore.as_retriever(search_kwargs={"k": 3}) | format_docs,
            # question: 입력 질문을 그대로 전달
            "question": RunnablePassthrough()
        }
        | prompt                # 프롬프트 템플릿에 context와 question 삽입
        | llm                   # LLM에 프롬프트 전달하여 답변 생성
        | StrOutputParser()     # LLM 출력을 문자열로 파싱
    )
    
    return rag_chain

# =========================
# 8. PDF 업로드 UI
# =========================
st.sidebar.header("📁 PDF 파일 업로드")
uploaded_file = st.sidebar.file_uploader(
    "PDF 파일을 선택하세요",
    type=["pdf"],
    help="분석할 PDF 문서를 업로드하세요"
)

# PDF가 업로드되고, 이전에 처리한 파일과 다른 경우에만 재처리
if uploaded_file is not None:
    if st.session_state.processed_file != uploaded_file.name:
        with st.spinner("📚 PDF를 읽고 임베딩을 생성하는 중입니다..."):
            # 8-1. PDF 처리
            documents = process_pdf(uploaded_file)
            
            # 8-2. 벡터스토어 생성
            st.session_state.vectorstore = create_vectorstore(
                documents,
                st.session_state.persist_dir
            )
            
            # 8-3. 처리된 파일명 저장
            st.session_state.processed_file = uploaded_file.name
            
            # 8-4. 채팅 기록 초기화
            st.session_state.chat_history = []
            
        st.sidebar.success(f"✅ 문서 처리 완료!\n총 청크 수: {len(documents)}")
        st.sidebar.info(f"📄 {uploaded_file.name}")

# =========================
# 9. 질의응답 UI
# =========================
if uploaded_file is None:
    st.info("👈 왼쪽 사이드바에서 PDF 파일을 업로드해주세요.")
    st.stop()

if st.session_state.vectorstore is None:
    st.warning("문서가 아직 처리되지 않았습니다. 잠시만 기다려주세요.")
    st.stop()

# =========================
# 10. 채팅 인터페이스
# =========================
st.header("💬 PDF에게 질문하기")

# 10-1. 이전 채팅 기록 표시
for i, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# 10-2. 질문 입력
question = st.chat_input("질문을 입력하세요...")

if question:
    # 10-3. 사용자 질문 표시
    with st.chat_message("user"):
        st.write(question)
    
    # 10-4. RAG 체인으로 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("🤔 답변을 생성하는 중..."):
            # RAG 체인 생성 및 실행
            rag_chain = create_rag_chain(
                st.session_state.vectorstore,
                model_name="gpt-4o-mini"
            )
            answer = rag_chain.invoke(question)
            
            # 답변 표시
            st.write(answer)
    
    # 10-5. 채팅 기록에 추가
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer
    })

# =========================
# 11. 사이드바 추가 정보
# =========================

st.sidebar.write("---")
st.sidebar.header("⚙️ 설정")
st.sidebar.info(f"""
- **모델**: gpt-4o-mini
- **청크 크기**: 500
- **청크 중첩**: 50
- **검색 문서 수**: 3개
""")

# =========================
# 12. 검색된 문서 확인 (선택적)
# =========================
if st.sidebar.checkbox("🔍 검색된 원본 문서 보기", value=False):
    if question and st.session_state.vectorstore:
        st.write("---")
        st.subheader("📑 검색된 원본 문서")
        
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieved_docs = retriever.invoke(question)
        
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"📄 문서 {i}"):
                st.markdown(f"**내용 (일부):**")
                st.text(doc.page_content[:300] + "...")
                st.markdown(f"**메타데이터:**")
                st.json(doc.metadata)

