# import os
# import chromadb
# from typing import TypedDict, List

# # Pydantic 및 LangChain 호환성을 위한 임포트
# from pydantic import PrivateAttr

# # LangChain 및 관련 라이브러리 임포트
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_community.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langgraph.graph import StateGraph, END
# from dotenv import load_dotenv

# # FlashRank 임포트
# try:
#     from flashrank import Ranker, RerankRequest
#     from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
#     from langchain_core.callbacks.manager import Callbacks
# except ImportError:
#     print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. 'pip install flashrank'를 실행해주세요.")
#     BaseDocumentCompressor = object # 임시 정의
#     Ranker = None

# # ValidationError 해결을 위해 PrivateAttr 사용
# class FlashRankRerank(BaseDocumentCompressor):
#     """FlashRank를 사용한 LangChain 문서 재정렬기"""
    
#     _ranker: Ranker = PrivateAttr()
#     top_n: int = 3

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._ranker = Ranker()

#     class Config:
#         arbitrary_types_allowed = True

#     def compress_documents(
#         self,
#         documents: List[Document],
#         query: str,
#         callbacks: Callbacks | None = None,
#     ) -> List[Document]:
#         if not documents:
#             return []
        
#         rerank_request = RerankRequest(
#             query=query,
#             passages=[{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
#         )
#         reranked_results = self._ranker.rerank(rerank_request)
        
#         final_docs = []
#         for item in reranked_results[:self.top_n]:
#             original_doc_index = item['id']
#             doc = documents[original_doc_index]
#             doc.metadata["relevance_score"] = item['score']
#             final_docs.append(doc)
#         return final_docs

# # --- 0. 초기 설정: API 키 로딩 ---
# load_dotenv()
# if not os.getenv("OPENAI_API_KEY"):
#     raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다.")

# # --- 1. 데이터 준비: 외부 파일에서 템플릿 로딩 ---
# # --- [수정된 부분] 파일 형식에 맞는 두 가지 로드 함수 정의 ---
# def load_line_by_line(file_path: str) -> List[str]:
#     """한 줄에 한 항목씩 있는 파일을 로드합니다."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             items = [line.strip() for line in f if line.strip()]
#         print(f"✅ {len(items)}개의 항목을 '{file_path}'에서 로드했습니다.")
#         return items
#     except FileNotFoundError:
#         print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
#         return []

# def load_by_separator(file_path: str, separator: str = '---') -> List[str]:
#     """구분자로 분리된 항목들을 로드합니다."""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#             items = [section.strip() for section in content.split(separator) if section.strip()]
#         print(f"✅ {len(items)}개의 항목을 '{file_path}'에서 로드했습니다.")
#         return items
#     except FileNotFoundError:
#         print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
#         return []

# # 각 파일에 맞는 로드 함수 호출
# APPROVED_TEMPLATES = load_line_by_line("./data/approved_templates.txt")
# REJECTED_TEMPLATES_TEXT = load_by_separator("./data/rejected_templates.txt")

# # 전역 변수로 Retriever를 저장하여 중복 로딩 방지
# retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected = None, None, None, None

# def setup_retrievers():
#     global retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected
#     if all([retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected]):
#         print("Retriever가 이미 로드되었습니다.")
#         return

#     print("Retriever 설정 시작...")
    
#     from chromadb.config import Settings

#     docs_compliance = TextLoader("./data/compliance_rules.txt", encoding='utf-8').load()
#     docs_generation = TextLoader("./data/generation_rules.txt", encoding='utf-8').load()
    
#     docs_whitelist = [Document(page_content=t) for t in APPROVED_TEMPLATES]
#     docs_rejected = [Document(page_content=t) for t in REJECTED_TEMPLATES_TEXT]

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     split_compliance = text_splitter.split_documents(docs_compliance)
#     split_generation = text_splitter.split_documents(docs_generation)
#     # 반려 템플릿은 의미 단위로 분리되지 않도록 원본 그대로 사용
#     split_rejected = docs_rejected

#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     vector_db_path = "./vector_db"
    
#     client_settings = Settings(anonymized_telemetry=False)
    
#     db_compliance = Chroma.from_documents(split_compliance, embeddings, collection_name="compliance_rules", persist_directory=vector_db_path, client_settings=client_settings)
#     db_generation = Chroma.from_documents(split_generation, embeddings, collection_name="generation_rules", persist_directory=vector_db_path, client_settings=client_settings)
#     db_whitelist = Chroma.from_documents(docs_whitelist, embeddings, collection_name="whitelist_templates", persist_directory=vector_db_path, client_settings=client_settings)
#     db_rejected = Chroma.from_documents(split_rejected, embeddings, collection_name="rejected_templates", persist_directory=vector_db_path, client_settings=client_settings)

#     def create_hybrid_retriever(vectorstore, docs):
#         if not docs:
#             return vectorstore.as_retriever(search_kwargs={"k": 5})

#         vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#         keyword_retriever = BM25Retriever.from_documents(docs)
#         keyword_retriever.k = 5

#         ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
        
#         if Ranker:
#             compressor = FlashRankRerank(top_n=3)
#             return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
#         return ensemble_retriever

#     retriever_compliance = create_hybrid_retriever(db_compliance, split_compliance)
#     retriever_generation = create_hybrid_retriever(db_generation, split_generation)
#     retriever_whitelist = create_hybrid_retriever(db_whitelist, docs_whitelist)
#     retriever_rejected = create_hybrid_retriever(db_rejected, split_rejected)
    
#     print("✅ Retriever 설정 완료!")

# # --- 2. LangGraph 파이프라인 정의 ---
# class GraphState(TypedDict):
#     original_request: str
#     hypothetical_template: str
#     template_draft: str
#     retrieved_examples: str
#     retrieved_rules: str
#     retrieved_rejections: str
#     validation_result: str
#     correction_attempts: int

# llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# hyde_prompt = ChatPromptTemplate.from_template(
#     """당신은 사용자의 요청을 완벽하게 이해하고 그에 맞는 이상적인 알림톡 템플릿을 상상해내는 전문가입니다.
#     오직 요청에 부합하는 완벽한 템플릿 텍스트만 생성하고, 다른 어떤 설명도 덧붙이지 마세요.

#     사용자 요청: "{request}"

#     가상 템플릿:
#     """
# )

# generation_prompt = ChatPromptTemplate.from_template(
#     """당신은 창의적이고 실수를 통해 배우는 알림톡 템플릿 전문가입니다.

#     # 지시사항
#     1. '사용자 원본 요청'을 바탕으로 새로운 정보성 알림톡 템플릿을 작성하세요.
#     2. **(중요) '과거 반려 사례'를 반드시 확인하여 동일한 실수를 반복하지 마세요.**
#     3. '참고용 성공 사례'의 스타일을 따르되, 절대 똑같이 만들지 마세요.
#     4. '필수 준수 규칙'을 지키고, #{{변수}} 사용법을 적절히 활용하세요.
#     5. 만약 사용자 요청이 광고성이라면, 정보성으로 발송 불가하다고 안내하고 '친구톡' 사용을 권장하세요.

#     # 사용자 원본 요청
#     {original_request}

#     # 과거 반려 사례 (오답 노트)
#     {rejections}

#     # 참고용 성공 사례
#     {examples}

#     # 필수 준수 규칙
#     {rules}

#     # 전문가 답변 (위 지시사항을 모두 반영하여 작성):
#     """
# )

# step_back_prompt = ChatPromptTemplate.from_template(
#     """당신은 문제의 본질을 파악하는 분석가입니다.
#     주어진 '템플릿 초안'을 보고, 이 템플릿이 정보통신망법이나 내부 규정과 관련하여 어떤 핵심적인 질문을 던지는지 한 문장으로 요약하세요.

#     # 템플릿 초안
#     {draft}

#     # 핵심 질문:
#     """
# )

# validation_prompt = ChatPromptTemplate.from_template(
#     """당신은 과거 판례까지 참고하는 매우 꼼꼼한 심사관입니다.
    
#     # 검수 대상 템플릿
#     {draft}

#     # 관련 규정
#     {rules}

#     # 유사한 과거 반려 사례 (판례)
#     {rejections}

#     # 지시사항
#     1. '검수 대상 템플릿'이 '유사한 과거 반려 사례'와 비슷한 문제를 가지고 있는지 먼저 확인하세요.
#     2. 그 다음 '관련 규정'을 종합적으로 검토하여 최종 위반 여부를 판단하세요.
#     3. 위반 사항이 없다면 "accept" 라고만 답변하세요.
#     4. 위반 사항이 있다면, 어떤 규정 또는 과거 사례를 근거로 위반인지 구체적으로 설명해주세요.
#     """
# )

# def hyde_generation(state: GraphState):
#     print("\n--- 1. HyDE 가상 템플릿 생성 시작 ---")
#     request = state['original_request']
#     chain = hyde_prompt | llm | StrOutputParser()
#     hypothetical_template = chain.invoke({"request": request})
#     state['hypothetical_template'] = hypothetical_template
#     print("✅ HyDE 생성 완료.")
#     return state

# def generate_draft(state: GraphState):
#     print("\n--- 2. 템플릿 초안 생성 시작 ---")
#     original_request = state['original_request']
#     query = state['hypothetical_template']
    
#     example_docs = retriever_whitelist.invoke(query)
#     state['retrieved_examples'] = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(example_docs)])
    
#     rule_docs = retriever_generation.invoke(query)
#     state['retrieved_rules'] = "\n\n".join([doc.page_content for doc in rule_docs])
    
#     rejected_docs = retriever_rejected.invoke(query)
#     state['retrieved_rejections'] = "\n\n".join([doc.page_content for doc in rejected_docs])
    
#     chain = generation_prompt | llm | StrOutputParser()
#     draft = chain.invoke({
#         "original_request": original_request, 
#         "rejections": state['retrieved_rejections'],
#         "examples": state['retrieved_examples'], 
#         "rules": state['retrieved_rules']
#     })
#     state['template_draft'] = draft
#     state['correction_attempts'] = 0
#     print("✅ 초안 생성 완료.")
#     print(f"생성된 초안:\n---\n{draft}\n---")
#     return state

# def validate_draft(state: GraphState):
#     print("\n--- 3. 규정 준수 검증 시작 ---")
#     draft = state['template_draft']
    
#     if "광고성" in draft or "친구톡" in draft:
#         state['validation_result'] = "accept"
#         print("✅ 검증 결과: 통과 (생성 단계에서 광고성으로 판단하여 안내함)")
#         return state

#     print("   - 3a. Step-Back 핵심 쟁점 생성 중...")
#     step_back_chain = step_back_prompt | llm | StrOutputParser()
#     step_back_question = step_back_chain.invoke({"draft": draft})
#     print(f"   - 생성된 핵심 쟁점: {step_back_question}")

#     print("   - 3b. 관련 규정 및 반려 사례 검색 중...")
#     compliance_docs = retriever_compliance.invoke(step_back_question)
#     rules = "\n\n".join([doc.page_content for doc in compliance_docs])
    
#     rejected_docs = retriever_rejected.invoke(draft)
#     rejections = "\n\n".join([doc.page_content for doc in rejected_docs])
    
#     print("   - 3c. 최종 검증 수행 중...")
#     chain = validation_prompt | llm | StrOutputParser()
#     result = chain.invoke({"draft": draft, "rules": rules, "rejections": rejections})
#     state['validation_result'] = result
    
#     if "accept" in result.lower():
#         print("✅ 검증 결과: 통과")
#     else:
#         print(f"🚨 검증 결과: 위반 발견 (시도 {state['correction_attempts'] + 1}회)")
#         print(f"위반 내용: {result}")
#     return state

# def self_correct(state: GraphState):
#     print("\n--- 4. 자가 수정 시작 ---")
    
#     correction_request = (
#         f"이전 템플릿은 다음의 이유로 반려되었습니다: '{state['validation_result']}'. "
#         f"이 문제를 해결하고, 원래 요청인 '{state['original_request']}'을 만족하는 새로운 정보성 템플릿을 생성해주세요."
#     )
    
#     chain = generation_prompt | llm | StrOutputParser()
#     new_draft = chain.invoke({
#         "original_request": correction_request,
#         "rejections": state['retrieved_rejections'],
#         "examples": state['retrieved_examples'],
#         "rules": state['retrieved_rules']
#     })
    
#     state['template_draft'] = new_draft
#     state['correction_attempts'] += 1
#     print("✅ 자가 수정 완료.")
#     return state

# def decide_next_step(state: GraphState):
#     if "accept" in state['validation_result'].lower():
#         return "end"
#     elif state['correction_attempts'] >= 1:
#         return "end_with_failure"
#     else:
#         return "self_correct"

# def build_graph():
#     workflow = StateGraph(GraphState)
    
#     workflow.add_node("hyde_generation", hyde_generation)
#     workflow.add_node("generate_draft", generate_draft)
#     workflow.add_node("validate_draft", validate_draft)
#     workflow.add_node("self_correct", self_correct)
    
#     workflow.set_entry_point("hyde_generation")
    
#     workflow.add_edge("hyde_generation", "generate_draft")
#     workflow.add_edge("generate_draft", "validate_draft")
#     workflow.add_conditional_edges(
#         "validate_draft",
#         decide_next_step,
#         {"self_correct": "self_correct", "end": END, "end_with_failure": END}
#     )
#     workflow.add_edge("self_correct", "validate_draft")
    
#     return workflow.compile()

# # --- 3. 메인 실행 로직 ---
# if __name__ == "__main__":
#     setup_retrievers()
#     app = build_graph()

#     user_request = "안녕하세요 #{회원명}님, 고용노동부 플랫폼 이용 지원에 대해 안내드립니다."
    
#     print(f"\n==================================================")
#     print(f"사용자 요청: {user_request}")
#     print(f"==================================================")
    
#     final_state = app.invoke({"original_request": user_request})

#     print(f"\n================ 최종 결과 ================")
#     if final_state and "accept" in final_state.get('validation_result', '').lower():
#         print("🎉 최종 생성된 답변:")
#         print(final_state['template_draft'])
#     elif final_state:
#         print("🔥 템플릿 생성 실패. 마지막 시도 결과:")
#         print(f"실패 사유: {final_state.get('validation_result', 'N/A')}")
#         print(f"마지막 초안:\n{final_state.get('template_draft', 'N/A')}")
#     else:
#         print("오류: 파이프라인이 최종 상태를 반환하지 못했습니다.")
#     print("============================================")

import os
import chromadb
from typing import TypedDict, List

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import PrivateAttr

# LangChain 및 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
    from langchain_core.callbacks.manager import Callbacks
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. 'pip install flashrank'를 실행해주세요.")
    BaseDocumentCompressor = object # 임시 정의
    Ranker = None

# ValidationError 해결을 위해 PrivateAttr 사용
class FlashRankRerank(BaseDocumentCompressor):
    """FlashRank를 사용한 LangChain 문서 재정렬기"""
    
    _ranker: Ranker = PrivateAttr()
    top_n: int = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ranker = Ranker()

    class Config:
        arbitrary_types_allowed = True

    def compress_documents(
        self,
        documents: List[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> List[Document]:
        if not documents:
            return []
        
        rerank_request = RerankRequest(
            query=query,
            passages=[{"id": i, "text": doc.page_content} for i, doc in enumerate(documents)]
        )
        reranked_results = self._ranker.rerank(rerank_request)
        
        final_docs = []
        for item in reranked_results[:self.top_n]:
            original_doc_index = item['id']
            doc = documents[original_doc_index]
            doc.metadata["relevance_score"] = item['score']
            final_docs.append(doc)
        return final_docs

# --- 0. 초기 설정: API 키 로딩 ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다.")

# --- 1. 데이터 준비: 외부 파일에서 템플릿 로딩 ---
def load_line_by_line(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            items = [line.strip() for line in f if line.strip()]
        print(f"✅ {len(items)}개의 항목을 '{file_path}'에서 로드했습니다.")
        return items
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
        return []

def load_by_separator(file_path: str, separator: str = '---') -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            items = [section.strip() for section in content.split(separator) if section.strip()]
        print(f"✅ {len(items)}개의 항목을 '{file_path}'에서 로드했습니다.")
        return items
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
        return []

APPROVED_TEMPLATES = load_line_by_line("./data/approved_templates.txt")
REJECTED_TEMPLATES_TEXT = load_by_separator("./data/rejected_templates.txt")

# 전역 변수로 Retriever를 저장하여 중복 로딩 방지
retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected = None, None, None, None

def setup_retrievers():
    global retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected
    if all([retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected]):
        print("Retriever가 이미 로드되었습니다.")
        return

    print("Retriever 설정 시작...")
    
    from chromadb.config import Settings

    docs_compliance = TextLoader("./data/compliance_rules.txt", encoding='utf-8').load()
    docs_generation = TextLoader("./data/generation_rules.txt", encoding='utf-8').load()
    
    docs_whitelist = [Document(page_content=t) for t in APPROVED_TEMPLATES]
    docs_rejected = [Document(page_content=t) for t in REJECTED_TEMPLATES_TEXT]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_compliance = text_splitter.split_documents(docs_compliance)
    split_generation = text_splitter.split_documents(docs_generation)
    split_rejected = docs_rejected

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db_path = "./vector_db"
    
    client_settings = Settings(anonymized_telemetry=False)
    
    db_compliance = Chroma.from_documents(split_compliance, embeddings, collection_name="compliance_rules", persist_directory=vector_db_path, client_settings=client_settings)
    db_generation = Chroma.from_documents(split_generation, embeddings, collection_name="generation_rules", persist_directory=vector_db_path, client_settings=client_settings)
    db_whitelist = Chroma.from_documents(docs_whitelist, embeddings, collection_name="whitelist_templates", persist_directory=vector_db_path, client_settings=client_settings)
    db_rejected = Chroma.from_documents(split_rejected, embeddings, collection_name="rejected_templates", persist_directory=vector_db_path, client_settings=client_settings)

    def create_hybrid_retriever(vectorstore, docs):
        if not docs:
            return vectorstore.as_retriever(search_kwargs={"k": 5})

        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k = 5

        ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
        
        if Ranker:
            compressor = FlashRankRerank(top_n=3)
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        return ensemble_retriever

    retriever_compliance = create_hybrid_retriever(db_compliance, split_compliance)
    retriever_generation = create_hybrid_retriever(db_generation, split_generation)
    retriever_whitelist = create_hybrid_retriever(db_whitelist, docs_whitelist)
    retriever_rejected = create_hybrid_retriever(db_rejected, split_rejected)
    
    print("✅ Retriever 설정 완료!")

# --- 2. LangGraph 파이프라인 정의 ---
class GraphState(TypedDict):
    original_request: str
    request_type: str
    hypothetical_template: str
    template_draft: str
    retrieved_examples: str
    retrieved_rules: str
    retrieved_rejections: str
    validation_result: str
    correction_attempts: int

llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# --- [수정된 부분] routing_prompt의 중괄호 이스케이프 처리 ---
routing_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 분석하는 라우터입니다.
    주어진 요청이 '짧은 의도'인지 '긴 초안'인지 판단하세요.

    - '짧은 의도': 한두 문장으로 된, 아이디어나 주제에 가까운 요청. (예: "주문 완료 메시지 만들어줘")
    - '긴 초안': 여러 줄로 구성되어 있으며, #{{변수}} 등을 포함하여 이미 완성된 템플릿 형태에 가까운 요청.

    오직 'intent' 또는 'draft' 둘 중 하나로만 답변하세요.

    사용자 요청:
    {request}
    """
)

hyde_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 완벽하게 이해하고 그에 맞는 이상적인 알림톡 템플릿을 상상해내는 전문가입니다.
    오직 요청에 부합하는 완벽한 템플릿 텍스트만 생성하고, 다른 어떤 설명도 덧붙이지 마세요.

    사용자 요청: "{request}"

    가상 템플릿:
    """
)

generation_prompt = ChatPromptTemplate.from_template(
    """당신은 창의적이고 실수를 통해 배우는 알림톡 템플릿 전문가입니다.

    # 지시사항
    1. '사용자 원본 요청'을 바탕으로 새로운 정보성 알림톡 템플릿을 작성하세요.
    2. **(중요) '과거 반려 사례'를 반드시 확인하여 동일한 실수를 반복하지 마세요.**
    3. '참고용 성공 사례'의 스타일을 따르되, 절대 똑같이 만들지 마세요.
    4. '필수 준수 규칙'을 지키고, #{{변수}} 사용법을 적절히 활용하세요.
    5. 만약 사용자 요청이 광고성이라면, 정보성으로 발송 불가하다고 안내하고 '친구톡' 사용을 권장하세요.

    # 사용자 원본 요청
    {original_request}

    # 과거 반려 사례 (오답 노트)
    {rejections}

    # 참고용 성공 사례
    {examples}

    # 필수 준수 규칙
    {rules}

    # 전문가 답변 (위 지시사항을 모두 반영하여 작성):
    """
)

step_back_prompt = ChatPromptTemplate.from_template(
    """당신은 문제의 본질을 파악하는 분석가입니다.
    주어진 '템플릿 초안'을 보고, 이 템플릿이 정보통신망법이나 내부 규정과 관련하여 어떤 핵심적인 질문을 던지는지 한 문장으로 요약하세요.

    # 템플릿 초안
    {draft}

    # 핵심 질문:
    """
)

validation_prompt = ChatPromptTemplate.from_template(
    """당신은 과거 판례까지 참고하는 매우 꼼꼼한 심사관입니다.
    
    # 검수 대상 템플릿
    {draft}

    # 관련 규정
    {rules}

    # 유사한 과거 반려 사례 (판례)
    {rejections}

    # 지시사항
    1. '검수 대상 템플릿'이 '유사한 과거 반려 사례'와 비슷한 문제를 가지고 있는지 먼저 확인하세요.
    2. 그 다음 '관련 규정'을 종합적으로 검토하여 최종 위반 여부를 판단하세요.
    3. 위반 사항이 없다면 "accept" 라고만 답변하세요.
    4. 위반 사항이 있다면, 어떤 규정 또는 과거 사례를 근거로 위반인지 구체적으로 설명해주세요.
    """
)

def route_request(state: GraphState):
    print("\n--- 1. 요청 유형 분석 (라우터) ---")
    request = state['original_request']
    
    chain = routing_prompt | llm | StrOutputParser()
    request_type = chain.invoke({"request": request})
    
    if "draft" in request_type.lower():
        state['request_type'] = "draft"
        print("✅ 요청 유형: 긴 초안 (검증/수정 경로로 진행)")
        state['template_draft'] = request
    else:
        state['request_type'] = "intent"
        print("✅ 요청 유형: 짧은 의도 (창작 경로로 진행)")
    
    return state

def hyde_generation(state: GraphState):
    print("\n--- 2a. HyDE 가상 템플릿 생성 시작 (창작 경로) ---")
    request = state['original_request']
    chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_template = chain.invoke({"request": request})
    state['hypothetical_template'] = hypothetical_template
    print("✅ HyDE 생성 완료.")
    return state

def generate_draft(state: GraphState):
    print("\n--- 2b. 템플릿 초안 생성 시작 (창작 경로) ---")
    original_request = state['original_request']
    query = state['hypothetical_template']
    
    example_docs = retriever_whitelist.invoke(query)
    state['retrieved_examples'] = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(example_docs)])
    
    rule_docs = retriever_generation.invoke(query)
    state['retrieved_rules'] = "\n\n".join([doc.page_content for doc in rule_docs])
    
    rejected_docs = retriever_rejected.invoke(query)
    state['retrieved_rejections'] = "\n\n".join([doc.page_content for doc in rejected_docs])
    
    chain = generation_prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "original_request": original_request, 
        "rejections": state['retrieved_rejections'],
        "examples": state['retrieved_examples'], 
        "rules": state['retrieved_rules']
    })
    state['template_draft'] = draft
    state['correction_attempts'] = 0
    print("✅ 초안 생성 완료.")
    print(f"생성된 초안:\n---\n{draft}\n---")
    return state

def validate_draft(state: GraphState):
    print("\n--- 3. 규정 준수 검증 시작 ---")
    draft = state['template_draft']
    
    if "광고성" in draft or "친구톡" in draft:
        state['validation_result'] = "accept"
        print("✅ 검증 결과: 통과 (생성 단계에서 광고성으로 판단하여 안내함)")
        return state

    print("   - 3a. Step-Back 핵심 쟁점 생성 중...")
    step_back_chain = step_back_prompt | llm | StrOutputParser()
    step_back_question = step_back_chain.invoke({"draft": draft})
    print(f"   - 생성된 핵심 쟁점: {step_back_question}")

    print("   - 3b. 관련 규정 및 반려 사례 검색 중...")
    compliance_docs = retriever_compliance.invoke(step_back_question)
    rules = "\n\n".join([doc.page_content for doc in compliance_docs])
    
    rejected_docs = retriever_rejected.invoke(draft)
    rejections = "\n\n".join([doc.page_content for doc in rejected_docs])
    
    print("   - 3c. 최종 검증 수행 중...")
    chain = validation_prompt | llm | StrOutputParser()
    result = chain.invoke({"draft": draft, "rules": rules, "rejections": rejections})
    state['validation_result'] = result
    
    if "accept" in result.lower():
        print("✅ 검증 결과: 통과")
    else:
        print(f"🚨 검증 결과: 위반 발견 (시도 {state['correction_attempts'] + 1}회)")
        print(f"위반 내용: {result}")
    return state

def self_correct(state: GraphState):
    print("\n--- 4. 자가 수정 시작 ---")
    
    correction_request = (
        f"이전 템플릿은 다음의 이유로 반려되었습니다: '{state['validation_result']}'. "
        f"이 문제를 해결하고, 원래 요청인 '{state['original_request']}'을 만족하는 새로운 정보성 템플릿을 생성해주세요."
    )
    
    chain = generation_prompt | llm | StrOutputParser()
    new_draft = chain.invoke({
        "original_request": correction_request,
        "rejections": state['retrieved_rejections'],
        "examples": state['retrieved_examples'],
        "rules": state['retrieved_rules']
    })
    
    state['template_draft'] = new_draft
    state['correction_attempts'] += 1
    print("✅ 자가 수정 완료.")
    return state

def decide_creation_path(state: GraphState):
    if state['request_type'] == "draft":
        return "validate_draft"
    else:
        return "hyde_generation"

def decide_next_step(state: GraphState):
    if "accept" in state['validation_result'].lower():
        return "end"
    elif state['correction_attempts'] >= 1:
        return "end_with_failure"
    else:
        return "self_correct"

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("router", route_request)
    workflow.add_node("hyde_generation", hyde_generation)
    workflow.add_node("generate_draft", generate_draft)
    workflow.add_node("validate_draft", validate_draft)
    workflow.add_node("self_correct", self_correct)
    
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        decide_creation_path,
        {
            "hyde_generation": "hyde_generation",
            "validate_draft": "validate_draft"
        }
    )
    
    workflow.add_edge("hyde_generation", "generate_draft")
    workflow.add_edge("generate_draft", "validate_draft")
    
    workflow.add_conditional_edges(
        "validate_draft",
        decide_next_step,
        {"self_correct": "self_correct", "end": END, "end_with_failure": END}
    )
    workflow.add_edge("self_correct", "validate_draft")
    
    return workflow.compile()

# --- 3. 메인 실행 로직 ---
if __name__ == "__main__":
    setup_retrievers()
    app = build_graph()

    requests_to_test = {
        "짧은 의도 (창작 경로 테스트)": "안녕하세요 #{회원명}님, 고용노동부 플랫폼 이용 지원에 대해 안내드립니다.",
        "긴 초안 (검증 경로 테스트)": """안녕하세요 #{회원명}님, 고용노동부 플랫폼 이용 지원에 대한 안내드립니다.

        ▶ 지원 대상: #{지원대상}
        ▶ 지원 내용: #{지원내용}
        ▶ 신청 기간: #{신청기간}

        자세한 내용은 고용노동부 홈페이지에서 확인하실 수 있습니다. 문의사항이 있으시면 언제든지 연락 주시기 바랍니다. 감사합니다."""
    }
    
    for test_name, user_request in requests_to_test.items():
        print(f"\n==================================================")
        print(f"테스트 시작: {test_name}")
        print(f"==================================================")
        print(f"사용자 요청:\n---\n{user_request}\n---")
        
        final_state = app.invoke({"original_request": user_request})

        print(f"\n================ 최종 결과: {test_name} ================")
        if final_state and "accept" in final_state.get('validation_result', '').lower():
            print("🎉 최종 생성된 답변:")
            print(final_state['template_draft'])
        elif final_state:
            print("🔥 템플릿 생성 실패. 마지막 시도 결과:")
            print(f"실패 사유: {final_state.get('validation_result', 'N/A')}")
            print(f"마지막 초안:\n{final_state.get('template_draft', 'N/A')}")
        else:
            print("오류: 파이프라인이 최종 상태를 반환하지 못했습니다.")
        print("============================================")
