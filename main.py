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
def load_templates_from_file(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            templates = [line.strip() for line in f if line.strip()]
        print(f"✅ {len(templates)}개의 템플릿을 '{file_path}'에서 로드했습니다.")
        return templates
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 목록으로 시작합니다.")
        return []

APPROVED_TEMPLATES = load_templates_from_file("./data/approved_templates.txt")

retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected = None, None, None, None

def setup_retrievers():
    global retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected
    if retriever_compliance and retriever_generation and retriever_whitelist and retriever_rejected:
        print("Retriever가 이미 로드되었습니다.")
        return

    print("Retriever 설정 시작...")
    
    from chromadb.config import Settings

    docs_compliance = TextLoader("./data/compliance_rules.txt", encoding='utf-8').load()
    docs_generation = TextLoader("./data/generation_rules.txt", encoding='utf-8').load()
    docs_whitelist = [Document(page_content=t) for t in APPROVED_TEMPLATES]
    docs_rejected = TextLoader("./data/rejected_templates.txt", encoding='utf-8').load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_compliance = text_splitter.split_documents(docs_compliance)
    split_generation = text_splitter.split_documents(docs_generation)
    split_rejected = text_splitter.split_documents(docs_rejected)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db_path = "./vector_db"
    
    client_settings = Settings(anonymized_telemetry=False)
    
    db_compliance = Chroma.from_documents(split_compliance, embeddings, collection_name="compliance_rules", persist_directory=vector_db_path, client_settings=client_settings)
    db_generation = Chroma.from_documents(split_generation, embeddings, collection_name="generation_rules", persist_directory=vector_db_path, client_settings=client_settings)
    db_whitelist = Chroma.from_documents(docs_whitelist, embeddings, collection_name="whitelist_templates", persist_directory=vector_db_path, client_settings=client_settings)
    db_rejected = Chroma.from_documents(split_rejected, embeddings, collection_name="rejected_templates", persist_directory=vector_db_path, client_settings=client_settings)

    def create_hybrid_retriever(vectorstore, docs):
        if not docs:
            return vectorstore.as_retriever(search_kwargs={"k": 7})

        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k = 7

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
    expanded_queries: List[str] # [수정] HyDE 필드를 쿼리 확장 필드로 변경
    template_draft: str
    retrieved_examples: str
    retrieved_rules: str
    retrieved_rejected_examples: str
    validation_result: str
    correction_attempts: int

llm = ChatOpenAI(model="gpt-4o", temperature=0.4)

# --- [추가] 쿼리 확장을 위한 프롬프트 ---
expansion_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 분석하여, 관련 정보를 검색하는 데 가장 효과적인 여러 개의 검색 질문을 생성하는 검색 전문가입니다.
    
    사용자의 원본 요청을 바탕으로, 다양한 관점에서 최소 3개의 구체적인 검색 질문을 생성해주세요.
    각 질문은 줄바꿈으로 구분해주세요. 다른 설명은 필요 없습니다.

    # 원본 요청:
    {original_request}

    # 확장된 검색 질문:
    """
)

generation_prompt = ChatPromptTemplate.from_template(
    """당신은 창의적이고 규정을 잘 아는 알림톡 템플릿 전문가입니다. 당신의 목표는 정보성 알림톡 템플릿을 만드는 것입니다.

    # 지시사항
    1. 먼저 '사용자 원본 요청'이 정보성인지 광고성인지 판단하세요.
    2. **[정보성 요청의 경우]**
        - 요청을 만족하는 새로운 정보성 알림톡 템플릿을 작성하세요.
        - '참고용 승인 템플릿'과 유사한 스타일을 따르되, 절대 똑같이 만들지 마세요.
        - '참고용 실패 사례'를 보고 동일한 실수를 절대 반복하지 마세요.
        - '필수 준수 규칙'을 반드시 지키고, #{{변수}} 사용법을 적절히 활용하세요.
    3. **[광고성 요청의 경우]**
        - 해당 요청은 광고성 내용(예: 할인, 쿠폰, 이벤트)을 포함하여 정보성 알림톡으로 발송이 불가하다고 명확히 설명하세요.
        - 사용자가 '친구톡'과 같은 광고성 메시지 채널을 사용해야 한다고 안내하세요.

    # 사용자 원본 요청
    {original_request}

    # 참고용 승인 템플릿 (정보성 요청일 경우에만 참고)
    {examples}

    # 참고용 실패 사례 (이런 실수를 반복하지 마세요)
    {rejected_examples}

    # 필수 준수 규칙 (정보성 템플릿 생성 시 참고)
    {rules}

    # 전문가 답변 (위 지시사항을 모두 반영하여 작성):
    """
)

validation_prompt = ChatPromptTemplate.from_template(
    """당신은 정보통신망법 및 내부 규정을 검수하는 매우 꼼꼼한 심사관입니다.
    
    # 검수 대상 템플릿
    {draft}

    # 관련 규정
    {rules}

    # 유사한 반려 사례 (참고용)
    {rejected_examples}

    # 지시사항
    1. '검수 대상 템플릿'이 **정보성인지 광고성인지 판단**하세요.
    2. **[판단 결과가 정보성일 경우]**
        - '관련 규정'과 '유사한 반려 사례'를 참고하여 위반 사항이 없는지 검토하세요.
        - 위반 사항이 없다면 "accept" 라고만 답변하세요.
        - 위반 사항이 있다면, 구체적으로 설명해주세요.
    3. **[판단 결과가 광고성일 경우]**
        - "광고성 메시지로 판단됨." 이라고 밝히고, 그 이유를 설명해주세요.
    """
)

# --- [추가] 쿼리 확장 노드 ---
def query_expansion(state: GraphState):
    print("\n--- 1. 쿼리 확장 시작 ---")
    original_request = state['original_request']
    
    chain = expansion_prompt | llm | StrOutputParser()
    result = chain.invoke({"original_request": original_request})
    
    queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
    state['expanded_queries'] = queries
    
    print("✅ 쿼리 확장 완료.")
    print(f"확장된 쿼리:\n---\n{queries}\n---")
    return state

# --- [수정] generate_draft가 다중 쿼리를 사용하도록 수정 ---
def generate_draft(state: GraphState):
    print("\n--- 2. 다중 쿼리 기반 정보 검색 및 초안 생성 ---")
    original_request = state['original_request']
    queries = state['expanded_queries']
    
    # 모든 쿼리에 대해 검색을 수행하고 결과를 합침
    all_example_docs, all_rule_docs, all_rejected_docs = [], [], []
    for q in queries:
        all_example_docs.extend(retriever_whitelist.invoke(q))
        all_rule_docs.extend(retriever_generation.invoke(q))
        all_rejected_docs.extend(retriever_rejected.invoke(q))

    # 중복된 문서 제거 (페이지 내용 기준)
    unique_examples = list({doc.page_content: doc for doc in all_example_docs}.values())
    unique_rules = list({doc.page_content: doc for doc in all_rule_docs}.values())
    unique_rejected = list({doc.page_content: doc for doc in all_rejected_docs}.values())

    # 프롬프트에 전달할 형태로 변환
    examples = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(unique_examples)])
    rules = "\n\n".join([doc.page_content for doc in unique_rules])
    rejected_examples = "\n\n".join([f"실패 사례 {i+1}:\n{doc.page_content}" for i, doc in enumerate(unique_rejected)])

    # 이후 단계에서 재사용할 수 있도록 상태에 저장
    state['retrieved_examples'] = examples
    state['retrieved_rules'] = rules
    state['retrieved_rejected_examples'] = rejected_examples
    
    chain = generation_prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "original_request": original_request, 
        "examples": examples, 
        "rejected_examples": rejected_examples,
        "rules": rules
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

    compliance_docs = retriever_compliance.invoke(draft)
    rules = "\n\n".join([doc.page_content for doc in compliance_docs])

    rejected_docs = retriever_rejected.invoke(draft)
    rejected_examples = "\n\n".join([f"유사 실패 사례 {i+1}:\n{doc.page_content}" for i, doc in enumerate(rejected_docs)])
    
    chain = validation_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "draft": draft, 
        "rules": rules,
        "rejected_examples": rejected_examples
    })
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
        "examples": state['retrieved_examples'],
        "rejected_examples": state['retrieved_rejected_examples'],
        "rules": state['retrieved_rules']
    })
    
    state['template_draft'] = new_draft
    state['correction_attempts'] += 1
    print("✅ 자가 수정 완료.")
    print(f"수정된 초안:\n---\n{new_draft}\n---")
    return state

def decide_next_step(state: GraphState):
    if "accept" in state['validation_result'].lower():
        return "end"
    elif state['correction_attempts'] >= 1:
        return "end_with_failure"
    else:
        return "self_correct"

# --- [수정] 그래프 빌드 로직 수정 ---
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("query_expansion", query_expansion)
    workflow.add_node("generate_draft", generate_draft)
    workflow.add_node("validate_draft", validate_draft)
    workflow.add_node("self_correct", self_correct)
    
    workflow.set_entry_point("query_expansion")
    
    workflow.add_edge("query_expansion", "generate_draft")
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

    # 테스트할 사용자 요청
    user_request = "안녕하세요 #{회원명}님, 고용노동부 플랫폼 이용 지원에 대해 안내드립니다. 고객센터 번호는 02-1111-1111입니다."
    
    print(f"\n==================================================")
    print(f"사용자 요청: {user_request}")
    print(f"==================================================")
    
    final_state = app.invoke({"original_request": user_request})

    print(f"\n================ 최종 결과 ================")
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