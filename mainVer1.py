import os
import json
import re
from typing import TypedDict, List, Optional

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import BaseModel, Field, PrivateAttr

# LangChain 및 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.base import BaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# FlashRank 임포트
try:
    from flashrank import Ranker, RerankRequest
    from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
    from langchain_core.callbacks.manager import Callbacks
except ImportError:
    print("FlashRank 또는 관련 모듈을 찾을 수 없습니다. 'pip install flashrank'를 실행해주세요.")
    BaseDocumentCompressor = object
    Ranker = None

# --- 상수 정의 ---
MAX_CORRECTION_ATTEMPTS = 3 # AI 자가 수정 최대 시도 횟수

class CustomRuleLoader(BaseLoader):
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        self.file_path = file_path
        self.encoding = encoding
    def load(self) -> List[Document]:
        docs = []
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f: content = f.read()
        except FileNotFoundError:
            print(f"🚨 경고: '{self.file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
            return []
        rule_blocks = re.findall(r'\[규칙 시작\](.*?)\[규칙 끝\]', content, re.DOTALL)
        for block in rule_blocks:
            lines, metadata, page_content, is_content_section = block.strip().split('\n'), {}, "", False
            for line in lines:
                if line.lower().startswith('content:'):
                    is_content_section = True
                    page_content += line[len('content:'):].strip() + "\n"
                    continue
                if is_content_section: page_content += line.strip() + "\n"
                else:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
            if page_content: docs.append(Document(page_content=page_content.strip(), metadata=metadata))
        print(f"✅ {len(docs)}개의 규칙을 '{self.file_path}'에서 구조화하여 로드했습니다.")
        return docs

class TemplateAnalysisResult(BaseModel):
    status: str = Field(description="템플릿의 최종 상태 (예: 'accepted', 'rejected')")
    reason: str = Field(description="다단계 추론 과정을 포함한 상세한 판단 이유.")
    evidence: Optional[str] = Field(None, description="판단의 근거가 된 규칙들의 rule_id 목록 (쉼표로 구분).")
    suggestion: Optional[str] = Field(None, description="개선을 위한 구체적인 제안.")
    revised_template: Optional[str] = Field(None, description="최종적으로 생성되거나 수정된 템플릿 텍스트. 'rejected' 상태일 경우 null이어야 함.")

class FlashRankRerank(BaseDocumentCompressor):
    _ranker: Ranker = PrivateAttr()
    top_n: int = 3
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ranker = Ranker()
    class Config: arbitrary_types_allowed = True
    def compress_documents(self, documents: List[Document], query: str, callbacks: Callbacks | None = None) -> List[Document]:
        if not documents: return []
        rerank_request = RerankRequest(query=query, passages=[{"id": i, "text": doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)])
        reranked_results = self._ranker.rerank(rerank_request)
        final_docs = []
        for item in reranked_results[:self.top_n]:
            doc = documents[item['id']]
            doc.metadata["relevance_score"] = item['score']
            final_docs.append(doc)
        return final_docs

load_dotenv()
if not os.getenv("OPENAI_API_KEY"): raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다.")

def load_line_by_line(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f: items = [line.strip() for line in f if line.strip()]
        print(f"✅ {len(items)}개의 항목을 '{file_path}'에서 로드했습니다.")
        return items
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
        return []

def load_by_separator(file_path: str, separator: str = '---') -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        items = [section.strip() for section in content.split(separator) if section.strip()]
        print(f"✅ {len(items)}개의 항목을 '{file_path}'에서 로드했습니다.")
        return items
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다. 빈 리스트로 시작합니다.")
        return []

APPROVED_TEMPLATES = load_line_by_line("./data/approved_templates.txt")
REJECTED_TEMPLATES_TEXT = load_by_separator("./data/rejected_templates.txt")

retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected = None, None, None, None

def setup_retrievers():
    global retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected
    if all([retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected]): return
    print("Retriever 설정 시작...")
    from chromadb.config import Settings
    docs_compliance = CustomRuleLoader("./data/compliance_rules.txt").load()
    docs_generation = CustomRuleLoader("./data/generation_rules.txt").load()
    docs_whitelist = [Document(page_content=t) for t in APPROVED_TEMPLATES]
    docs_rejected = [Document(page_content=t) for t in REJECTED_TEMPLATES_TEXT]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db_path = "./vector_db"
    client_settings = Settings(anonymized_telemetry=False)
    
    def create_db(name, docs):
        if docs: return Chroma.from_documents(docs, embeddings, collection_name=name, persist_directory=vector_db_path, client_settings=client_settings)
        return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=vector_db_path, client_settings=client_settings)

    db_compliance, db_generation, db_whitelist, db_rejected = create_db("compliance_rules", docs_compliance), create_db("generation_rules", docs_generation), create_db("whitelist_templates", docs_whitelist), create_db("rejected_templates", docs_rejected)
    
    def create_hybrid_retriever(vectorstore, docs):
        if not docs: return vectorstore.as_retriever(search_kwargs={"k": 5})
        vector_retriever, keyword_retriever = vectorstore.as_retriever(search_kwargs={"k": 5}), BM25Retriever.from_documents(docs)
        keyword_retriever.k = 5
        ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])
        if Ranker:
            compressor = FlashRankRerank(top_n=3)
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)
        return ensemble_retriever
        
    retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected = create_hybrid_retriever(db_compliance, docs_compliance), create_hybrid_retriever(db_generation, docs_generation), create_hybrid_retriever(db_whitelist, docs_whitelist), create_hybrid_retriever(db_rejected, docs_rejected)
    print("✅ Retriever 설정 완료!")

class GraphState(TypedDict):
    original_request: str
    user_choice: str
    selected_style: str
    template_draft: str
    validation_result: Optional[TemplateAnalysisResult]
    correction_attempts: int

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

expansion_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 핵심 의도와 '선택된 스타일'을 바탕으로, 정보가 풍부한 알림톡 템플릿 초안을 확장하는 전문가입니다.
    당신의 유일한 임무는 아래 지시사항에 따라 **정보가 확장된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.
    # 지시사항
    1. '사용자 핵심 의도'를 바탕으로, '선택된 스타일'에 맞는 완전한 템플릿 초안을 만드세요.
    2. '유사한 성공 사례'를 참고하여, 어떤 정보(예: 지원 대상, 신청 기간 등)를 추가해야 할지 **추론**하고, 적절한 #{{변수}}를 사용하세요.
    # 사용자 핵심 의도: {original_request}
    # 선택된 스타일: {style}
    # 유사한 성공 사례 (참고용): {examples}
    # 확장된 템플릿 초안 (오직 템플릿 텍스트만 출력):"""
)

validation_prompt_structured = ChatPromptTemplate.from_template(
    """당신은 과거 판례와 법규를 근거로 판단하는 매우 꼼꼼한 최종 심사관입니다.
    주어진 JSON 형식에 맞춰서만 답변해야 합니다.

    # 검수 대상 템플릿: {draft}
    # 관련 규정 (메타데이터 포함): {rules}
    # 유사한 과거 반려 사례 (판례): {rejections}

    # 지시사항
    1. **(매우 중요) 'reason' 필드를 다음과 같은 다단계 추론 과정에 따라 상세하게 작성하세요:**
        a. **사실 확인:** 먼저, '검수 대상 템플릿'에 어떤 내용(예: 쿠폰, 할인율, 사용처 등)이 포함되어 있는지 객관적으로 서술하세요.
        b. **규정 연결:** 다음으로, 확인된 사실과 가장 관련성이 높은 '관련 규정' 또는 '유사한 과거 반려 사례'를 1~3개 찾아 연결하세요. 이때, 각 근거의 `rule_id`와 내용을 반드시 인용하세요.
        c. **최종 결론:** 마지막으로, 위 사실과 규정을 종합하여 왜 이 템플릿이 'accepted' 또는 'rejected'인지 명확한 결론을 내리세요.
    2. **'evidence' 필드에는 당신이 인용한 규칙들의 `rule_id`를 쉼표로 구분하여 나열하세요.** (예: "COMP-DEF-002, COMP-CASE-003")
    3. 위반 사항이 없다면 'status'를 'accepted'로 설정하고, 'revised_template'에 원본 초안을 그대로 넣으세요.
    4. 위반 사항이 있다면 'status'를 'rejected'로 설정하고, 'suggestion'에 구체적인 개선 방안을 제시하세요.
    5. 만약 광고성 내용(할인, 쿠폰, 이벤트 등)이 문제라면, 'suggestion'에 '친구톡' 사용을 권장하는 내용을 포함하세요.

    # 출력 형식 (JSON):
    {format_instructions}
    """
)

correction_prompt_template = """당신은 지적된 문제점을 해결하여 더 나은 대안을 제시하는 전문 카피라이터입니다.
당신의 유일한 임무는 아래 지시사항에 따라 **수정된 템플릿 초안 하나만**을 생성하는 것입니다. 초안 외에 다른 설명은 절대로 덧붙이지 마세요.

# 원래 사용자 요청: {original_request}
# 이전에 제안했던 템플릿 (반려됨): {rejected_draft}
# 반려 사유 및 개선 제안: {rejection_reason}

# 지시사항
1. '반려 사유 및 개선 제안'을 완벽하게 이해하고, 지적된 모든 문제점을 해결하세요.
2. '원래 사용자 요청'의 핵심 의도는 유지해야 합니다.
{dynamic_instruction}

# 수정된 템플릿 초안 (오직 템플릿 텍스트만 출력):
"""

def recommend_fast_track(state: GraphState):
    print("\n--- 1. 유사 템플릿 추천 (Fast-Track) ---")
    request = state['original_request']
    similar_docs = retriever_whitelist.invoke(request)
    if not similar_docs:
        print("✅ 유사한 기존 템플릿을 찾지 못했습니다. 바로 신규 생성 프로세스를 시작합니다.")
        state['user_choice'] = 'new_template'
        return state
    print("💡 요청하신 내용과 가장 유사한 기존 템플릿 3개를 찾았습니다.")
    for i, doc in enumerate(similar_docs):
        print("-" * 20 + f"\n  추천 템플릿 {i+1}:\n{doc.page_content}\n" + "-" * 20)
    while True:
        choice = input(f"\n이 중에서 사용하실 템플릿 번호를 입력하시거나, 신규 생성을 원하시면 '4'를 입력해주세요 (1, 2, 3, 4): ")
        if choice in ['1', '2', '3']:
            state['user_choice'], state['template_draft'] = 'fast_track', similar_docs[int(choice)-1].page_content
            print(f"✅ {choice}번 템플릿을 선택했습니다. 선택된 템플릿으로 검증을 시작합니다.")
            return state
        elif choice == '4':
            state['user_choice'] = 'new_template'
            print("✅ 신규 템플릿 생성을 선택했습니다.")
            return state
        else: print("🚨 잘못된 입력입니다. 1, 2, 3, 4 중 하나를 입력해주세요.")

def select_style(state: GraphState):
    print("\n--- 2. 신규 템플릿 스타일 선택 ---")
    print("새로운 템플릿을 생성합니다. 원하시는 스타일을 선택해주세요.\n1. 기본형\n2. 이미지형\n3. 아이템 리스트형")
    style_map = {'1': '기본형', '2': '이미지형', '3': '아이템 리스트형'}
    while True:
        choice = input("\n원하는 스타일의 번호를 입력해주세요 (1, 2, 3): ")
        if choice in style_map:
            state['selected_style'] = style_map[choice]
            print(f"✅ '{state['selected_style']}' 스타일을 선택했습니다.")
            return state
        else: print("🚨 잘못된 입력입니다. 1, 2, 3 중 하나를 입력해주세요.")

def expand_intent(state: GraphState):
    print("\n--- 3. 의도 확장 및 초안 생성 ---")
    original_request, style = state['original_request'], state['selected_style']
    example_docs = retriever_whitelist.invoke(original_request)
    examples = "\n\n".join([f"예시 {i+1}:\n{doc.page_content}" for i, doc in enumerate(example_docs)])
    expansion_chain = expansion_prompt | llm | StrOutputParser()
    expanded_draft = expansion_chain.invoke({"original_request": original_request, "style": style, "examples": examples})
    state['template_draft'] = expanded_draft
    print(f"✅ 의도 확장 완료. 생성된 초안:\n---\n{expanded_draft}\n---")
    return state

def validate_draft(state: GraphState):
    print(f"\n--- 4. 규정 준수 검증 (AI 시도: {state.get('correction_attempts', 0) + 1}) ---")
    draft = state['template_draft']
    parser = JsonOutputParser(pydantic_object=TemplateAnalysisResult)
    step_back_chain = ChatPromptTemplate.from_template("이 템플릿의 핵심 쟁점은 무엇인가?: {draft}") | llm | StrOutputParser()
    step_back_question = step_back_chain.invoke({"draft": draft})
    print(f"   - 생성된 핵심 쟁점: {step_back_question}")
    
    compliance_docs = retriever_compliance.invoke(step_back_question)
    rules_with_metadata = "\n\n".join([f"문서 메타데이터: {doc.metadata}\n문서 내용: {doc.page_content}" for doc in compliance_docs])
    
    rejected_docs = retriever_rejected.invoke(draft)
    rejections = "\n\n".join([doc.page_content for doc in rejected_docs])
    
    validation_chain = validation_prompt_structured | llm | parser
    result = validation_chain.invoke({"draft": draft, "rules": rules_with_metadata, "rejections": rejections, "format_instructions": parser.get_format_instructions()})
    
    state['validation_result'] = TemplateAnalysisResult.model_validate(result)
    if state['validation_result'].status == 'accepted':
        print("✅ 검증 결과: 통과")
    else:
        print(f"🚨 검증 결과: 위반 발견")
        print(f"   - 상세 이유:\n{state['validation_result'].reason}")
        print(f"   - 개선 제안: {state['validation_result'].suggestion}")
    return state

def self_correct_draft(state: GraphState):
    print("\n--- 5. AI 자가 수정 시작 ---")
    
    attempts = state.get('correction_attempts', 0)
    
    if attempts == 0:
        instruction = "3. 광고성 문구를 제거하거나, 정보성 내용으로 순화하는 등, 제안된 방향에 맞게 템플릿을 수정하세요."
    elif attempts == 1:
        instruction = "3. **(2차 수정)** 아직도 문제가 있습니다. 이번에는 '쿠폰', '할인', '이벤트', '특가'와 같은 명백한 광고성 단어를 사용하지 마세요. 대신 '고객님께 적용 가능한 혜택', '새로운 소식'과 같은 정보성 표현으로 순화하여 다시 작성해보세요."
    else:
        instruction = """3. **(최종 수정: 관점 전환)** 여전히 광고성으로 보입니다. 이것이 마지막 시도입니다.
        - **관점 전환:** 메시지의 주체를 '우리(사업자)'에서 '고객님'으로 완전히 바꾸세요.
        - **목적 변경:** '판매'나 '방문 유도'가 아니라, '고객님이 과거에 동의한 내용에 따라 고객님의 권리(혜택) 정보를 안내'하는 것으로 목적을 재정의하세요.
        - **근거 제시:** 메시지 하단에 '※ 본 메시지는 OOO 정보 수신에 동의하신 고객님께만 발송됩니다.'와 같이, 이 메시지가 스팸이 아닌 근거를 명확히 포함시키세요.
        - **표현 순화:** '할인율', '행사' 같은 직접적인 단어를 '우대 혜택', '적용 가능' 등으로 최대한 순화하세요."""

    base_prompt = ChatPromptTemplate.from_template(correction_prompt_template)
    correction_prompt = base_prompt.partial(dynamic_instruction=instruction)
    
    correction_chain = correction_prompt | llm | StrOutputParser()
    
    new_draft = correction_chain.invoke({
        "original_request": state['original_request'],
        "rejected_draft": state['template_draft'],
        "rejection_reason": state['validation_result'].reason + "\n개선 제안: " + state['validation_result'].suggestion
    })
    
    state['template_draft'] = new_draft
    state['correction_attempts'] = attempts + 1
    
    print(f"✅ AI 자가 수정 완료. 수정된 초안:\n---\n{new_draft}\n---")
    return state

# --- [신규 추가] 인간-AI 협업 노드 ---
def human_in_the_loop(state: GraphState):
    print("\n--- 6. AI 최종 실패: 사용자 수정 단계 ---")
    print("🔥 AI가 모든 수정을 시도했지만, 최종적으로 규정 준수에 실패했습니다.")
    print("\n마지막으로 반려된 초안은 다음과 같습니다:")
    print("-" * 20)
    print(state['template_draft'])
    print("-" * 20)
    print("\n반려 사유:")
    print(state['validation_result'].reason)
    print("\n개선 제안:")
    print(state['validation_result'].suggestion)
    
    while True:
        choice = input("\n직접 수정하여 마지막 검증을 시도하시겠습니까? (y/n): ").lower()
        if choice == 'y':
            print("\n템플릿을 직접 수정해주세요. (수정을 마치려면 Enter 키를 두 번 누르세요)")
            user_input_lines = []
            while True:
                line = input()
                if not line:
                    break
                user_input_lines.append(line)
            
            user_edited_draft = "\n".join(user_input_lines)
            if not user_edited_draft.strip():
                print("🚨 입력된 내용이 없습니다. 수정을 취소합니다.")
                state['user_choice'] = 'exit'
                return state

            state['template_draft'] = user_edited_draft
            state['correction_attempts'] = 99 # 사용자가 수정했음을 나타내는 플래그
            print("\n✅ 사용자 수정안이 접수되었습니다. 마지막 최종 검증을 시작합니다.")
            return state
        elif choice == 'n':
            print("✅ 수정을 포기하셨습니다. 프로세스를 종료합니다.")
            state['user_choice'] = 'exit'
            return state
        else:
            print("🚨 잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")

def decide_fast_track_path(state: GraphState):
    return "validate_draft" if state['user_choice'] == 'fast_track' else "select_style"

def decide_next_step(state: GraphState):
    if state['validation_result'].status == 'accepted':
        return "end"
    elif state.get('correction_attempts', 0) < MAX_CORRECTION_ATTEMPTS:
        return "self_correct"
    else:
        # AI의 모든 시도가 실패하면, human_in_the_loop로 이동
        return "human_in_the_loop"

# --- [신규 추가] 사용자 수정 후의 라우터 ---
def decide_after_human_edit(state: GraphState):
    if state.get('user_choice') == 'exit':
        return "end_with_failure"
    else:
        # 사용자가 수정한 템플릿을 검증하러 감
        return "validate_draft"

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("recommend_fast_track", recommend_fast_track)
    workflow.add_node("select_style", select_style)
    workflow.add_node("expand_intent", expand_intent)
    workflow.add_node("validate_draft", validate_draft)
    workflow.add_node("self_correct_draft", self_correct_draft)
    workflow.add_node("human_in_the_loop", human_in_the_loop) # 신규 노드 추가
    
    workflow.set_entry_point("recommend_fast_track")
    workflow.add_conditional_edges("recommend_fast_track", decide_fast_track_path, {"select_style": "select_style", "validate_draft": "validate_draft"})
    workflow.add_edge("select_style", "expand_intent")
    workflow.add_edge("expand_intent", "validate_draft")
    
    workflow.add_conditional_edges(
        "validate_draft",
        decide_next_step,
        {
            "self_correct": "self_correct_draft",
            "end": END,
            "human_in_the_loop": "human_in_the_loop" # 실패 시 human_in_the_loop로
        }
    )
    workflow.add_edge("self_correct_draft", "validate_draft")
    
    # --- [신규 추가] Human-in-the-loop 경로 ---
    workflow.add_conditional_edges(
        "human_in_the_loop",
        decide_after_human_edit,
        {
            "validate_draft": "validate_draft",
            "end_with_failure": END
        }
    )
    
    return workflow.compile()

if __name__ == "__main__":
    setup_retrievers()
    app = build_graph()
    user_request = input("\n안녕하세요! 어떤 알림톡 템플릿을 만들어 드릴까요?\n>> ")
    
    initial_state = {"original_request": user_request, "correction_attempts": 0}
    
    final_state = app.invoke(initial_state)
    print(f"\n================ 최종 결과 ================")
    
    final_result = final_state.get('validation_result')
    if final_result:
        # 사용자가 수정을 포기한 경우, 최종 결과는 반려된 상태로 유지
        if final_state.get('user_choice') == 'exit':
             final_result.status = 'rejected'
             final_result.revised_template = None

        # 최종적으로 반려된 경우, 수정된 템플릿은 null 처리
        if final_result.status == 'rejected':
            final_result.revised_template = None
        
        print(json.dumps(final_result.model_dump(), indent=2, ensure_ascii=False))
    else:
        print("오류: 파이프라인이 최종 상태를 반환하지 못했습니다.")
    print("============================================")
