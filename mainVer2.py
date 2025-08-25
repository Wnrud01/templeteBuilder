import os
import json
import re
from typing import TypedDict, List, Dict, Optional
import uuid

# Pydantic 및 LangChain 호환성을 위한 임포트
from pydantic import PrivateAttr

# LangChain 및 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
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

class FlashRankRerank(BaseDocumentCompressor):
    _ranker: Ranker = PrivateAttr()
    top_n: int = 3
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if Ranker:
            self._ranker = Ranker()
    class Config:
        arbitrary_types_allowed = True
    def compress_documents(self, documents: List[Document], query: str, callbacks: Callbacks | None = None) -> List[Document]:
        if not documents or not Ranker: return documents[:self.top_n]
        passages = [{"id": i, "text": doc.page_content, "metadata": doc.metadata} for i, doc in enumerate(documents)]
        rerank_request = RerankRequest(query=query, passages=passages)
        reranked_results = self._ranker.rerank(rerank_request)
        final_docs = []
        for item in reranked_results[:self.top_n]:
            doc = documents[item['id']]
            doc.metadata["relevance_score"] = item['score']
            final_docs.append(doc)
        return final_docs

# --- 0. 초기 설정 ---
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY가 .env 파일에 없습니다.")

# --- 1. 데이터 준비 ---
retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected, retriever_guide = None, None, None, None, None

def load_jsonl_to_docs(file_path: str) -> List[Document]:
    docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    content = data.get("content")
                    metadata = data.get("metadata", {})
                    doc = Document(page_content=content, metadata=metadata)
                    docs.append(doc)
        print(f"✅ {len(docs)}개의 문서를 '{file_path}'에서 로드했습니다.")
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다.")
    except json.JSONDecodeError as e:
        print(f"🚨 오류: {file_path} 파일 파싱 중 오류 발생 - {e}")
    return docs

def load_line_by_line(file_path: str) -> List[Document]:
    docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(Document(page_content=line.strip()))
        print(f"✅ {len(docs)}개의 템플릿을 '{file_path}'에서 로드했습니다.")
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다.")
    return docs

def load_by_separator(file_path: str, separator: str = '---') -> List[Document]:
    docs = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        items = [section.strip() for section in content.split(separator) if section.strip()]
        for item in items:
            docs.append(Document(page_content=item))
        print(f"✅ {len(docs)}개의 항목을 '{file_path}'에서 로드했습니다.")
    except FileNotFoundError:
        print(f"🚨 경고: '{file_path}' 파일을 찾을 수 없습니다.")
    return docs

def setup_retrievers():
    global retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected, retriever_guide
    if all([retriever_compliance, retriever_generation, retriever_whitelist, retriever_rejected, retriever_guide]):
        print("Retriever가 이미 로드되었습니다.")
        return

    print("Retriever 설정 시작...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docs_whitelist = load_line_by_line('./data/approved_templates.txt')
    docs_rejected = load_by_separator('./data/rejected_templates.txt')
    docs_compliance = load_jsonl_to_docs('./data/compliance_rules.jsonl')
    docs_guide = load_jsonl_to_docs('./data/alimtalk_guide.jsonl')

    try:
        docs_generation = TextLoader("./data/generation_rules.txt", encoding='utf-8').load()
        print(f"✅ {len(docs_generation)}개의 문서를 './data/generation_rules.txt'에서 로드했습니다.")
    except Exception as e:
        print(f"🚨 경고: generation_rules.txt 로드 실패 - {e}")
        docs_generation = []

    def create_hybrid_retriever(collection_name, docs, split_docs=False):
        if not docs:
            print(f"⚠️  '{collection_name}'에 대한 문서가 없어 리트리버 생성을 건너뜁니다.")
            return None

        if split_docs:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(docs, embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

        keyword_retriever = BM25Retriever.from_documents(docs)
        keyword_retriever.k = 7

        ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, keyword_retriever], weights=[0.5, 0.5])

        if Ranker:
            compressor = FlashRankRerank(top_n=3)
            return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

        return ensemble_retriever

    retriever_whitelist = create_hybrid_retriever("whitelist_templates", docs_whitelist)
    retriever_rejected = create_hybrid_retriever("rejected_templates", docs_rejected)
    retriever_compliance = create_hybrid_retriever("compliance_rules", docs_compliance)
    retriever_guide = create_hybrid_retriever("alimtalk_guide", docs_guide)
    retriever_generation = create_hybrid_retriever("generation_rules", docs_generation, split_docs=True)

    print("✅ 모든 Retriever 설정 완료!")


# --- 2. LangGraph 파이프라인 정의 ---
class GraphState(TypedDict):
    original_request: str
    request_category: Dict[str, str]
    recommendations: List[Dict]
    user_choice_1: str
    chosen_template: Optional[str]
    available_styles: List[Dict]
    user_choice_2: str
    final_request: str
    template_draft: str
    validation_result: Dict
    correction_attempts: int
    retrieved_guide: str
    retrieved_examples: str
    retrieved_rejected_examples: str

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

classification_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 요청을 가장 적합한 카테고리로 분류하는 전문가입니다.
사용자의 요청을 분석하여, 아래 제공된 1차, 2차 카테고리 중에서 가장 적합한 것을 하나씩 선택해주세요.
반드시 JSON 형식으로만 답변해야 합니다. 예: {{"분류 1차": "구매", "분류 2차": "구매완료"}}

# 1차 분류 옵션:
회원, 구매, 예약, 서비스이용, 리포팅, 배송, 법적고지, 업무알림, 쿠폰/포인트, 기타

# 2차 분류 옵션:
회원가입, 인증/비밀번호/로그인, 회원정보/회원혜택, 구매완료, 상품가입, 진행상태, 구매취소, 구매예약/입고알림, 예약완료/예약내역, 예약상태, 예약취소, 예약알림/리마인드, 이용안내/공지, 신청접수, 처리완료, 이용도구, 방문서비스, 피드백 요청, 구매감사/이용확인, 리마인드, 피드백, 요금청구, 계약/견적, 안전/피해예방, 뉴스래터, 거래알림, 배송상태, 배송예정, 배송완료, 배송실패, 수신동의, 개인정보, 약관변경, 휴면 관련, 주문/예약, 내부 업무 알림, 쿠폰발급, 쿠폰사용, 포인트적립, 포인트사용, 쿠폰/포인트안내, 기타

# 사용자 요청:
{original_request}
"""
)

generation_prompt = ChatPromptTemplate.from_template(
    """당신은 사용자의 구체적인 지시사항에 따라 알림톡 템플릿을 생성하는 AI 전문가입니다.

# 최종 목표:
아래 '최종 요청'을 만족하는 템플릿 초안을 생성하세요.

# 참고 자료:
- **관련 제작 가이드**: 
{guide}

- **유사 성공 사례**: 
{examples}

- **유사 실패 사례 (이런 실수를 반복하지 마세요)**: 
{rejected_examples}

# 최종 요청:
{final_request}

# 템플릿 초안 (위 모든 내용을 종합하여 요청에 가장 부합하는 완벽한 템플릿 하나만 생성):
"""
)

validation_prompt = ChatPromptTemplate.from_template(
    """당신은 정보통신망법 및 내부 규정을 검수하는 매우 꼼꼼한 AI 심사관입니다. 당신의 판단은 명확한 근거에 기반해야 합니다.
    결과는 반드시 JSON 형식으로만 답변해주세요.

    # 검수 대상 템플릿:
    {draft}

    # 관련 규정 (메타데이터 포함):
    {rules}

    # 지시사항:
    1. '검수 대상 템플릿'이 정보성인지 광고성인지 먼저 판단하세요.
    2. [정보성일 경우]: '관련 규정'을 위반하는지 검토하세요. 위반 사항이 없다면 `{{"status": "accept"}}` 라고만 답변하세요. 위반 사항이 있다면, 어떤 규칙(rule_id)을 어떻게 위반했는지 구체적인 이유와 함께 `{{"status": "reject", "reason": "규칙 위반: [구체적 설명]"}}` 형식으로 답변하세요.
    3. [광고성일 경우]: 어떤 문구 때문에 광고성인지 설명과 함께 `{{"status": "reject", "reason": "광고성 메시지: [구체적 설명]"}}` 형식으로 답변하세요.
    """
)

correction_prompt = ChatPromptTemplate.from_template(
    """당신은 까다로운 심사관을 통과시키기 위해 템플릿을 수정하는 창의적인 카피라이터입니다.

    # 수정 목표:
    아래 '반려 사유'를 참고하여 '기존 초안'을 수정하세요. 수정된 템플릿은 반드시 정보성 메시지로 분류되어야 합니다.

    # 수정 전략 (현재 {attempts}차 시도):
    {strategy}

    # 기존 초안:
    {draft}

    # 반려 사유:
    {reason}
    
    # 참고: 고객과의 연결고리 (이 중 하나를 활용하여 정보성으로 전환)
    - 가정 1: 고객은 "브랜드 행사 알림 수신"에 명시적으로 동의했습니다.
    - 가정 2: 고객은 VIP이며, "입점 브랜드 행사 우선 알림" 혜택이 있습니다.
    - 가정 3: 고객은 과거 제품 구매 시 "관련 혜택 정보 수신"에 동의했습니다.

    # 수정된 템플릿 초안 (반드시 템플릿 내용만 간결하게 출력):
    """
)

# --- 인터랙티브 워크플로우 노드들 ---
def classify_request(state: GraphState):
    print("\n--- 1. 사용자 요청 분류 시작 ---")
    original_request = state['original_request']
    chain = classification_prompt | llm | JsonOutputParser()
    try:
        category = chain.invoke({"original_request": original_request})
        state['request_category'] = category
        print(f"✅ 요청 분류 완료: {category}")
    except Exception as e:
        print(f"🚨 요청 분류 중 오류 발생: {e}. 기본 카테고리로 진행합니다.")
        state['request_category'] = {}
    return state

def recommend_templates(state: GraphState):
    print("\n--- 2. 유사 템플릿 추천 시작 ---")
    original_request = state['original_request']

    if retriever_whitelist:
        recommended_docs = retriever_whitelist.invoke(original_request)
        recommendations = [{"id": i+1, "content": doc.page_content} for i, doc in enumerate(recommended_docs)]
        state['recommendations'] = recommendations
        print(f"✅ 사용자의 요청과 유사한 기존 템플릿을 {len(recommendations)}개 찾았습니다.")
    else:
        state['recommendations'] = []
        print("⚠️ 유사 템플릿 리트리버가 설정되지 않아 추천을 건너뜁니다.")

    return state

def present_styles(state: GraphState):
    print("\n--- 3. 템플릿 스타일 선택 제시 ---")

    if retriever_guide:
        try:
            base_retriever = retriever_guide.base_retriever
            if hasattr(base_retriever, 'base_retriever'):
                vectorstore = base_retriever.base_retriever.vectorstore
            elif hasattr(base_retriever, 'retrievers'):
                vectorstore = base_retriever.retrievers[0].vectorstore
            else:
                vectorstore = base_retriever.vectorstore

            style_docs = vectorstore.similarity_search(
                "강조 유형",
                filter={"part_title": "강조 유형"}
            )
        except Exception as e:
            print(f"🚨 스타일 검색 중 오류 발생: {e}. 기본 방식으로 재시도합니다.")
            style_docs = retriever_guide.invoke("템플릿 강조 유형")

        available_styles = []
        seen_titles = set()
        for doc in style_docs:
            title = doc.metadata.get('section_title')
            if title and title not in seen_titles:
                summary = doc.page_content.split('\n')[0]
                available_styles.append({"title": title, "summary": summary})
                seen_titles.add(title)
        state['available_styles'] = available_styles
    else:
        state['available_styles'] = []
        print("⚠️ 가이드 리트리버가 설정되지 않아 스타일 제시를 건너뜁니다.")

    return state

def prepare_final_request(state: GraphState):
    print("\n--- 4. 최종 생성 요청 준비 ---")
    original_request = state['original_request']

    if state.get('user_choice_1') == 'recommend':
        chosen_template_content = state['chosen_template']
        final_request = f"'{original_request}' 요청을 반영하여, 아래 템플릿을 기반으로 수정 및 확장해주세요.\n\n[기반 템플릿]:\n{chosen_template_content}"
    else:
        chosen_style_title = state['user_choice_2']
        final_request = f"'{original_request}' 요청에 맞춰, '{chosen_style_title}' 스타일의 새로운 템플릿을 생성해주세요."

    state['final_request'] = final_request
    print(f"✅ 최종 생성 요청이 준비되었습니다.")
    return state

def generate_draft(state: GraphState):
    print("\n--- 5. 가이드 기반 초안 생성 시작 ---")
    final_request = state['final_request']

    guide_docs = retriever_guide.invoke(final_request) if retriever_guide else []
    example_docs = retriever_whitelist.invoke(final_request) if retriever_whitelist else []
    rejected_docs = retriever_rejected.invoke(final_request) if retriever_rejected else []
    generation_docs = retriever_generation.invoke(final_request) if retriever_generation else []

    guide = "\n\n".join([f"가이드 섹션: {doc.metadata.get('section_title', '')}\n{doc.page_content}" for doc in guide_docs])
    guide += "\n\n" + "\n\n".join([doc.page_content for doc in generation_docs])
    examples = "\n\n".join([doc.page_content for doc in example_docs])
    rejected_examples = "\n\n".join([doc.page_content for doc in rejected_docs])

    state.update({ "retrieved_guide": guide, "retrieved_examples": examples, "retrieved_rejected_examples": rejected_examples })

    chain = generation_prompt | llm | StrOutputParser()
    draft = chain.invoke({ "guide": guide, "examples": examples, "rejected_examples": rejected_examples, "final_request": final_request })

    state['template_draft'] = draft
    state['correction_attempts'] = 0
    print("✅ 초안 생성 완료.")
    print(f"생성된 초안:\n---\n{draft}\n---")
    return state

def validate_draft(state: GraphState):
    print(f"\n--- 6. 규정 준수 검증 (시도: {state['correction_attempts'] + 1}) ---")
    draft = state['template_draft']

    rules_docs = retriever_compliance.invoke(draft) if retriever_compliance else []
    rules = "\n\n".join([f"Rule ID: {doc.metadata.get('rule_id', 'N/A')}\n{doc.page_content}" for doc in rules_docs])

    validation_chain = validation_prompt | llm | JsonOutputParser()
    try:
        validation_result = validation_chain.invoke({"draft": draft, "rules": rules})
    except Exception as e:
        print(f"🚨 검증 중 JSON 파싱 오류 발생: {e}. 반려 처리합니다.")
        validation_result = {"status": "reject", "reason": "검증 시스템 오류"}

    state['validation_result'] = validation_result

    if validation_result.get("status") == "accept":
        print("✅ 검증 통과!")
    else:
        print(f"🚨 검증 반려. 이유: {validation_result.get('reason')}")

    return state

def correct_draft(state: GraphState):
    print("\n--- 7. AI 자동 수정 시작 ---")

    attempts = state['correction_attempts']
    draft = state['template_draft']
    reason = state['validation_result'].get('reason', '알 수 없는 이유')

    if attempts == 0:
        strategy = "1차 수정: 반려 사유에 언급된 광고성 표현이나 문구를 최소한으로 변경하여 정보성으로 만드세요. (예: '단독 특가!' -> '회원님께 적용되는 혜택 안내')"
    elif attempts == 1:
        strategy = "2차 수정: 좀 더 적극적으로 광고성 단어를 정보성 단어로 순화하세요. 고객이 요청한 정보(예: 주문, 예약, 포인트)를 중심으로 메시지를 재구성하세요."
    else:
        strategy = "3차(최종) 수정: 관점을 완전히 바꾸세요. '고객과의 연결고리' 중 하나를 선택하여, 이 메시지가 고객의 사전 동의나 자격에 따른 '정보 안내'임을 명확히 하는 문장을 서두에 추가하여 메시지 전체의 성격을 바꾸세요."

    print(f"수정 전략: {strategy}")

    correction_chain = correction_prompt | llm | StrOutputParser()
    corrected_draft = correction_chain.invoke({
        "attempts": attempts + 1,
        "strategy": strategy,
        "draft": draft,
        "reason": reason
    })

    state['template_draft'] = corrected_draft
    state['correction_attempts'] = attempts + 1

    print("✅ AI 수정 완료. 새로운 초안:")
    print(f"---\n{corrected_draft}\n---")

    return state

def get_user_correction(state: GraphState):
    print("\n--- 8. 사용자 직접 수정 ---")
    final_draft = state['template_draft']
    reason = state['validation_result'].get('reason', '알 수 없음')

    print("AI의 자동 수정으로 검증을 통과하지 못했습니다.")
    print("마지막으로 직접 수정할 기회를 드립니다.")
    print("\n🔥 현재 템플릿 (반려) 🔥")
    print(final_draft)
    print(f"\n최종 반려 사유: {reason}")
    print("-" * 50)

    print("위 템플릿을 직접 수정해주세요. 수정을 마친 후 Enter를 두 번 눌러 입력을 완료하세요.")
    user_input_lines = []
    while True:
        try:
            line = input()
            if not line:
                break
            user_input_lines.append(line)
        except EOFError:
            break

    user_corrected_draft = "\n".join(user_input_lines)

    if not user_corrected_draft.strip():
        print("입력이 없어 수정을 건너뜁니다.")
        state['correction_attempts'] += 1
        return state

    state['template_draft'] = user_corrected_draft
    state['correction_attempts'] += 1

    print("\n✅ 사용자 수정이 반영되었습니다. 최종 검증을 시작합니다.")
    print(f"---\n{user_corrected_draft}\n---")

    return state

def decide_recommendation_branch(state: GraphState):
    if state.get('user_choice_1') == 'recommend':
        return "prepare_request"
    else:
        return "present_styles"

def decide_correction_or_end(state: GraphState):
    if state['validation_result'].get("status") == "accept":
        return "end"

    if state['correction_attempts'] < 3:
        return "correct_ai"

    if state['correction_attempts'] == 3:
        return "correct_user"

    return "end"

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("classify_request", classify_request)
    workflow.add_node("recommend_templates", recommend_templates)
    workflow.add_node("present_styles", present_styles)
    workflow.add_node("prepare_final_request", prepare_final_request)
    workflow.add_node("generate_draft", generate_draft)
    workflow.add_node("validate_draft", validate_draft)
    workflow.add_node("correct_draft", correct_draft)
    workflow.add_node("get_user_correction", get_user_correction)

    workflow.set_entry_point("classify_request")

    workflow.add_edge("classify_request", "recommend_templates")
    workflow.add_conditional_edges(
        "recommend_templates",
        decide_recommendation_branch,
        {"prepare_request": "prepare_final_request", "present_styles": "present_styles"}
    )
    workflow.add_edge("present_styles", "prepare_final_request")
    workflow.add_edge("prepare_final_request", "generate_draft")
    workflow.add_edge("generate_draft", "validate_draft")

    workflow.add_conditional_edges(
        "validate_draft",
        decide_correction_or_end,
        {
            "correct_ai": "correct_draft",
            "correct_user": "get_user_correction",
            "end": END
        }
    )

    workflow.add_edge("correct_draft", "validate_draft")
    workflow.add_edge("get_user_correction", "validate_draft")

    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_after=["recommend_templates", "present_styles"]
    )

# --- 3. 메인 실행 로직 (인터랙티브) ---
if __name__ == "__main__":
    setup_retrievers()
    app = build_graph()

    print("\n==================================================")
    print("템플릿 생성 어시스턴트 V7.4 (입력 대기 수정)")
    print("==================================================")
    original_request = input("어떤 템플릿을 만들어 드릴까요? >> ")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    inputs = {"original_request": original_request}

    # 1단계: 첫 번째 중단점까지 실행
    current_state = None
    for event in app.stream(inputs, config=config):
        (node, state) = next(iter(event.items()))
        current_state = state
        if node == "recommend_templates":
            break

    # 2단계: 사용자 입력 받기
    recommendations = current_state.get("recommendations", [])
    print("\n--------------------------------------------------")
    print("🧐 이런 템플릿은 어떠세요? (유사 템플릿 추천)")
    if recommendations:
        for r in recommendations:
            if r and r.get('content'):
                print(f"\n[{r['id']}] {r['content']}")
    else:
        print("추천할 만한 유사 템플릿을 찾지 못했습니다.")
    print("\n--------------------------------------------------")

    choice = input("사용할 템플릿 번호를 입력하거나, 새로 만들려면 '0'을 입력하세요 >> ")

    # 3단계: 사용자 입력에 따라 상태 업데이트 후 나머지 그래프 실행
    final_events = []
    if choice != '0' and choice.isdigit() and recommendations and 1 <= int(choice) <= len(recommendations):
        # 추천 템플릿 선택
        user_choice_data = {
            "user_choice_1": "recommend",
            "chosen_template": recommendations[int(choice)-1]['content']
        }
        app.update_state(config, user_choice_data)
        # 나머지 그래프 실행
        final_events = list(app.stream(None, config=config))
    else:
        # 새로 만들기 선택
        app.update_state(config, {"user_choice_1": "new"})
        # 다음 중단점(스타일 제시)까지 실행
        for event in app.stream(None, config=config):
            (node, state) = next(iter(event.items()))
            current_state = state
            if node == "present_styles":
                break

        # 스타일 선택 입력 받기
        available_styles = current_state.get("available_styles", [])
        print("\n--------------------------------------------------")
        print("🎨 새로 만들 템플릿의 스타일을 선택해주세요.")
        if available_styles:
            for i, s in enumerate(available_styles):
                if s and s.get('title'):
                    print(f"[{i+1}] {s['title']}: {s.get('summary', '')}")
        else:
            print("선택 가능한 스타일을 찾지 못했습니다. 기본형으로 생성됩니다.")
        print("\n--------------------------------------------------")

        style_choice_num = input(f"원하는 스타일 번호를 입력하세요 (1-{len(available_styles) if available_styles else 1}) >> ")

        chosen_style = ""
        if available_styles and style_choice_num.isdigit() and 1 <= int(style_choice_num) <= len(available_styles):
            chosen_style = available_styles[int(style_choice_num)-1]['title']
        else:
            print("잘못된 입력이거나 선택 가능한 스타일이 없습니다. '기본형'으로 생성합니다.")
            chosen_style = "기본형"

        # 스타일 선택 상태 업데이트 후 나머지 그래프 실행
        app.update_state(config, {"user_choice_2": chosen_style})
        final_events = list(app.stream(None, config=config))

    # 4단계: 최종 결과 출력
    print("\n================ 최종 결과 ================")
    if final_events:
        last_node, final_state = next(iter(final_events[-1].items()))

        if final_state:
            final_draft = final_state.get('template_draft', '오류: 최종 템플릿을 생성하지 못했습니다.')
            validation_result = final_state.get('validation_result', {})

            if validation_result.get('status') == 'accept':
                print("🎉 최종 템플릿 (승인) 🎉")
                print(final_draft)
            else:
                print("🔥 최종 템플릿 (반려) 🔥")
                print(final_draft)
                print(f"\n최종 반려 사유: {validation_result.get('reason', '알 수 없음')}")
        else:
            print("오류: 그래프의 최종 상태를 가져오지 못했습니다.")
    else:
        print("템플릿 생성 과정이 완료되었거나 사용자에 의해 중단되었습니다.")
    print("============================================")
