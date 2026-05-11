#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pwd


# In[ ]:


# step 1

from pathlib import Path
from typing import List, Dict

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


BASE_DIR = Path("/tf/RAG")
GUIDELINE_DIR = BASE_DIR / "data" / "guidelines"
PAPER_DIR = BASE_DIR / "data" / "paper"


# =========================
# 1. metadata 정의
# =========================
GUIDELINE_METADATA = {
    "D1_ksgo_2023.pdf": {
        "doc_id": "D1",
        "organization": "KSGO",
        "year": 2023,
        "source_type": "guideline",
        "guideline_type": "clinical_guideline",
        "topic": "ovarian cancer",
        "priority": 3.0,
    },
    "D2_ksgo_2025.pdf": {
        "doc_id": "D2",
        "organization": "KSGO",
        "year": 2025,
        "source_type": "guideline",
        "guideline_type": "update_guideline",
        "topic": "ovarian cancer",
        "priority": 3.2,
    },
    "D3_ksgo_2018.pdf": {
        "doc_id": "D3",
        "organization": "KSGO",
        "year": 2018,
        "source_type": "guideline",
        "guideline_type": "consensus_statement",
        "topic": "ovarian cancer",
        "priority": 2.7,
    },
    "D4_esgo_esmo_esp_2024.pdf": {
        "doc_id": "D4",
        "organization": "ESGO-ESMO-ESP",
        "year": 2024,
        "source_type": "guideline",
        "guideline_type": "consensus_statement",
        "topic": "ovarian cancer",
        "priority": 3.1,
    },
    "D5_esmo_nonepithelial_2018.pdf": {
        "doc_id": "D5",
        "organization": "ESMO",
        "year": 2018,
        "source_type": "guideline",
        "guideline_type": "clinical_guideline",
        "topic": "non-epithelial ovarian cancer",
        "priority": 2.6,
    },
    "D6_esmo_epithelial_2023.pdf": {
        "doc_id": "D6",
        "organization": "ESMO",
        "year": 2023,
        "source_type": "guideline",
        "guideline_type": "clinical_guideline",
        "topic": "epithelial ovarian cancer",
        "priority": 3.0,
    },
    "D7_esmo_hbo_2022.pdf": {
        "doc_id": "D7",
        "organization": "ESMO",
        "year": 2022,
        "source_type": "guideline",
        "guideline_type": "clinical_guideline",
        "topic": "hereditary breast-ovarian cancer",
        "priority": 2.3,
    },
    "D8_NICE_2023.pdf": {
        "doc_id": "D8",
        "organization": "NICE",
        "year": 2023,
        "source_type": "guideline",
        "guideline_type": "clinical_guideline",
        "topic": "ovarian cancer",
        "priority": 2.3,
    },
    "D9_NCCN_2025.pdf": {
        "doc_id": "D9",
        "organization": "NCCN",
        "year": 2025,
        "source_type": "guideline",
        "guideline_type": "clinical_guideline",
        "topic": "ovarian cancer",
        "priority": 2.3,
    },
    "D10_NCCN_2026.pdf": {
        "doc_id": "D10",
        "organization": "NCCN",
        "year": 2026,
        "source_type": "guideline",
        "guideline_type": "pocket_guideline",
        "topic": "ovarian cancer",
        "priority": 2.8,
    },
    "D11_esgo_pocket_2024.pdf": {
        "doc_id": "D11",
        "organization": "ESGO",
        "year": 2024,
        "source_type": "guideline",
        "guideline_type": "pocket_guideline",
        "topic": "ovarian cancer",
        "priority": 2.8,
    },
}

PAPER_METADATA = {
    "P1_ovarian_cancer.pdf": {
        "doc_id": "P1",
        "organization": "Nature Reviews Disease Primers",
        "year": 2016,
        "source_type": "paper",
        "paper_type": "review",
        "topic": "ovarian cancer overview",
        "priority": 1.0,
    },
    "P2_Treatment_of_recurrent_ovarian_cancer.pdf": {
        "doc_id": "P2",
        "organization": "Annals of Oncology",
        "year": 2019,
        "source_type": "paper",
        "paper_type": "treatment_review",
        "topic": "recurrent ovarian cancer treatment",
        "priority": 1.3,
    },
    "P3_Personalized_Treatment_in_Ovarian_Cancer.pdf": {
        "doc_id": "P3",
        "organization": "Cancers",
        "year": 2022,
        "source_type": "paper",
        "paper_type": "review",
        "topic": "personalized treatment",
        "priority": 1.1,
    },
    "P4_Comparative_effectiveness_and_safety.pdf": {
        "doc_id": "P4",
        "organization": "clinical study",
        "year": 2024,
        "source_type": "paper",
        "paper_type": "clinical_study",
        "topic": "comparative effectiveness and safety",
        "priority": 1.2,
    },
    "P5_Prognostic_Factors_After_the_First_Recurrence_of_Ovarian_Cancer.pdf": {
        "doc_id": "P5",
        "organization": "clinical study",
        "year": 2017,
        "source_type": "paper",
        "paper_type": "clinical_study",
        "topic": "prognostic factors after recurrence",
        "priority": 1.1,
    },
    "P6_Immunotherapy_in_Recurrent_Ovarian_Cancer.pdf": {
        "doc_id": "P6",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "review",
        "topic": "recurrent ovarian cancer immunotherapy",
        "priority": 1.0,
    },
    "P7_Survivorship_in_advanced_ovarian_cancer.pdf": {
        "doc_id": "P7",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "survivorship",
        "priority": 1.0,
    },
    "P8_Treatment_of_recurrent_ovarian_cancer.pdf": {
        "doc_id": "P8",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "survivorship",
        "priority": 1.0,
    },
    "P9_ESMO_Clinical_Practice_Guideline.pdf": {
        "doc_id": "P9",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "survivorship",
        "priority": 1.0,
    },
    "P10_PARP_inhibitors_in_ovarian_cancer.pdf": {
        "doc_id": "P10",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "ovarian_cancer",
        "priority": 1.0,
    },
    "P12_High-grade_serous_ovarian_cancer.pdf": {
        "doc_id": "P12",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "ovarian_cancer",
        "priority": 1.0,
    },
    "P13_Follow-up_strategies_in_ovarian_cancer.pdf": {
        "doc_id": "P13",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "ovarian_cancer",
        "priority": 1.0,
    },
    "P14_PARP_inhibitor_era_in_ovarian_cancer_treatment_2024.pdf": {
        "doc_id": "P14",
        "organization": "review",
        "year": 2024,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "ovarian_cancer",
        "priority": 1.0,
    },
    "P15_Randomized_Trial_of_Cytoreductive_Surgery_for_Relapsed_2021.pdf": {
        "doc_id": "P15",
        "organization": "review",
        "year": 2021,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "Cytoreductive_Surgery",
        "priority": 1.0,
    },
    "P16_Treatment_options_for_recurrent_platinum_resistant_ovarian_cancer_2023.pdf": {
        "doc_id": "P16",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "recurrent",
        "priority": 1.0,
    },
    "P17_WHO_health_literacy.pdf": {
        "doc_id": "P17",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "survivorship",
        "priority": 1.0,
    },
    "P18_Comparative_effectiveness_and_safety_of_treatment_regimens_for_recurrent_advanced_ovarian_cancer_2025.pdf": {
        "doc_id": "P18",
        "organization": "review",
        "year": 2025,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "survivorship",
        "priority": 1.0,
    },
    "P19_esmo_ESGO_ESMO_ESP_consensus_conference_recommendations_on_ovarian_cancer_2024.pdf": {
        "doc_id": "P19",
        "organization": "review",
        "year": 2024,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "ovarian cancer",
        "priority": 1.0,
    },
    "P20_esmo_treatment_of_recurrent_ovarian_cancer_pdf.pdf": {
        "doc_id": "P20",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "treatment",
        "priority": 1.0,
    },
    "P21_Immunotherapy_in_Recurrent_Ovarian_Cancer_2025.pdf": {
        "doc_id": "P21",
        "organization": "review",
        "year": 2025,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "Immunotherapy",
        "priority": 1.0,
    },
    "P22_Niraparib_Maintenance_Therapy_in_Platinum_Sensitive_Recurrent_Ovarian_Cancer_2016.pdf": {
        "doc_id": "P22",
        "organization": "review",
        "year": 2016,
        "source_type": "paper",
        "paper_type": "survivorship_review",
        "topic": "Therapy",
        "priority": 1.0,
    },
    
}


# =========================
# 2. 텍스트 정리 함수
# =========================
def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    text = "\n".join(lines)
    return text.strip()


def remove_obvious_noise_lines(text: str) -> str:
    cleaned_lines = []

    for line in text.splitlines():
        l = line.strip()
        low = l.lower()

        # 빈 줄
        if not l:
            continue

        # 페이지 번호처럼 보이는 줄
        if len(l) <= 5 and any(ch.isdigit() for ch in l):
            continue

        # 너무 짧은 URL/doi 성격 줄
        if low.startswith("http"):
            continue

        # 헤더/푸터성
        if "https://ejgo.org" in low:
            continue

        cleaned_lines.append(l)

    return "\n".join(cleaned_lines).strip()


# =========================
# 3. 문단 단위 후보 만들기
# =========================
def split_into_paragraph_candidates(text: str) -> List[str]:
    """
    1차: 빈 줄 기준
    2차: 너무 긴 경우 줄 단위 병합
    """
    blocks = [blk.strip() for blk in text.split("\n\n") if blk.strip()]

    # 빈 줄이 거의 없는 PDF를 대비해서 추가 처리
    if len(blocks) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current = []

        for line in lines:
            current.append(line)
            # 문장 끝나면 약하게 끊기
            if line.endswith(".") or line.endswith(":"):
                blocks.append(" ".join(current).strip())
                current = []

        if current:
            blocks.append(" ".join(current).strip())

    return blocks


def merge_short_blocks(blocks: List[str], min_len: int = 250) -> List[str]:
    merged = []
    buffer = ""

    for blk in blocks:
        if not buffer:
            buffer = blk
            continue

        if len(buffer) < min_len:
            buffer = buffer + " " + blk
        else:
            merged.append(buffer.strip())
            buffer = blk

    if buffer:
        merged.append(buffer.strip())

    return merged


# =========================
# 4. chunk 유효성 판단
# =========================
def is_reference_like(text: str) -> bool:
    t = text.lower()
    reference_patterns = [
        "et al.",
        "ann oncol",
        "j clin oncol",
        "lancet",
        "doi:",
        "suppl",
        "phase iii trial",
        "randomized",
        "placebo-controlled",
        "vol.",
        "pp.",
    ]
    return t.count(".") > 5 and any(p in t for p in reference_patterns)


def has_too_many_numbers(text: str) -> bool:
    words = text.split()
    if not words:
        return True
    n = sum(1 for w in words if any(c.isdigit() for c in w))
    return (n / len(words)) > 0.20


def is_useful_paragraph(text: str, source_type: str) -> bool:
    t = text.lower().strip()

    if len(t) < 120:
        return False

    # cover / meta / useless
    bad_patterns = [
        "pocket guidelines",
        "published in",
        "table of contents",
        "author contributions",
        "conflict of interest",
        "funding",
        "copyright",
        "all rights reserved",
        "orcid",
        "supplementary material",
    ]
    if any(p in t for p in bad_patterns):
        return False

    # intro / methods / abstract 계열
    intro_patterns = [
        "this review",
        "we discuss",
        "introduction",
        "background",
        "objective",
        "abstract",
        "aim of this study",
        "purpose of this",
        "methods",
        "study design",
        "systematic review",
        "meta-analysis",
        "pico",
        "committee",
        "voting procedure",
        "summary of evidence supporting",
        "current guidelines cover",
        "developing the recommendations",
        # 🔥 추가 (중요)
        "the development of guidelines",
        "mission to improve",
        "consensus conference",
        "standard operating procedures",
        "sop",
        "following the",
        "aim was to discuss",
    ]
    if any(p in t for p in intro_patterns):
        return False

    # references
    if is_reference_like(t):
        return False

    if has_too_many_numbers(t):
        return False

    # 최소 문장 수
    if t.count(".") < 2:
        return False

    # guideline는 조금 넓게, paper는 조금 엄격하게
    if source_type == "paper":
        clinical_keywords = [
            "treatment", "therapy", "chemotherapy", "platinum",
            "bevacizumab", "parp", "surgery", "management",
            "recurrence", "recurrent", "survival", "maintenance",
            "follow-up", "surveillance"
        ]
        if not any(k in t for k in clinical_keywords):
            return False
    if source_type == "guideline":
        guideline_keywords = [
            "treatment",
            "therapy",
            "management",
            "recurrence",
            "follow-up",
            "surveillance",
            "surgery",
            "chemotherapy",
            "platinum",
            "bevacizumab",
            "parp",
        ]
        if not any(k in t for k in guideline_keywords):
            return False
    return True


# =========================
# 5. 긴 문단 재분할
# =========================
def split_long_block(block_text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs = splitter.create_documents([block_text])
    return [d.page_content.strip() for d in docs if d.page_content.strip()]


# =========================
# 6. PDF 하나 처리
# =========================
def load_and_chunk_one(pdf_path: Path, meta: Dict) -> Dict[str, List[Document]]:
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    raw_page_docs = []
    final_chunks = []

    chunk_counter = 0

    for page_doc in pages:
        page_text = page_doc.page_content
        page_text = normalize_text(page_text)
        page_text = remove_obvious_noise_lines(page_text)

        if not page_text:
            continue

        raw_page_docs.append(page_doc)

        paragraph_blocks = split_into_paragraph_candidates(page_text)
        paragraph_blocks = merge_short_blocks(paragraph_blocks, min_len=250)

        for block in paragraph_blocks:
            if len(block) > 1200:
                sub_blocks = split_long_block(block)
            else:
                sub_blocks = [block]

            for sub in sub_blocks:
                sub = normalize_text(sub)
                if not is_useful_paragraph(sub, meta["source_type"]):
                    continue

                metadata = dict(page_doc.metadata)
                metadata["source_file"] = pdf_path.name
                metadata["doc_id"] = meta["doc_id"]
                metadata["organization"] = meta["organization"]
                metadata["year"] = meta["year"]
                metadata["source_type"] = meta["source_type"]
                metadata["topic"] = meta["topic"]
                metadata["priority"] = meta["priority"]
                metadata["chunk_index"] = chunk_counter

                if meta["source_type"] == "guideline":
                    metadata["guideline_type"] = meta["guideline_type"]
                else:
                    metadata["paper_type"] = meta["paper_type"]

                final_chunks.append(
                    Document(
                        page_content=sub,
                        metadata=metadata
                    )
                )
                chunk_counter += 1

    return {
        "raw_pages": raw_page_docs,
        "chunks": final_chunks,
    }


# =========================
# 7. 전체 chunk 빌드
# =========================
def build_all_chunks():
    all_chunks = []

    guideline_files = sorted(GUIDELINE_DIR.glob("*.pdf"))
    paper_files = sorted(PAPER_DIR.glob("*.pdf"))

    print(f"[CHECK] guideline files: {len(guideline_files)}")
    for p in guideline_files:
        print(" -", p.name)

    print(f"\n[CHECK] paper files: {len(paper_files)}")
    for p in paper_files:
        print(" -", p.name)

    # guideline
    for pdf_path in guideline_files:
        meta = GUIDELINE_METADATA.get(pdf_path.name)
        if meta is None:
            print(f"[SKIP] guideline metadata 없음: {pdf_path.name}")
            continue

        print(f"\n[LOAD GUIDELINE] {pdf_path.name}")
        result = load_and_chunk_one(pdf_path, meta)
        print(f"  -> raw pages: {len(result['raw_pages'])}")
        print(f"  -> filtered chunks: {len(result['chunks'])}")
        all_chunks.extend(result["chunks"])

    # paper
    for pdf_path in paper_files:
        meta = PAPER_METADATA.get(pdf_path.name)
        if meta is None:
            print(f"[SKIP] paper metadata 없음: {pdf_path.name}")
            continue

        print(f"\n[LOAD PAPER] {pdf_path.name}")
        result = load_and_chunk_one(pdf_path, meta)
        print(f"  -> raw pages: {len(result['raw_pages'])}")
        print(f"  -> filtered chunks: {len(result['chunks'])}")
        all_chunks.extend(result["chunks"])

    return all_chunks

# ==== 별도 파일 없이 하나의 주피터 노트북에서 부를 수 있게 함수 추가 정의

def build_all_chunks():
    all_chunks = []
    guideline_files = sorted(GUIDELINE_DIR.glob("*.pdf"))
    paper_files = sorted(PAPER_DIR.glob("*.pdf"))
    # guideline
    for pdf_path in guideline_files:
        meta = GUIDELINE_METADATA.get(pdf_path.name)
        if meta is None:
            continue
        result = load_and_chunk_one(pdf_path, meta)
        all_chunks.extend(result["chunks"])
    # paper
    for pdf_path in paper_files:
        meta = PAPER_METADATA.get(pdf_path.name)
        if meta is None:
            continue
        result = load_and_chunk_one(pdf_path, meta)
        all_chunks.extend(result["chunks"])
    return all_chunks
# =========================
# 8. main
# =========================
def main():
    all_chunks = build_all_chunks()

    print(f"\n총 filtered chunk 수: {len(all_chunks)}")

    if all_chunks:
        print("\n=== 첫 번째 filtered chunk 샘플 ===")
        print(all_chunks[0].metadata)
        print(all_chunks[0].page_content[:1000])


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:


# step 2

from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings



BASE_DIR = Path("/tf/RAG")
FAISS_DIR = BASE_DIR / "model" / "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

all_chunks = build_all_chunks()
print(f"총 chunk 수: {len(all_chunks)}")
print(all_chunks[0].metadata)
print(all_chunks[0].page_content[:500])

def main():
    print("[LOAD] filtered chunks from step1")
    all_chunks = build_all_chunks()
    print(f"총 filtered chunk 수: {len(all_chunks)}")

    if all_chunks:
        print("\n=== 첫 번째 chunk 샘플 ===")
        print(all_chunks[0].metadata)
        print(all_chunks[0].page_content[:500])

    print("\n[EMBED] HuggingFace embeddings 로드")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[FAISS] 인덱스 생성")
    db = FAISS.from_documents(all_chunks, embeddings)

    FAISS_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"[SAVE] 저장: {FAISS_DIR}")
    db.save_local(str(FAISS_DIR))

    print("\n완료!")
    print(f"저장 위치: {FAISS_DIR}")


if __name__ == "__main__":
    main()


# In[ ]:


# step 3

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

FAISS_DIR = "/tf/RAG/model/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def is_meaningful_chunk(text: str) -> bool:
    text = text.lower().strip()

    if len(text) < 80:
        return False

    # ❗ 완전 중요: methodology 제거
    bad_patterns = [
        "pocket guidelines",
        "published in",
        "table of contents",
        "author",
        "keywords",
        "doi:",
        "copyright",

        # 🔥 추가 (핵심)
        "methods",
        "developing the recommendations",
        "committee",
        "systematic review",
        "meta-analysis",
        "pico",
        "study design",
    ]

    if any(p in text for p in bad_patterns):
        return False

    # intro 제거
    intro_patterns = [
        "this review",
        "we discuss",
        "introduction",
        "background",
        "objective",
        "abstract",
        "aim of this study",
        "current guidelines cover",
    ]

    if any(p in text for p in intro_patterns):
        return False

    if text.count(".") < 2:
        return False

    return True


def has_treatment_signal(text: str) -> bool:
    text = text.lower()

    keywords = [
        "treatment",
        "therapy",
        "chemotherapy",
        "platinum",
        "bevacizumab",
        "parp",
        "surgery",
        "management",
        "relapse",
        "recurrent",
        "recurrence",
    ]
    return any(k in text for k in keywords)


def rerank_docs(docs, query: str):
    q = query.lower()
    scored = []

    for doc in docs:
        meta = doc.metadata
        text = doc.page_content.lower()

        score = 0.0

        # 1) guideline 우선
        if meta.get("source_type") == "guideline":
            score += 3.0
        else:
            score += 1.0

        # 2) metadata priority 반영
        score += float(meta.get("priority", 1.0))

        # 3) query 핵심 단어 반영
        query_keywords = [
            "recurrent",
            "recurrence",
            "treatment",
            "therapy",
            "management",
            "platinum",
            "survival",
        ]
        for kw in query_keywords:
            if kw in q and kw in text:
                score += 0.8

        # 4) guideline/paper 종류 반영
        guideline_type = meta.get("guideline_type")
        paper_type = meta.get("paper_type")

        if guideline_type in ["clinical_guideline", "update_guideline", "consensus_statement"]:
            score += 1.0
        elif guideline_type == "pocket_guideline":
            score += 0.4

        if paper_type == "treatment_review":
            score += 0.8
        elif paper_type == "clinical_study":
            score += 0.5
        elif paper_type == "review":
            score += 0.2

        # 5) 치료 관련 실제 내용이면 가산점
        if has_treatment_signal(text):
            score += 1.5
        else:
            score -= 1.0

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored]


def print_doc(doc, idx):
    print(f"\n--- Top {idx} ---")
    print(doc.metadata)
    print(doc.page_content[:700])


def main():
    print("[LOAD] embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[LOAD] FAISS")
    db = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    query = "clinical guideline recommendation for treatment of recurrent ovarian cancer"
    print("\n[QUERY]", query)

    # 넉넉히 가져온 뒤 필터링 + rerank
    docs = db.similarity_search(query, k=15)

    # hard filtering
    docs = [doc for doc in docs if is_meaningful_chunk(doc.page_content)]

    # rerank
    docs = rerank_docs(docs, query)[:5]

    print(f"\n[RESULT] top {len(docs)}\n")

    for i, doc in enumerate(docs, 1):
        print_doc(doc, i)


if __name__ == "__main__":
    main()


# In[ ]:


import os
print(os.path.expanduser("~/.cache/huggingface/transformers"))


# In[ ]:


# step 4 - llama2

import os
from pathlib import Path
from typing import Dict
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------
# 환경 변수 설정 (로컬 전용)
# ----------------------
os.environ["HF_HOME"] = "/root/.cache/huggingface"  # Hugging Face 캐시
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ----------------------
# 경로 설정
# ----------------------
FAISS_DIR = Path("/tf/RAG/model/faiss_index")  # FAISS DB 경로
MODEL_PATH = "/tf/RAG/model/Llama-2-7b-chat-hf"  # 로컬 모델 폴더
OUTPUT_PATH = Path("/tf/RAG/output/patient_report_kor_llama2.txt")  # 최종 보고서 저장
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

# ----------------------
# 헬퍼 함수
# ----------------------
def is_good_chunk(text: str) -> bool:
    t = text.lower()
    bad_keywords = ["pubmed", "crossref", "references", "available online", "doi.org", "copyright"]
    return not any(k in t for k in bad_keywords) and len(text.strip()) >= 120

def get_lesion_side(lesion_location: str) -> str:
    if not lesion_location:
        return "불명확"
    t = lesion_location.lower()
    if "bilateral" in t or "both" in t or "양측" in t:
        return "양측"
    if "right" in t or "우측" in t:
        return "우측"
    if "left" in t or "좌측" in t:
        return "좌측"
    return "불명확"

def get_class_description(patient_data: Dict) -> str:
    diagnosis = str(patient_data.get("hierarchical_diagnosis", "정보 없음")).strip()
    malignancy = str(patient_data.get("malignancy", "정보 없음")).strip().lower()
    recurrence = bool(patient_data.get("recurrence", False))
    if malignancy == "malignant":
        if recurrence:
            return f"이번 결과는 재발 가능성까지 함께 고려한 악성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
        return f"이번 결과는 원발성 악성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
    if malignancy == "benign":
        return f"이번 결과는 양성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
    return f"현재 결과는 '{diagnosis}' 범주로 해석되지만, 악성/양성 정보는 명확하지 않습니다."

def get_prediction_focus_summary(patient_data: Dict) -> str:
    lesion_side = get_lesion_side(patient_data.get("lesion_location", ""))
    diagnosis = patient_data.get("hierarchical_diagnosis", "정보 없음")
    malignancy = patient_data.get("malignancy", "정보 없음")
    recurrence = "재발 가능성 고려" if patient_data.get("recurrence") else "처음 발견된 병변 가능성"
    grad_cam_finding = patient_data.get("grad_cam_finding", "특이 영상 소견 정보 없음")
    return (
        f"병변 위치는 {lesion_side}로 정리됩니다. "
        f"모델은 이 병변을 '{diagnosis}' 범주로 해석했고, "
        f"악성 여부는 '{malignancy}', 현재 상태는 '{recurrence}'로 판단했습니다. "
        f"영상에서 참고한 추가 소견은 '{grad_cam_finding}'입니다."
    )

def build_search_query(patient_data: Dict) -> str:
    query_parts = []
    lesion = patient_data.get("lesion_location", "")
    diagnosis = patient_data.get("hierarchical_diagnosis", "")
    if "(" in lesion and ")" in lesion:
        query_parts.append(lesion.split("(")[1].split(")")[0].strip())
    else:
        query_parts.append(lesion if lesion else "Ovary")
    if "(" in diagnosis and ")" in diagnosis:
        query_parts.append(diagnosis.split("(")[1].split(")")[0].strip())
    else:
        query_parts.append(diagnosis if diagnosis else "Epithelial Ovarian Cancer")
    query_parts.append("recurrent" if patient_data.get("recurrence") else "primary")
    query_parts.append("patient explanation follow up treatment guideline")
    return " ".join([p for p in query_parts if p])

# ----------------------
# FAISS DB 로드
# ----------------------
def load_vector_db():
    print("[STATUS] Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[STATUS] Loading FAISS Vector DB...")
    db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve_guidelines(db, patient_data: Dict, k: int = 5) -> str:
    search_query = build_search_query(patient_data)
    print(f"[STATUS] Searching Guidelines... (Query: {search_query})")
    docs = db.similarity_search(search_query, k=k)
    filtered_docs = [doc for doc in docs if is_good_chunk(doc.page_content)]
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('organization', 'Unknown')} {doc.metadata.get('year', 'Unknown')}]\n{doc.page_content}"
        for doc in filtered_docs
    )
    return context_text if context_text else "No relevant guidelines found."

# ----------------------
# 로컬 모델 LLM 로드
# ----------------------
# load_llm 함수 수정 예시
def load_llm(model_path: str):
    # 1. 절대 경로로 변환하여 인식률 높이기
    full_path = os.path.abspath(model_path)
    print(f"[STATUS] Loading Local LLM from: {full_path}")
    
    # 2. 경로 존재 여부 사전 체크
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {full_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. 로컬 파일만 사용하도록 명시 (이미 설정하셨지만 다시 확인)
    tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        full_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
        device_map="auto" if device == "cuda" else None
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=700,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        top_p=0.9,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

# ----------------------
# 환자 보고서 생성
# ----------------------
def build_patient_prompt(patient_data: Dict, context: str) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    diagnosis = patient_data.get("hierarchical_diagnosis", "N/A")
    malignancy = patient_data.get("malignancy", "N/A")
    recurrence = "재발이 의심되는 상태" if patient_data.get("recurrence") else "처음 발견된 병변으로 해석되는 상태"
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    class_description = get_class_description(patient_data)
    prediction_focus = get_prediction_focus_summary(patient_data)

    prompt_template = """
당신은 환자에게 검사 결과를 쉽게 설명해주는 의료 설명 도우미다.

아래 요구사항에 맞춰 한국어 환자용 설명 보고서를 작성하라.

[핵심 요구사항]
1. 전문가 대상이 아닌 일반인이 이해할 수 있는 쉬운 설명으로 쓸 것
2. 예측 결과에 집중해서 설명할 것
3. 병변 위치는 우측/좌측/양측/불명확처럼 쉽게 전달할 것
4. class 설명은 "이번 결과가 어떤 범주에 가까운지" 형태로 쉽게 풀어 쓸 것
5. 이미지와 임상 정보만으로는 바로 알기 어려운 내용 중,
   가이드라인을 참고하면 추가로 생각해볼 수 있는 검사/치료/추적관찰 내용을 설명할 것
6. 확정 진단처럼 단정하지 말고, 참고 설명이라는 톤을 유지할 것
7. 불안을 과도하게 유발하는 표현은 피하되 중요한 내용은 숨기지 말 것
8. 검색된 가이드라인에 없는 내용은 지어내지 말 것

[환자 정보]
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 진단명: {diagnosis}
- 악성 여부: {malignancy}
- 상태: {recurrence}
- 영상 참고 소견: {grad_cam_finding}
- class 설명: {class_description}
- 예측 결과 요약: {prediction_focus}

[검색된 임상 가이드라인]
{context}

[출력 형식]
## 환자용 간단 보고서
### 1. 이번 결과가 의미하는 것
### 2. 병변 위치와 현재 분류
### 3. 추가로 생각해볼 검사 또는 치료 방향
### 4. 앞으로 주의 깊게 볼 부분
### 5. 꼭 기억할 점

어렵지 않고 짧고 분명하게 작성할 것.
"""
    prompt = PromptTemplate(
        input_variables=[
            "lesion_location", "lesion_side", "diagnosis", "malignancy",
            "recurrence", "grad_cam_finding", "class_description",
            "prediction_focus", "context"
        ],
        template=prompt_template
    )
    return prompt.format(
        lesion_location=lesion_location,
        lesion_side=lesion_side,
        diagnosis=diagnosis,
        malignancy=malignancy,
        recurrence=recurrence,
        grad_cam_finding=grad_cam_finding,
        class_description=class_description,
        prediction_focus=prediction_focus,
        context=context
    )

def build_patient_info_block(patient_data: Dict) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    diagnosis = patient_data.get("hierarchical_diagnosis", "N/A")
    malignancy = patient_data.get("malignancy", "N/A")
    recurrence = "재발성 의심" if patient_data.get("recurrence") else "원발성 의심"
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    class_description = get_class_description(patient_data)
    prediction_focus = get_prediction_focus_summary(patient_data)
    return f"""
==================================================
                 환자 정보 / 예측 요약
==================================================
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 예측 진단명: {diagnosis}
- 악성 여부: {malignancy}
- 상태 분류: {recurrence}
- 영상 참고 소견: {grad_cam_finding}
- class 설명: {class_description}

[예측 결과 요약]
{prediction_focus}
""".strip()

def clean_response(text: str) -> str:
    return text.strip() if text else ""

def generate_patient_report(patient_data: Dict, model_path: str = MODEL_PATH) -> Dict:
    db = load_vector_db()
    context = retrieve_guidelines(db, patient_data, k=5)
    llm = load_llm(model_path)
    patient_prompt = build_patient_prompt(patient_data, context)
    print("[STATUS] Generating patient report...")
    patient_report = clean_response(llm.invoke(patient_prompt))
    patient_info_block = build_patient_info_block(patient_data)
    final_text = (
        patient_info_block
        + "\n\n"
        + "=" * 50
        + "\n                환자용 간단 설명 보고서\n"
        + "=" * 50
        + "\n"
        + patient_report
        + "\n"
    )
    return {
        "patient_info": patient_info_block,
        "patient_report": patient_report,
        "final_text": final_text
    }

# ----------------------
# 메인 실행 예시
# ----------------------
if __name__ == "__main__":
    patient_test = {
        "lesion_location": "Right Ovary",
        "hierarchical_diagnosis": "Epithelial Ovarian Cancer",
        "malignancy": "Malignant",
        "recurrence": True,
        "grad_cam_finding": "Suspected peritoneal metastasis near right ovary."
    }

    try:
        result = generate_patient_report(patient_test, model_path=MODEL_PATH)
        print(result["final_text"])
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(result["final_text"])
        print(f"\n[SUCCESS] Saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] {e}")


# In[ ]:


# step 4 - llama 3 

import os
from pathlib import Path
from typing import Dict
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------
# 환경 변수 설정 (로컬 전용)
# ----------------------
os.environ["HF_HOME"] = "/root/.cache/huggingface"  # Hugging Face 캐시
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ----------------------
# 경로 설정
# ----------------------
FAISS_DIR = Path("/tf/RAG/model/faiss_index")  # FAISS DB 경로
MODEL_PATH = "/tf/RAG/model/Meta-Llama-3-8B-Instruct"  # 로컬 모델 폴더
OUTPUT_PATH = Path("/tf/RAG/output/patient_report_kor_llama3.txt")  # 최종 보고서 저장
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

# ----------------------
# 헬퍼 함수
# ----------------------
def is_good_chunk(text: str) -> bool:
    t = text.lower()
    bad_keywords = ["pubmed", "crossref", "references", "available online", "doi.org", "copyright"]
    return not any(k in t for k in bad_keywords) and len(text.strip()) >= 120

def get_lesion_side(lesion_location: str) -> str:
    if not lesion_location:
        return "불명확"
    t = lesion_location.lower()
    if "bilateral" in t or "both" in t or "양측" in t:
        return "양측"
    if "right" in t or "우측" in t:
        return "우측"
    if "left" in t or "좌측" in t:
        return "좌측"
    return "불명확"

def get_class_description(patient_data: Dict) -> str:
    diagnosis = str(patient_data.get("hierarchical_diagnosis", "정보 없음")).strip()
    malignancy = str(patient_data.get("malignancy", "정보 없음")).strip().lower()
    recurrence = bool(patient_data.get("recurrence", False))
    if malignancy == "malignant":
        if recurrence:
            return f"이번 결과는 재발 가능성까지 함께 고려한 악성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
        return f"이번 결과는 원발성 악성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
    if malignancy == "benign":
        return f"이번 결과는 양성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
    return f"현재 결과는 '{diagnosis}' 범주로 해석되지만, 악성/양성 정보는 명확하지 않습니다."

def get_prediction_focus_summary(patient_data: Dict) -> str:
    lesion_side = get_lesion_side(patient_data.get("lesion_location", ""))
    diagnosis = patient_data.get("hierarchical_diagnosis", "정보 없음")
    malignancy = patient_data.get("malignancy", "정보 없음")
    recurrence = "재발 가능성 고려" if patient_data.get("recurrence") else "처음 발견된 병변 가능성"
    grad_cam_finding = patient_data.get("grad_cam_finding", "특이 영상 소견 정보 없음")
    return (
        f"병변 위치는 {lesion_side}로 정리됩니다. "
        f"모델은 이 병변을 '{diagnosis}' 범주로 해석했고, "
        f"악성 여부는 '{malignancy}', 현재 상태는 '{recurrence}'로 판단했습니다. "
        f"영상에서 참고한 추가 소견은 '{grad_cam_finding}'입니다."
    )

def build_search_query(patient_data: Dict) -> str:
    query_parts = []
    lesion = patient_data.get("lesion_location", "")
    diagnosis = patient_data.get("hierarchical_diagnosis", "")
    if "(" in lesion and ")" in lesion:
        query_parts.append(lesion.split("(")[1].split(")")[0].strip())
    else:
        query_parts.append(lesion if lesion else "Ovary")
    if "(" in diagnosis and ")" in diagnosis:
        query_parts.append(diagnosis.split("(")[1].split(")")[0].strip())
    else:
        query_parts.append(diagnosis if diagnosis else "Epithelial Ovarian Cancer")
    query_parts.append("recurrent" if patient_data.get("recurrence") else "primary")
    query_parts.append("patient explanation follow up treatment guideline")
    return " ".join([p for p in query_parts if p])

# ----------------------
# FAISS DB 로드
# ----------------------
def load_vector_db():
    print("[STATUS] Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[STATUS] Loading FAISS Vector DB...")
    db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve_guidelines(db, patient_data: Dict, k: int = 5) -> str:
    search_query = build_search_query(patient_data)
    print(f"[STATUS] Searching Guidelines... (Query: {search_query})")
    docs = db.similarity_search(search_query, k=k)
    filtered_docs = [doc for doc in docs if is_good_chunk(doc.page_content)]
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('organization', 'Unknown')} {doc.metadata.get('year', 'Unknown')}]\n{doc.page_content}"
        for doc in filtered_docs
    )
    return context_text if context_text else "No relevant guidelines found."

# ----------------------
# 로컬 모델 LLM 로드
# ----------------------
# load_llm 함수 수정 예시
def load_llm(model_path: str):
    # 1. 절대 경로로 변환하여 인식률 높이기
    full_path = os.path.abspath(model_path)
    print(f"[STATUS] Loading Local LLM from: {full_path}")
    
    # 2. 경로 존재 여부 사전 체크
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {full_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. 로컬 파일만 사용하도록 명시 (이미 설정하셨지만 다시 확인)
    tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        full_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
        device_map="auto" if device == "cuda" else None
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=700,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        top_p=0.9,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

# ----------------------
# 환자 보고서 생성
# ----------------------
def build_patient_prompt(patient_data: Dict, context: str) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    diagnosis = patient_data.get("hierarchical_diagnosis", "N/A")
    malignancy = patient_data.get("malignancy", "N/A")
    recurrence = "재발이 의심되는 상태" if patient_data.get("recurrence") else "처음 발견된 병변으로 해석되는 상태"
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    class_description = get_class_description(patient_data)
    prediction_focus = get_prediction_focus_summary(patient_data)

    prompt_template = """
당신은 환자에게 검사 결과를 쉽게 설명해주는 의료 설명 도우미다.

아래 요구사항에 맞춰 한국어 환자용 설명 보고서를 작성하라.

[핵심 요구사항]
1. 전문가 대상이 아닌 일반인이 이해할 수 있는 쉬운 설명으로 쓸 것
2. 예측 결과에 집중해서 설명할 것
3. 병변 위치는 우측/좌측/양측/불명확처럼 쉽게 전달할 것
4. class 설명은 "이번 결과가 어떤 범주에 가까운지" 형태로 쉽게 풀어 쓸 것
5. 이미지와 임상 정보만으로는 바로 알기 어려운 내용 중,
   가이드라인을 참고하면 추가로 생각해볼 수 있는 검사/치료/추적관찰 내용을 설명할 것
6. 확정 진단처럼 단정하지 말고, 참고 설명이라는 톤을 유지할 것
7. 불안을 과도하게 유발하는 표현은 피하되 중요한 내용은 숨기지 말 것
8. 검색된 가이드라인에 없는 내용은 지어내지 말 것

[환자 정보]
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 진단명: {diagnosis}
- 악성 여부: {malignancy}
- 상태: {recurrence}
- 영상 참고 소견: {grad_cam_finding}
- class 설명: {class_description}
- 예측 결과 요약: {prediction_focus}

[검색된 임상 가이드라인]
{context}

[출력 형식]
## 환자용 간단 보고서
### 1. 이번 결과가 의미하는 것
### 2. 병변 위치와 현재 분류
### 3. 추가로 생각해볼 검사 또는 치료 방향
### 4. 앞으로 주의 깊게 볼 부분
### 5. 꼭 기억할 점

어렵지 않고 짧고 분명하게 작성할 것.
"""
    prompt = PromptTemplate(
        input_variables=[
            "lesion_location", "lesion_side", "diagnosis", "malignancy",
            "recurrence", "grad_cam_finding", "class_description",
            "prediction_focus", "context"
        ],
        template=prompt_template
    )
    return prompt.format(
        lesion_location=lesion_location,
        lesion_side=lesion_side,
        diagnosis=diagnosis,
        malignancy=malignancy,
        recurrence=recurrence,
        grad_cam_finding=grad_cam_finding,
        class_description=class_description,
        prediction_focus=prediction_focus,
        context=context
    )

def build_patient_info_block(patient_data: Dict) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    diagnosis = patient_data.get("hierarchical_diagnosis", "N/A")
    malignancy = patient_data.get("malignancy", "N/A")
    recurrence = "재발성 의심" if patient_data.get("recurrence") else "원발성 의심"
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    class_description = get_class_description(patient_data)
    prediction_focus = get_prediction_focus_summary(patient_data)
    return f"""
==================================================
                 환자 정보 / 예측 요약
==================================================
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 예측 진단명: {diagnosis}
- 악성 여부: {malignancy}
- 상태 분류: {recurrence}
- 영상 참고 소견: {grad_cam_finding}
- class 설명: {class_description}

[예측 결과 요약]
{prediction_focus}
""".strip()

def clean_response(text: str) -> str:
    return text.strip() if text else ""

def generate_patient_report(patient_data: Dict, model_path: str = MODEL_PATH) -> Dict:
    db = load_vector_db()
    context = retrieve_guidelines(db, patient_data, k=5)
    llm = load_llm(model_path)
    patient_prompt = build_patient_prompt(patient_data, context)
    print("[STATUS] Generating patient report...")
    patient_report = clean_response(llm.invoke(patient_prompt))
    patient_info_block = build_patient_info_block(patient_data)
    final_text = (
        patient_info_block
        + "\n\n"
        + "=" * 50
        + "\n                환자용 간단 설명 보고서\n"
        + "=" * 50
        + "\n"
        + patient_report
        + "\n"
    )
    return {
        "patient_info": patient_info_block,
        "patient_report": patient_report,
        "final_text": final_text
    }

# ----------------------
# 메인 실행 예시
# ----------------------
if __name__ == "__main__":
    patient_test = {
        "lesion_location": "Right Ovary",
        "hierarchical_diagnosis": "Epithelial Ovarian Cancer",
        "malignancy": "Malignant",
        "recurrence": True,
        "grad_cam_finding": "Suspected peritoneal metastasis near right ovary."
    }

    try:
        result = generate_patient_report(patient_test, model_path=MODEL_PATH)
        print(result["final_text"])
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(result["final_text"])
        print(f"\n[SUCCESS] Saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] {e}")


# In[ ]:


# step 4 - Qwen 2.5 

import os
from pathlib import Path
from typing import Dict
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------
# 환경 변수 설정 (로컬 전용)
# ----------------------
os.environ["HF_HOME"] = "/root/.cache/huggingface"  # Hugging Face 캐시
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ----------------------
# 경로 설정
# ----------------------
FAISS_DIR = Path("/tf/RAG/model/faiss_index")  # FAISS DB 경로
MODEL_PATH = "/tf/RAG/model/Qwen2.5-14B-Instruct"  # 로컬 모델 폴더
OUTPUT_PATH = Path("/tf/RAG/output/patient_report_kor_qwen25.txt")  # 최종 보고서 저장
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)  # 폴더 없으면 생성

# ----------------------
# 헬퍼 함수
# ----------------------
def is_good_chunk(text: str) -> bool:
    t = text.lower()
    bad_keywords = ["pubmed", "crossref", "references", "available online", "doi.org", "copyright"]
    return not any(k in t for k in bad_keywords) and len(text.strip()) >= 120

def get_lesion_side(lesion_location: str) -> str:
    if not lesion_location:
        return "불명확"
    t = lesion_location.lower()
    if "bilateral" in t or "both" in t or "양측" in t:
        return "양측"
    if "right" in t or "우측" in t:
        return "우측"
    if "left" in t or "좌측" in t:
        return "좌측"
    return "불명확"

def get_class_description(patient_data: Dict) -> str:
    diagnosis = str(patient_data.get("hierarchical_diagnosis", "정보 없음")).strip()
    malignancy = str(patient_data.get("malignancy", "정보 없음")).strip().lower()
    recurrence = bool(patient_data.get("recurrence", False))
    if malignancy == "malignant":
        if recurrence:
            return f"이번 결과는 재발 가능성까지 함께 고려한 악성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
        return f"이번 결과는 원발성 악성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
    if malignancy == "benign":
        return f"이번 결과는 양성 병변 범주에 가깝습니다. 세부 분류는 '{diagnosis}'입니다."
    return f"현재 결과는 '{diagnosis}' 범주로 해석되지만, 악성/양성 정보는 명확하지 않습니다."

def get_prediction_focus_summary(patient_data: Dict) -> str:
    lesion_side = get_lesion_side(patient_data.get("lesion_location", ""))
    diagnosis = patient_data.get("hierarchical_diagnosis", "정보 없음")
    malignancy = patient_data.get("malignancy", "정보 없음")
    recurrence = "재발 가능성 고려" if patient_data.get("recurrence") else "처음 발견된 병변 가능성"
    grad_cam_finding = patient_data.get("grad_cam_finding", "특이 영상 소견 정보 없음")
    return (
        f"병변 위치는 {lesion_side}로 정리됩니다. "
        f"모델은 이 병변을 '{diagnosis}' 범주로 해석했고, "
        f"악성 여부는 '{malignancy}', 현재 상태는 '{recurrence}'로 판단했습니다. "
        f"영상에서 참고한 추가 소견은 '{grad_cam_finding}'입니다."
    )

def build_search_query(patient_data: Dict) -> str:
    query_parts = []
    lesion = patient_data.get("lesion_location", "")
    diagnosis = patient_data.get("hierarchical_diagnosis", "")
    if "(" in lesion and ")" in lesion:
        query_parts.append(lesion.split("(")[1].split(")")[0].strip())
    else:
        query_parts.append(lesion if lesion else "Ovary")
    if "(" in diagnosis and ")" in diagnosis:
        query_parts.append(diagnosis.split("(")[1].split(")")[0].strip())
    else:
        query_parts.append(diagnosis if diagnosis else "Epithelial Ovarian Cancer")
    query_parts.append("recurrent" if patient_data.get("recurrence") else "primary")
    query_parts.append("patient explanation follow up treatment guideline")
    return " ".join([p for p in query_parts if p])

# ----------------------
# FAISS DB 로드
# ----------------------
def load_vector_db():
    print("[STATUS] Initializing Embedding Model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[STATUS] Loading FAISS Vector DB...")
    db = FAISS.load_local(str(FAISS_DIR), embeddings, allow_dangerous_deserialization=True)
    return db

def retrieve_guidelines(db, patient_data: Dict, k: int = 5) -> str:
    search_query = build_search_query(patient_data)
    print(f"[STATUS] Searching Guidelines... (Query: {search_query})")
    docs = db.similarity_search(search_query, k=k)
    filtered_docs = [doc for doc in docs if is_good_chunk(doc.page_content)]
    context_text = "\n\n".join(
        f"[Source: {doc.metadata.get('organization', 'Unknown')} {doc.metadata.get('year', 'Unknown')}]\n{doc.page_content}"
        for doc in filtered_docs
    )
    return context_text if context_text else "No relevant guidelines found."

# ----------------------
# 로컬 모델 LLM 로드
# ----------------------
# load_llm 함수 수정 예시
def load_llm(model_path: str):
    # 1. 절대 경로로 변환하여 인식률 높이기
    full_path = os.path.abspath(model_path)
    print(f"[STATUS] Loading Local LLM from: {full_path}")
    
    # 2. 경로 존재 여부 사전 체크
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {full_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 3. 로컬 파일만 사용하도록 명시 (이미 설정하셨지만 다시 확인)
    tokenizer = AutoTokenizer.from_pretrained(full_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        full_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
        device_map="auto" if device == "cuda" else None
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=700,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.15,
        no_repeat_ngram_size=3,
        top_p=0.9,
        return_full_text=False
    )
    return HuggingFacePipeline(pipeline=pipe)

# ----------------------
# 환자 보고서 생성
# ----------------------
def build_patient_prompt(patient_data: Dict, context: str) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    diagnosis = patient_data.get("hierarchical_diagnosis", "N/A")
    malignancy = patient_data.get("malignancy", "N/A")
    recurrence = "재발이 의심되는 상태" if patient_data.get("recurrence") else "처음 발견된 병변으로 해석되는 상태"
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    class_description = get_class_description(patient_data)
    prediction_focus = get_prediction_focus_summary(patient_data)

    prompt_template = """
당신은 환자에게 검사 결과를 쉽게 설명해주는 의료 설명 도우미다.

아래 요구사항에 맞춰 한국어 환자용 설명 보고서를 작성하라.

[핵심 요구사항]
1. 전문가 대상이 아닌 일반인이 이해할 수 있는 쉬운 설명으로 쓸 것
2. 예측 결과에 집중해서 설명할 것
3. 병변 위치는 우측/좌측/양측/불명확처럼 쉽게 전달할 것
4. class 설명은 "이번 결과가 어떤 범주에 가까운지" 형태로 쉽게 풀어 쓸 것
5. 이미지와 임상 정보만으로는 바로 알기 어려운 내용 중,
   가이드라인을 참고하면 추가로 생각해볼 수 있는 검사/치료/추적관찰 내용을 설명할 것
6. 확정 진단처럼 단정하지 말고, 참고 설명이라는 톤을 유지할 것
7. 불안을 과도하게 유발하는 표현은 피하되 중요한 내용은 숨기지 말 것
8. 검색된 가이드라인에 없는 내용은 지어내지 말 것

[환자 정보]
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 진단명: {diagnosis}
- 악성 여부: {malignancy}
- 상태: {recurrence}
- 영상 참고 소견: {grad_cam_finding}
- class 설명: {class_description}
- 예측 결과 요약: {prediction_focus}

[검색된 임상 가이드라인]
{context}

[출력 형식]
## 환자용 간단 보고서
### 1. 이번 결과가 의미하는 것
### 2. 병변 위치와 현재 분류
### 3. 추가로 생각해볼 검사 또는 치료 방향
### 4. 앞으로 주의 깊게 볼 부분
### 5. 꼭 기억할 점

어렵지 않고 짧고 분명하게 작성할 것.
"""
    prompt = PromptTemplate(
        input_variables=[
            "lesion_location", "lesion_side", "diagnosis", "malignancy",
            "recurrence", "grad_cam_finding", "class_description",
            "prediction_focus", "context"
        ],
        template=prompt_template
    )
    return prompt.format(
        lesion_location=lesion_location,
        lesion_side=lesion_side,
        diagnosis=diagnosis,
        malignancy=malignancy,
        recurrence=recurrence,
        grad_cam_finding=grad_cam_finding,
        class_description=class_description,
        prediction_focus=prediction_focus,
        context=context
    )

def build_patient_info_block(patient_data: Dict) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    diagnosis = patient_data.get("hierarchical_diagnosis", "N/A")
    malignancy = patient_data.get("malignancy", "N/A")
    recurrence = "재발성 의심" if patient_data.get("recurrence") else "원발성 의심"
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    class_description = get_class_description(patient_data)
    prediction_focus = get_prediction_focus_summary(patient_data)
    return f"""
==================================================
                 환자 정보 / 예측 요약
==================================================
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 예측 진단명: {diagnosis}
- 악성 여부: {malignancy}
- 상태 분류: {recurrence}
- 영상 참고 소견: {grad_cam_finding}
- class 설명: {class_description}

[예측 결과 요약]
{prediction_focus}
""".strip()

def clean_response(text: str) -> str:
    return text.strip() if text else ""

def generate_patient_report(patient_data: Dict, model_path: str = MODEL_PATH) -> Dict:
    db = load_vector_db()
    context = retrieve_guidelines(db, patient_data, k=5)
    llm = load_llm(model_path)
    patient_prompt = build_patient_prompt(patient_data, context)
    print("[STATUS] Generating patient report...")
    patient_report = clean_response(llm.invoke(patient_prompt))
    patient_info_block = build_patient_info_block(patient_data)
    final_text = (
        patient_info_block
        + "\n\n"
        + "=" * 50
        + "\n                환자용 간단 설명 보고서\n"
        + "=" * 50
        + "\n"
        + patient_report
        + "\n"
    )
    return {
        "patient_info": patient_info_block,
        "patient_report": patient_report,
        "final_text": final_text
    }

# ----------------------
# 메인 실행 예시
# ----------------------
if __name__ == "__main__":
    patient_test = {
        "lesion_location": "Right Ovary",
        "hierarchical_diagnosis": "Epithelial Ovarian Cancer",
        "malignancy": "Malignant",
        "recurrence": True,
        "grad_cam_finding": "Suspected peritoneal metastasis near right ovary."
    }

    try:
        result = generate_patient_report(patient_test, model_path=MODEL_PATH)
        print(result["final_text"])
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(result["final_text"])
        print(f"\n[SUCCESS] Saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"\n[ERROR] {e}")


# In[ ]:




