from pathlib import Path
from typing import Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path("E:\project\Ewha_graduate_project\src\RAG")
GUIDELINE_DIR = BASE_DIR / "data" / "guidelines"
PAPER_DIR = BASE_DIR / "data" / "paper"
FAISS_DIR = BASE_DIR / "faissDB" / "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    "P11_Immunotherapy_in_ovarian_cancer.pdf": {
        "doc_id": "P11",
        "organization": "review",
        "year": 2023,
        "source_type": "paper",
        "paper_type": "Immunotherapy",
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

        if not l:
            continue

        if len(l) <= 5 and any(ch.isdigit() for ch in l):
            continue

        if low.startswith("http"):
            continue

        if "https://ejgo.org" in low:
            continue

        cleaned_lines.append(l)

    return "\n".join(cleaned_lines).strip()


def split_into_paragraph_candidates(text: str) -> List[str]:
    blocks = [blk.strip() for blk in text.split("\n\n") if blk.strip()]

    if len(blocks) <= 1:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        blocks = []
        current = []

        for line in lines:
            current.append(line)

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

    if is_reference_like(t):
        return False

    if has_too_many_numbers(t):
        return False

    if t.count(".") < 2:
        return False

    clinical_keywords = [
        "treatment",
        "therapy",
        "chemotherapy",
        "platinum",
        "bevacizumab",
        "parp",
        "surgery",
        "management",
        "recurrence",
        "recurrent",
        "survival",
        "maintenance",
        "follow-up",
        "surveillance",
    ]

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

    if source_type == "paper" and not any(k in t for k in clinical_keywords):
        return False

    if source_type == "guideline" and not any(k in t for k in guideline_keywords):
        return False

    return True


def split_long_block(block_text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = splitter.create_documents([block_text])
    return [d.page_content.strip() for d in docs if d.page_content.strip()]


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
                        metadata=metadata,
                    )
                )

                chunk_counter += 1

    return {
        "raw_pages": raw_page_docs,
        "chunks": final_chunks,
    }


def build_all_chunks(verbose: bool = True) -> List[Document]:
    all_chunks = []

    guideline_files = sorted(GUIDELINE_DIR.glob("*.pdf"))
    paper_files = sorted(PAPER_DIR.glob("*.pdf"))

    if verbose:
        print(f"[CHECK] guideline files: {len(guideline_files)}")
        for p in guideline_files:
            print(" -", p.name)

        print(f"\n[CHECK] paper files: {len(paper_files)}")
        for p in paper_files:
            print(" -", p.name)

    for pdf_path in guideline_files:
        meta = GUIDELINE_METADATA.get(pdf_path.name)

        if meta is None:
            if verbose:
                print(f"[SKIP] guideline metadata 없음: {pdf_path.name}")
            continue

        if verbose:
            print(f"\n[LOAD GUIDELINE] {pdf_path.name}")

        result = load_and_chunk_one(pdf_path, meta)

        if verbose:
            print(f"  -> raw pages: {len(result['raw_pages'])}")
            print(f"  -> filtered chunks: {len(result['chunks'])}")

        all_chunks.extend(result["chunks"])

    for pdf_path in paper_files:
        meta = PAPER_METADATA.get(pdf_path.name)

        if meta is None:
            if verbose:
                print(f"[SKIP] paper metadata 없음: {pdf_path.name}")
            continue

        if verbose:
            print(f"\n[LOAD PAPER] {pdf_path.name}")

        result = load_and_chunk_one(pdf_path, meta)

        if verbose:
            print(f"  -> raw pages: {len(result['raw_pages'])}")
            print(f"  -> filtered chunks: {len(result['chunks'])}")

        all_chunks.extend(result["chunks"])

    return all_chunks


def build_faiss_db(all_chunks: List[Document]) -> None:
    if not all_chunks:
        raise ValueError("생성된 chunk가 없습니다. PDF 경로와 metadata를 확인하세요.")

    print(f"\n총 filtered chunk 수: {len(all_chunks)}")

    print("\n=== 첫 번째 chunk 샘플 ===")
    print(all_chunks[0].metadata)
    print(all_chunks[0].page_content[:500])

    print("\n[EMBED] HuggingFace embeddings 로드")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[FAISS] 인덱스 생성")
    db = FAISS.from_documents(all_chunks, embeddings)

    FAISS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[SAVE] 저장: {FAISS_DIR}")
    db.save_local(str(FAISS_DIR))

    print("\n[STEP 2 완료] FAISS DB 저장 완료")


def is_meaningful_chunk(text: str) -> bool:
    text = text.lower().strip()

    if len(text) < 80:
        return False

    bad_patterns = [
        "pocket guidelines",
        "published in",
        "table of contents",
        "author",
        "keywords",
        "doi:",
        "copyright",
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

        if meta.get("source_type") == "guideline":
            score += 3.0
        else:
            score += 1.0

        score += float(meta.get("priority", 1.0))

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

        if has_treatment_signal(text):
            score += 1.5
        else:
            score -= 1.0

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored]


def print_doc(doc: Document, idx: int) -> None:
    print(f"\n--- Top {idx} ---")
    print(doc.metadata)
    print(doc.page_content[:700])


def test_retrieval() -> None:
    print("\n[LOAD] embeddings")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print("[LOAD] FAISS")
    db = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    query = "clinical guideline recommendation for treatment of recurrent ovarian cancer"
    print("\n[QUERY]", query)

    docs = db.similarity_search(query, k=15)
    docs = [doc for doc in docs if is_meaningful_chunk(doc.page_content)]
    docs = rerank_docs(docs, query)[:5]

    print(f"\n[RESULT] top {len(docs)}")

    for i, doc in enumerate(docs, 1):
        print_doc(doc, i)


def main() -> None:
    print("=" * 60)
    print("[STEP 1] PDF 문서 로드 및 chunk 생성")
    print("=" * 60)

    all_chunks = build_all_chunks(verbose=True)

    print("=" * 60)
    print("[STEP 2] FAISS DB 구축 및 저장")
    print("=" * 60)

    build_faiss_db(all_chunks)

    print("=" * 60)
    print("[STEP 3] FAISS 검색 테스트")
    print("=" * 60)

    test_retrieval()

    print("=" * 60)
    print("[DONE] RAG FAISS DB 구축 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()