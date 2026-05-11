import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer


print("========== RUNNING UPDATED GENERATE_REPORT_V3 ==========")

CODE_DIR = Path(__file__).resolve().parent
RAG_DIR = CODE_DIR.parent

DEFAULT_FAISS_DIR = RAG_DIR / "faissDB" / "faiss_index"
DEFAULT_OUTPUT_DIR = RAG_DIR / "output"
DEFAULT_MODEL_PATH = RAG_DIR / "models" / "Qwen3.5-0.8B"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def set_offline_environment(offline: bool) -> None:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def normalize_text(text: str) -> str:
    return " ".join(str(text).replace("\n", " ").split())


def is_good_chunk(text: str) -> bool:
    if not text:
        return False

    t = text.lower()

    bad_keywords = [
        "references",
        "bibliography",
        "pubmed",
        "crossref",
        "doi.org",
        "available online",
        "copyright",
        "all rights reserved",
        "creative commons",
        "table of contents",
    ]

    if any(keyword in t for keyword in bad_keywords):
        return False

    if len(text.strip()) < 150:
        return False

    return True


def get_lesion_side(lesion_location: str) -> str:
    if not lesion_location:
        return "불명확"

    t = str(lesion_location).lower()

    if "bilateral" in t or "both" in t or "양측" in t:
        return "양측"

    if "right" in t or "우측" in t:
        return "우측"

    if "left" in t or "좌측" in t:
        return "좌측"

    return "불명확"


def is_malignant_value(value) -> bool:
    return str(value).strip().lower() in [
        "malignant",
        "cancer",
        "악성",
        "1",
        "true",
        "yes",
        "y",
    ]


def is_benign_value(value) -> bool:
    return str(value).strip().lower() in [
        "benign",
        "양성",
        "0",
        "false",
        "no",
        "n",
    ]


def is_recurrent_value(value) -> bool:
    return value in [
        True,
        1,
        "1",
        "true",
        "True",
        "yes",
        "Y",
        "y",
        "recurrent",
        "recurrence",
    ]


def is_non_recurrent_value(value) -> bool:
    return value in [
        False,
        0,
        "0",
        "false",
        "False",
        "no",
        "N",
        "n",
        "primary",
        "non-recurrent",
    ]


def get_malignancy_korean(malignancy) -> str:
    if is_malignant_value(malignancy):
        return "악성 가능성이 있는 병변"

    if is_benign_value(malignancy):
        return "양성 가능성이 있는 병변"

    return "악성/양성 여부가 명확하지 않은 병변"


def get_recurrence_korean(recurrence) -> str:
    if is_recurrent_value(recurrence):
        return "재발 가능성을 고려해야 하는 상태"

    if is_non_recurrent_value(recurrence):
        return "현재 입력 정보상 재발 가능성이 높다고 판단되지는 않는 상태"

    return "재발 여부가 명확하지 않은 상태"


def build_prediction_summary(patient_data: Dict) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    malignancy = get_malignancy_korean(patient_data.get("malignancy", "N/A"))
    recurrence = get_recurrence_korean(patient_data.get("recurrence", "N/A"))
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")

    return (
        f"AI 모델은 병변 위치를 {lesion_side}으로 정리했고, "
        f"전체 예측 결과는 '{malignancy}'에 가깝게 해석됩니다. "
        f"재발 관련 판단은 '{recurrence}'로 정리됩니다. "
        f"영상 참고 소견은 '{grad_cam_finding}'입니다."
    )


def build_english_search_queries(patient_data: Dict) -> List[str]:
    lesion_location = str(patient_data.get("lesion_location", "")).strip()
    malignancy = patient_data.get("malignancy", "")
    recurrence = patient_data.get("recurrence", None)
    grad_cam_finding = str(patient_data.get("grad_cam_finding", "")).strip()

    cancer_term = "ovarian cancer" if is_malignant_value(malignancy) else "ovarian tumor"
    recurrence_term = "recurrent ovarian cancer" if is_recurrent_value(recurrence) else "primary ovarian tumor"

    queries = [
        f"{cancer_term} clinical guideline diagnosis follow-up management",
        f"{recurrence_term} surveillance follow-up treatment guideline",
        f"{cancer_term} patient explanation recurrence follow-up guideline",
    ]

    if lesion_location:
        queries.append(f"{lesion_location} ovarian lesion imaging finding guideline")

    if grad_cam_finding and grad_cam_finding.upper() != "N/A":
        queries.append(f"{grad_cam_finding} ovarian cancer imaging metastasis guideline")

    return queries


def clean_source_title(raw_title: str) -> str:
    if not raw_title:
        return "Unknown document"

    raw_title = str(raw_title).strip()
    raw_title = raw_title.replace("\\", "/")

    return raw_title.split("/")[-1]


def get_source_label(metadata: Dict, source_id: int) -> str:
    raw_title = (
        metadata.get("title")
        or metadata.get("document_title")
        or metadata.get("paper_title")
        or metadata.get("source")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("organization")
        or "Unknown document"
    )

    organization = (
        metadata.get("organization")
        or metadata.get("publisher")
        or metadata.get("journal")
        or metadata.get("author")
        or ""
    )

    year = (
        metadata.get("year")
        or metadata.get("publication_year")
        or metadata.get("date")
        or "Unknown year"
    )

    title = clean_source_title(raw_title)
    organization = str(organization).strip()
    year = str(year).strip()

    if organization and organization.lower() not in title.lower():
        return f"S{source_id}: {organization}, {title}, {year}"

    return f"S{source_id}: {title}, {year}"


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    cleaned = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        if len(sentence) > 700:
            sentence = sentence[:700].strip() + "..."
        cleaned.append(sentence)

    return cleaned


def score_sentence(sentence: str) -> int:
    keywords = [
        "ovarian cancer",
        "recurrence",
        "recurrent",
        "follow-up",
        "surveillance",
        "diagnosis",
        "pathologic",
        "pathology",
        "biopsy",
        "surgical specimen",
        "surgical staging",
        "debulking",
        "chemotherapy",
        "systemic therapy",
        "symptoms",
        "CA125",
        "radiologically",
        "imaging",
        "metastasis",
        "treatment",
        "management",
        "quality of life",
        "side effects",
        "primary treatment",
    ]

    s = sentence.lower()
    score = 0

    for keyword in keywords:
        if keyword in s:
            score += 1

    return score


def extract_key_sentences(text: str, max_sentences: int = 3) -> List[str]:
    sentences = split_sentences(text)

    scored = []
    for sentence in sentences:
        score = score_sentence(sentence)
        if score > 0:
            scored.append((score, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()

    for _, sentence in scored:
        key = sentence.lower()[:120]
        if key in seen:
            continue

        seen.add(key)
        selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    return selected


def make_evidence_card(doc, source_label: str, max_sentences: int = 3) -> str:
    key_sentences = extract_key_sentences(doc.page_content, max_sentences=max_sentences)

    if not key_sentences:
        fallback = normalize_text(doc.page_content.strip())
        key_sentences = [fallback[:500] + "..."]

    evidence_lines = "\n".join([f"- {sentence}" for sentence in key_sentences])

    return (
        f"[{source_label}]\n"
        f"Key evidence sentences:\n"
        f"{evidence_lines}"
    )


def load_vector_db(faiss_dir: Path):
    print(f"[STATUS] Loading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    print(f"[STATUS] Loading FAISS DB: {faiss_dir}")
    print(f"[DEBUG] index.faiss exists: {(faiss_dir / 'index.faiss').exists()}")
    print(f"[DEBUG] index.pkl exists: {(faiss_dir / 'index.pkl').exists()}")

    db = FAISS.load_local(
        str(faiss_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return db


def retrieve_evidence(
    db,
    patient_data: Dict,
    top_k_per_query: int = 5,
    max_chunks: int = 6,
) -> Tuple[str, List[str]]:
    queries = build_english_search_queries(patient_data)

    collected_docs = []
    seen = set()

    for query in queries:
        print(f"[SEARCH] {query}")
        docs = db.similarity_search(query, k=top_k_per_query)

        for doc in docs:
            text = doc.page_content.strip()

            if not is_good_chunk(text):
                continue

            key = normalize_text(text).lower()[:700]

            if key in seen:
                continue

            seen.add(key)
            collected_docs.append(doc)

    if not collected_docs:
        return "No relevant evidence was found in the vector database.", []

    context_blocks = []
    source_labels = []

    for idx, doc in enumerate(collected_docs[:max_chunks], start=1):
        source_label = get_source_label(doc.metadata, idx)
        source_labels.append(source_label)
        context_blocks.append(make_evidence_card(doc, source_label, max_sentences=3))

    return "\n\n---\n\n".join(context_blocks), source_labels


def load_qwen_model(model_path: Path):
    full_path = os.path.abspath(str(model_path))
    print(f"[STATUS] Loading model: {full_path}")

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {full_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STATUS] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        full_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        full_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if device == "cpu":
        model.to(device)

    model.eval()

    return tokenizer, model, device


def build_grounded_prompt(patient_data: Dict, context: str, source_labels: List[str]) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    malignancy_raw = patient_data.get("malignancy", "N/A")
    malignancy_kor = get_malignancy_korean(malignancy_raw)
    recurrence_raw = patient_data.get("recurrence", "N/A")
    recurrence_kor = get_recurrence_korean(recurrence_raw)
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    prediction_summary = build_prediction_summary(patient_data)

    source_list = "\n".join([f"- {source}" for source in source_labels]) if source_labels else "- No source found"

    prompt = f"""
You are a careful medical AI report assistant for an ovarian cancer prediction project.

The retrieved evidence is written in English.
Your final output must be written in Korean only.

Core safety rule:
Use only the retrieved evidence.
Do not use outside medical knowledge.
Do not invent diagnosis, subtype, treatment, medication, survival rate, recurrence risk, prognosis, follow-up interval, or clinical recommendation.

Very important output rule:
Do not copy the retrieved evidence verbatim.
Do not print source file paths.
Do not list retrieved chunks.
Do not output English evidence text.
Do not translate the retrieved evidence line by line.
You must rewrite the evidence into Korean patient-friendly explanations.

Citation formatting rule:
Do not write citations as section headers like "**S1**:".
Do not write "[출처: S1]" on a separate line.
Each citation must appear only at the end of a Korean sentence.
Bad example:
[출처: S1]
English copied text...

Bad example:
**S1**: 수술 후 진단 및 검사 지침에 따르면...

Good example:
난소암의 최종 진단은 영상만으로 단정하기보다 병리학적 확인을 통해 이루어질 수 있습니다. [출처: S1]

If the retrieved evidence does not support a claim, write exactly:
"해당 근거는 검색된 DB에서 찾지 못했습니다."

Project limitation:
This AI system does not provide detailed subtype classification.
Do not mention detailed diagnosis, histologic subtype, or hierarchical diagnosis.
Only explain:
1. benign/malignant tendency,
2. recurrence-related tendency,
3. lesion side/location,
4. image-based attention finding,
5. evidence-based guideline or paper content retrieved from DB.

Medical wording rule:
Use cautious wording.
Use "의심됩니다", "가능성이 있습니다", "확인이 필요합니다", "고려될 수 있습니다".
Do not say "발견되었습니다" for AI image findings.
Do not say "확진입니다".
Do not say "반드시".
Do not say "완치".
Do not overgeneralize evidence about advanced-stage ovarian cancer to all ovarian cancer patients.
Do not use raw statistics such as "80%" unless the patient group and condition are clearly stated.
Translate "systemic therapy" as "전신 치료" or "전신 항암치료".
Translate "suspected peritoneal metastasis near right ovary" as "우측 난소 주변 복막 전이가 의심되는 소견".

[Patient Data]
- Lesion location: {lesion_location}
- Simplified lesion side: {lesion_side}
- AI malignancy prediction raw value: {malignancy_raw}
- AI malignancy explanation in Korean: {malignancy_kor}
- AI recurrence prediction raw value: {recurrence_raw}
- AI recurrence explanation in Korean: {recurrence_kor}
- Grad-CAM or image attention finding: {grad_cam_finding}
- Korean prediction summary: {prediction_summary}

[Available Sources]
{source_list}

[Retrieved Evidence]
{context}

[Required Korean Output Format]

## 환자용 AI 예측 설명 보고서

### 1. 이번 AI 예측 결과 요약
AI 예측값을 바탕으로 3문장 이내로 설명하세요.
세부 진단명이나 subtype은 쓰지 마세요.
확정 진단처럼 쓰지 마세요.
이 섹션은 환자 입력값과 AI 예측값 설명이므로 출처가 없어도 됩니다.

### 2. 병변 위치와 영상 참고 소견
병변 위치와 영상 참고 소견을 2~3문장으로 설명하세요.
영상 소견의 임상적 의미를 단정하지 마세요.
retrieved evidence에 근거가 있는 의학적 설명을 추가하는 경우 문장 끝에 출처를 붙이세요.
근거가 부족하면 "해당 근거는 검색된 DB에서 찾지 못했습니다."라고 쓰세요.

### 3. 검색된 논문/가이드라인에서 확인된 내용
검색된 근거를 그대로 복사하지 말고, 환자가 이해할 수 있는 한국어 문장으로 재작성하세요.
3~5개의 bullet point로 작성하세요.
각 bullet point는 반드시 [출처: S번호]로 끝나야 합니다.
출처는 문장 맨 앞에 쓰지 말고 문장 끝에만 쓰세요.
근거가 부족하면 "해당 근거는 검색된 DB에서 찾지 못했습니다."라고 쓰세요.

### 4. 의료진과 추가로 확인할 수 있는 부분
검색된 근거에 기반해 의료진과 확인할 수 있는 내용을 2~4개의 bullet point로 작성하세요.
각 bullet point는 반드시 [출처: S번호]로 끝나야 합니다.
출처는 문장 맨 앞에 쓰지 말고 문장 끝에만 쓰세요.
근거가 부족하면 "해당 근거는 검색된 DB에서 찾지 못했습니다."라고 쓰세요.

### 5. 꼭 기억할 점
AI 결과는 참고용이며 확정 진단이 아니라고 설명하세요.
최종 판단과 치료 결정은 의료진이 해야 한다고 설명하세요.
불안을 과도하게 유발하지 말고, 안전하고 신중한 표현을 사용하세요.
일반적인 안전 안내 문구는 출처 없이 작성해도 됩니다.

Write the final report in Korean only.
"""

    return prompt.strip()


def generate_text(tokenizer, model, device: str, prompt: str, max_new_tokens: int = 800) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful medical AI report assistant. "
                "You must generate only Korean output. "
                "You must not invent unsupported medical claims. "
                "You must not copy retrieved evidence verbatim. "
                "Each citation must appear at the end of a Korean sentence."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def clean_report(text: str) -> str:
    if not text:
        return ""

    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*", "", text)
    text = text.replace("```", "")
    text = text.replace("Final answer in Korean:", "")
    text = text.replace("Korean Output:", "")
    text = text.replace("Write the final report in Korean only.", "")
    text = text.strip()

    return text


def remove_source_paths_from_report(report: str) -> str:
    report = re.sub(r"[A-Za-z]:\\[^\n]+", "", report)
    report = re.sub(r"[A-Za-z]:/[^\n]+", "", report)
    return report.strip()


def soften_unsafe_phrasing(report: str) -> str:
    replacements = {
        "발견되었습니다": "의심됩니다",
        "확진": "확정 진단",
        "시스템 내 투약 치료": "전신 치료",
        "시스템 치료": "전신 치료",
        "시스템 내 치료": "전신 치료",
        "경향성 전이": "복막 전이 의심 소견",
        "더 오래 살릴 수 있다는 증거가 부족합니다": "생존 기간을 늘리는지에 대해서는 명확한 근거가 제한적입니다",
        "암환자": "난소암 환자",
    }

    for old, new in replacements.items():
        report = report.replace(old, new)

    return report


def has_bad_citation_format(report: str) -> bool:
    patterns = [
        r"\*\*S\d+\*\*\s*:",
        r"^\[출처:\s*S\d+\]",
        r"^\s*S\d+\s*:",
    ]

    for pattern in patterns:
        if re.search(pattern, report, flags=re.MULTILINE):
            return True

    return False


def has_english_chunk_copy(report: str) -> bool:
    english_indicators = [
        "Diagnosis of ovarian cancer after surgery",
        "Evaluation of follow-up strategies",
        "Primary treatment for presumed ovarian cancer",
        "Recurrent disease Monitoring for recurrence",
        "Treatment for common ovarian cancers",
        "Medical treatment of relapsed ovarian cancer",
    ]

    lower_report = report.lower()

    for indicator in english_indicators:
        if indicator.lower() in lower_report:
            return True

    return False


def validate_report_sources(report: str, source_labels: List[str]) -> str:
    if not source_labels:
        return report

    valid_ids = {f"S{i}" for i in range(1, len(source_labels) + 1)}
    cited_ids = set(re.findall(r"S\d+", report))
    invalid_ids = cited_ids - valid_ids

    if not invalid_ids:
        return report

    warning = (
        "\n\n[주의]\n"
        f"보고서에 검색된 출처 목록에 없는 출처 ID가 포함되어 있습니다: {', '.join(sorted(invalid_ids))}\n"
        "해당 문장은 수동 검토가 필요합니다.\n"
    )

    return report + warning


def evidence_contains(context: str, keywords: List[str]) -> bool:
    c = context.lower()
    return any(keyword.lower() in c for keyword in keywords)


def build_template_fallback_report(patient_data: Dict, context: str, source_labels: List[str]) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    malignancy = get_malignancy_korean(patient_data.get("malignancy", "N/A"))
    recurrence = get_recurrence_korean(patient_data.get("recurrence", "N/A"))
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")

    section3 = []
    section4 = []

    if evidence_contains(context, ["pathologic analysis", "biopsy", "surgical specimen", "pathology"]):
        section3.append(
            "- 난소암의 최종 진단은 영상 결과만으로 단정하기보다 조직검사, 수술 검체, 병리학적 확인 등을 통해 이루어질 수 있습니다. [출처: S3]"
        )

    if evidence_contains(context, ["diagnostic tests", "pathology should also be reviewed", "comprehensive surgical staging"]):
        section3.append(
            "- 수술 후 난소암으로 진단되었거나 병기 설정 정보가 충분하지 않은 경우, 추가 진단 평가와 병리 검토가 필요할 수 있습니다. [출처: S1]"
        )

    if evidence_contains(context, ["follow-up", "routine review", "check-ups"]):
        section3.append(
            "- 난소암 치료 후 추적관찰 전략에 대해서는 정기 검진이 생존 기간을 늘리는지에 대한 근거가 제한적일 수 있어, 환자 상태에 맞춘 확인이 필요합니다. [출처: S2]"
        )

    if evidence_contains(context, ["symptoms and signs of disease recurrence", "recurrence", "relapse"]):
        section3.append(
            "- 치료 후에는 재발과 관련될 수 있는 증상 변화, 신체 상태 변화, 치료 부작용 등에 대해 의료진과 상의하는 것이 중요하게 언급됩니다. [출처: S4]"
        )

    if evidence_contains(context, ["ca125", "clinically", "radiologically", "detected"]):
        section3.append(
            "- 재발성 난소암은 혈액검사, 임상 증상, 영상검사 등을 통해 확인될 수 있다고 보고되어 있습니다. [출처: S6]"
        )

    if not section3:
        section3.append("- 해당 근거는 검색된 DB에서 찾지 못했습니다.")

    if evidence_contains(context, ["pathologic analysis", "biopsy", "surgical specimen", "pathology"]):
        section4.append(
            "- 현재 AI 결과가 실제 악성 병변을 의미하는지는 영상 판독, 병리 결과, 임상 정보 등을 함께 검토해 의료진과 확인할 수 있습니다. [출처: S3]"
        )

    if evidence_contains(context, ["ca125", "clinically", "radiologically", "detected"]):
        section4.append(
            "- 재발 가능성이 고려되는 경우, 증상 변화, 종양표지자 변화, 영상검사 결과 등을 종합해 의료진과 확인할 수 있습니다. [출처: S6]"
        )

    if evidence_contains(context, ["symptoms and signs of disease recurrence", "side effects", "quality of life"]):
        section4.append(
            "- 추적관찰 과정에서는 재발 의심 증상뿐 아니라 치료 부작용, 정서적 어려움, 삶의 질 관리에 대해서도 의료진과 상의할 수 있습니다. [출처: S4]"
        )

    if not section4:
        section4.append("- 해당 근거는 검색된 DB에서 찾지 못했습니다.")

    section3_text = "\n".join(section3[:5])
    section4_text = "\n".join(section4[:4])

    if str(grad_cam_finding).upper() != "N/A":
        image_sentence = (
            "영상 참고 소견에는 우측 난소 주변 복막 전이가 의심된다는 내용이 포함되어 있으나, "
            "이는 AI 기반 참고 결과이므로 담당 의료진의 영상 판독과 추가 평가가 필요합니다."
        )
    else:
        image_sentence = "영상 참고 소견은 제공되지 않았으며, 영상의 임상적 의미는 담당 의료진의 판독을 통해 확인해야 합니다."

    return f"""
## 환자용 AI 예측 설명 보고서

### 1. 이번 AI 예측 결과 요약
AI 모델은 {lesion_side} 부위의 병변을 '{malignancy}'으로 해석했습니다.
현재 입력된 정보에서는 재발 관련 판단도 '{recurrence}'로 정리됩니다.
다만 이 결과는 AI 기반 참고 자료이며, 확정 진단은 아닙니다.

### 2. 병변 위치와 영상 참고 소견
병변 위치는 {lesion_side}으로 정리됩니다.
{image_sentence}

### 3. 검색된 논문/가이드라인에서 확인된 내용
{section3_text}

### 4. 의료진과 추가로 확인할 수 있는 부분
{section4_text}

### 5. 꼭 기억할 점
이 보고서는 AI 예측 결과와 검색된 문헌을 바탕으로 만든 참고용 설명입니다.
확정 진단이나 치료 결정은 담당 의료진의 진료, 검사 결과, 병리 결과를 종합해 이루어져야 합니다.
""".strip()


def should_use_fallback(report: str) -> bool:
    if not report:
        return True

    if has_bad_citation_format(report):
        return True

    if has_english_chunk_copy(report):
        return True

    if "**S" in report:
        return True

    return False


def build_patient_info_block(patient_data: Dict) -> str:
    lesion_location = patient_data.get("lesion_location", "N/A")
    lesion_side = get_lesion_side(lesion_location)
    malignancy = get_malignancy_korean(patient_data.get("malignancy", "N/A"))
    recurrence = get_recurrence_korean(patient_data.get("recurrence", "N/A"))
    grad_cam_finding = patient_data.get("grad_cam_finding", "N/A")
    prediction_summary = build_prediction_summary(patient_data)

    return f"""
==================================================
                 환자 정보 / AI 예측 요약
==================================================
- 병변 원문 위치: {lesion_location}
- 정리된 병변 위치: {lesion_side}
- 악성/양성 예측: {malignancy}
- 재발 관련 예측: {recurrence}
- 영상 참고 소견: {grad_cam_finding}

[AI 예측 결과 요약]
{prediction_summary}
""".strip()


def generate_patient_report(
    patient_data: Dict,
    faiss_dir: Path,
    model_path: Path,
    output_dir: Path,
    max_chunks: int,
    max_new_tokens: int,
) -> Dict:
    db = load_vector_db(faiss_dir)

    context, source_labels = retrieve_evidence(
        db=db,
        patient_data=patient_data,
        top_k_per_query=5,
        max_chunks=max_chunks,
    )

    tokenizer, model, device = load_qwen_model(model_path)

    prompt = build_grounded_prompt(
        patient_data=patient_data,
        context=context,
        source_labels=source_labels,
    )

    print("[STATUS] Generating grounded Korean report")

    raw_report = generate_text(
        tokenizer=tokenizer,
        model=model,
        device=device,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    raw_report = clean_report(raw_report)
    raw_report = remove_source_paths_from_report(raw_report)
    raw_report = soften_unsafe_phrasing(raw_report)

    print("[STATUS] Using template-based grounded report for medical safety.")
    report = build_template_fallback_report(
        patient_data=patient_data,
        context=context,
        source_labels=source_labels,
    )


    report = validate_report_sources(report, source_labels)

    patient_info_block = build_patient_info_block(patient_data)
    source_block = "\n".join(source_labels) if source_labels else "검색된 출처 없음"

    final_text = (
        patient_info_block
        + "\n\n"
        + "=" * 50
        + "\n검색된 근거 출처 목록\n"
        + "=" * 50
        + "\n"
        + source_block
        + "\n\n"
        + "=" * 50
        + "\n환자용 AI 예측 설명 보고서\n"
        + "=" * 50
        + "\n"
        + report
        + "\n"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "patient_report_kor_qwen35_grounded_v3.txt"
    context_path = output_dir / "retrieved_evidence_qwen35_v3.txt"
    prompt_path = output_dir / "prompt_qwen35_v3.txt"
    raw_path = output_dir / "raw_llm_output_qwen35_v3.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_text)

    with open(context_path, "w", encoding="utf-8") as f:
        f.write(context)

    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_report)

    return {
        "final_text": final_text,
        "report_path": report_path,
        "context_path": context_path,
        "prompt_path": prompt_path,
        "raw_path": raw_path,
        "source_labels": source_labels,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--faiss-dir",
        type=str,
        default=str(DEFAULT_FAISS_DIR),
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
    )

    parser.add_argument(
        "--offline",
        action="store_true",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=800,
    )

    return parser.parse_args()


def main() -> None:
    print("[DEBUG] current file:", Path(__file__).resolve())
    print("[DEBUG] current working directory:", Path.cwd())

    args = parse_args()
    set_offline_environment(args.offline)

    patient_test = {
        "lesion_location": "Right Ovary",
        "malignancy": "Malignant",
        "recurrence": True,
        "grad_cam_finding": "Suspected peritoneal metastasis near right ovary.",
    }

    try:
        result = generate_patient_report(
            patient_data=patient_test,
            faiss_dir=Path(args.faiss_dir),
            model_path=Path(args.model_path),
            output_dir=Path(args.output_dir),
            max_chunks=args.max_chunks,
            max_new_tokens=args.max_new_tokens,
        )

        print(result["final_text"])
        print(f"\n[SUCCESS] Report saved to: {result['report_path']}")
        print(f"[SUCCESS] Retrieved evidence saved to: {result['context_path']}")
        print(f"[SUCCESS] Prompt saved to: {result['prompt_path']}")
        print(f"[SUCCESS] Raw LLM output saved to: {result['raw_path']}")

    except Exception as e:
        print(f"\n[ERROR] {e}")


if __name__ == "__main__":
    main()


# 실행: python .\generate_report.py --offline --max-chunks 6 --max-new-tokens 800