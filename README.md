# 멀티모달 의료 데이터를 활용한 난소암 계층적 진단·재발 예후 예측 및 LLM 기반 임상 보고서 생성 AI 모델 개발
Development of an AI Model for Hierarchical Diagnosis and Recurrence Prognosis Prediction of Ovarian Cancer Using Multimodal Medical Data with LLM-Based Clinical Report Generation

## Repository Overview
본 Repository는 의료 영상과 임상 데이터를 통합한 멀티모달 인공지능 모델을 활용하여 난소 종양의 계층적 진단과 재발 예후 예측을 수행하는 연구 프로젝트의 구현 코드와 실험 내용을 담고 있다.
졸업 프로젝트 연구 트랙의 일환으로 수행되었으며, 실제 임상 환경에서 활용 가능한 explainable 의료 AI 모델 개발을 목표로 한다. 

## 연구 주제
**멀티모달 의료 데이터를 활용한 계층적 난소암 진단 및 재발 예후 예측 AI 모델 개발**  
본 연구는 의료 영상(CT) 과 임상 데이터(EHR) 를 통합한 멀티모달 인공지능 모델을 통해
1. 난소 종양의 계층적 진단(양성 → 악성 → 병기) 을 수행하고,
2. 치료 이후 재발 위험 및 예후를 예측
3. 예측 결과에 대한 xAI 및 LLM 기반 임상 보고서 자동 생성을 통한 임상 활용성 강화
---

## 연구 배경
- 난소암은 조기 진단이 어렵고 재발률이 높아 정확한 진단 및 예후 예측이 매우 중요함.
- 기존 의료 AI 연구는 단일 모달(영상 또는 임상 데이터)에 의존하는 경우가 많아 임상 정보의 복합적 맥락을 충분히 반영하지 못함.
- 또한, 높은 예측 성능에도 불구하고 의사결정 근거가 불투명하여 실제 임상 적용에 한계가 존재함.
- 이에 본 연구는 멀티모달 데이터 기반의 계층적 예측 구조와 설명가능한 인공지능(XAI) 기법을 결합하여, 임상적으로 신뢰 가능한 난소암 진단 및 재발 예측 모델을 제안한다.

---

## 문제 정의 (Probelm Definition)
본 연구는 다음의 세 가지 예측 문제를 계층적(hierarchical) 으로 해결한다.
step 1) 난소 종양 양성 vs 악성 분류
step 2) 악성 종양에 대한 병기(Stage) 분류
step 3) 치료 이후 재발 위험 및 예후 예측
이를 통해 실제 임상 의사결정 흐름과 유사한 단계적 판단 구조를 모델에 반영한다.

---

## Dataset
1. [AIHub 난소암 데이터셋](https://www.aihub.or.kr/aihubdata/data/view.do?pageIndex=1&currMenu=115&topMenu=100&srchOptnCnd=OPTNCND001&searchKeyword=%EB%82%9C%EC%86%8C%EC%95%94&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20&aihubDataSe=data&dataSetSn=71727)
2. [MMOTU - Ovarian Ultrasound Images Dataset](https://www.kaggle.com/datasets/orvile/mmotu-ovarian-ultrasound-images-dataset/data)
3. [Ovarian Cancer Risk and Progression Data](https://www.kaggle.com/datasets/datasetengineer/ovarian-cancer-risk-and-progression-data?select=Ovarian_patient_data.csv)
* 현재 Repository에 포함된 실험 코드는 공개 Kaggle 데이터 기반 초기 검증 단계에 해당한다.

---

## 연구 내용
1. **멀티모달 데이터 전처리**
   - 초음파 영상 전처리 : normalization, resizing
   - 임상 데이터 전처리 : 결측치 처리, 범주형 변수 encoding

2. **멀티모달 AI 모델 설계**
   - Image Encoder : ResNet50, ViT, ResNet50/ViT hybrid
   - Text Encoder : MLP
   - Multimodal fusion : Late Fusion, Attention-based fusion
   - output head : 계층적 진단 분류, 재발 위험 및 예후 예측

3. **XAI**
   - Grad-CAM : 초음파/CT 영상에서 진단에 기여한 중요 영역 시각화
   - Attention Map : multimodal fusion 과정에서의 중요 정보 확인
   - Feature Importance : 임상 변수별 예측 기여도 해석

4. **평가 지표**
   - classification 성능
     - Accuracy
     - F1-score
     - Sensitivity / Specificity
   - 재발 및 예후 예측
     - C-index
     - Survival-related metrics

5. **임상적 가치 및 기대 효과**
   - 의료진 : 진단 결과와 함께 제시되는 설명을 통해 신뢰도 높은 임상 판단 가능
   - 환자 : 진단 및 예후 결과에 대한 직관적 이해 가능
   - 연구적 가치
     - multimodal 의료 AI의 실제 임상 적용 가능성 제시
     - 계층적 예측 + XAI 결합 모델의 새로운 연구 사례 제안

---

## 윤리적 고려
- 본 연구는 공개 데이터셋만을 활용하며 개인 식별 정보는 포함하지 않음.

---

## 팀 정보
- **프로젝트 멤버**: 김문경(2376029), 박선영(2376100), 안서연(2276177)  
- **팀 지도 교수**: 황의원 교수님  
- **트랙**: 연구 트랙  

---

## 최종 수정일
2025.12.18
