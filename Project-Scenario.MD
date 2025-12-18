
| 항목 | 내용 |
|------|------|
| **프로젝트명** | 멀티모달 의료 데이터를 활용한 난소암 계층적 진단·재발 예후 예측 및 LLM 기반 임상 보고서 생성 AI 모델 개발 |
| **프로젝트 키워드** | 난소암, 난소암 진단, 재발 예측, Medical data pre-processing, Multi-modal, Transfer learning, Medical AI, Explainable AI, LLM |
| **트랙** | 연구 트랙 |
| **프로젝트 멤버** | 김문경, 박선영, 안서연 |
| **팀 지도 교수** | 황의원 교수님 |
| **무엇을 만들고자 하는가** |  본 연구는 초음파·CT·임상 정보를 통합한 멀티모달 AI를 개발하여 난소암 병변 탐지 및 계층적 진단(양성/악성·세부 subtype), 재발 위험도 예측까지 자동화하고, LLM을 활용해 의사의 진단 보조를 위한 구조화된 임상 보고서를 생성하는 시스템을 구축하고자 한다.  |
| **고객** | 의료기관, 산부인과 의사, 영상의학과 의사, 환자(일반인) |
| **Pain Point** | 난소 종양은 흔히 발견되지만, 악성 여부와 subtype에 따라 치료 전략과 예후가 크게 달라지기 때문에 정확한 진단이 중요하다. 초음파는 1차 진단 도구로 필수적이지만 검사자의 숙련도에 따라 정확도 차이가 크며, CT 역시 영상 판독 경험의 차이로 인해 악성 위험 추정의 변동성이 존재한다. 이 때문에 진단의 일관성이 낮고, 전문 인력 부족 지역에서는 적절한 판독과 의뢰가 이루어지지 못해 오진·진단 지연·불필요한 수술이 발생하고 있다. 특히 난소암은 치료 후에도 높은 재발률을 보이는 암종으로, 현재 임상에서는 정량적이고 표준화된 재발 예측 도구가 부족하여 수술·항암 전략을 결정하는 데 어려움이 크다. 따라서 본 연구를 통해, 임상에서 멀티모달 정보를 일관되게 결합해 계층적 진단과 재발 예측을 수행하며, 의사의 진단 보조를 위한 임상 보고서를 자동으로 제시해줄 수 있는 AI 시스템을 구축하고자 한다.|
| **사용할 소프트웨어 패키지의 명칭과 핵심기능/용도, 사용시나리오** | **Python** : 프로젝트 전반의 데이터 처리, 모델 구현 및 실험 관리를 위한 기본 프로그래밍 언어로 사용<br>**Pytorch** : 딥러닝 모델 개발 및 학습 프레임워크<br>**Grad-CAM** : 딥러닝 모델 설명 가능성(XAI) 지원, 초음파 또는 CT 영상 입력에 대해 모델이 주목한 영역을 heatmap 형태로 시각화하여, 악성 판단의 근거를 직관적으로 제시<br>**Matplotlib**: 학습 과정 시각화 및 모델 성능 모니터링<br>**Numpy, Pandas** : 영상 및 임상 데이터의 전처리, 배열 연산, 결측치 처리 및 데이터 구조 관리<br>**Scikit-learn** : 데이터셋 분할과 함께 Accuracy, F1-score, ROC-AUC 등 모델 성능 평가 지표를 계산하는 데 활용<br>**Google Colab** : GPU 환경에서의 모델 학습 및 실험을 수행하기 위한 클라우드 기반 개발 환경으로 활용<br>**ChatGPT** : 멀티모달 AI 모델의 예측 결과와 설명 정보를 입력으로 받아, 자연어 기반의 임상 요약 보고서를 자동 생성하는 데 활용|
| **사용할 소프트웨어 패키지의 명칭과 URL** | [**Python**](https://www.python.org/)<br>[**Pytorch**](https://pytorch.org/)<br>[**Grad-CAM**](https://github.com/ ramprs/grad-cam)<br>[**Matplotlib**](https://matplotlib.org/)<br>[**Numpy**](https://numpy.org/)<br>[**jupyter**](https://jupyter.org/)<br>[**python**](www.python.org)<br>[**numpy**](https://numpy.org/)<br>[**pandas**](https://pandas.pydata.org/)<br>[**Scikit-learn**](https://scikit-learn.org/stable/index.html)<br>[**Google Colab**](https://colab.google/notebooks/)<br>[**ChatGPT**](https://chatgpt.com/)|
| **팀 그라운드룰** | [GroundRule](https://github.com/kmk0105-S2/CapstoneDesign-25/blob/main/GroudRule.MD) |
| **최종 수정일** | 2025.12.18. |

