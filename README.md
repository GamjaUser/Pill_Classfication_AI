# Pill_Classfication_AI (Classification typo)
## Summary
알약 50종을 학습하여 알약 이미지를 테스트 하였을 때 어떤 알약인지 판별해주는 AI 입니다. 뒤섞이거나 집에 돌아다니는 알약이 무엇인지 궁금하다면 사진을 찍어 어떤 알약인지 쉽게 알 수 있도록 하는 취지에서 개발하였습니다.

(※ 팀 프로젝트로, 본 작성자가 맡은 인공지능 파트의 코드만을 공유했습니다. 결과 사진은 팀 프로젝트 최종 결과입니다.)
___
## Base
* OS : WSL2
* Platform : Docker (with anaconda)
* Docker image : pqowie/pytorch_tensorflow [https://hub.docker.com/r/pqowie/pytorch_tensorflow]
___
## Installation
Python 3.11, cudnn 9.3.0 with the following installed :
```
git clone https://github.com/GamjaUser/Pill_Classfication_AI.git
cd Pill_Classfication_AI
pip install -r requirements.txt
```
___
## Dataset
* link
  * AI-Hub (경구약제 이미지 데이터) [https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=&topMenu=&aihubDataSe=data&dataSetSn=576]
* use
  * TL_10_단일.zip l 71.91 MB l key: 66073
  * TS_10_단일.zip l 93.05 GB l key: 66162
* type
  * 미래트리메부틴정 100mg/병
  * 큐레틴정(빌베리건조엑스)
  * 콘택골드캡슐 10mg/PTP
  * 에스케이코스카플러스정
  * 에스케이코스카플러스에프정
  * 클로미딘정 100mg/병
  * 사리돈에이정 250mg/PTP
  * 스트라테라캡슐 25mg
  * 마도파정125
  * 노바스크정 5mg
  * 네비레트정(네비보롤염산염)
  * 대웅알벤다졸정 400mg/PTP
  * 토르셈정 10mg
  * 세비보정(텔비부딘)
  * 마도파정
  * 코다론정(아미오다론염산염)
  * 렉스펜정 300mg/PTP
  * 아프로벨정 150mg
  * 아프로벨정 300mg
  * 로자살탄정 100mg
  * 웰부트린엑스엘정 300mg
  * 모푸렌정(모사프리드시트르산염)
  * 레보펙신정 500mg
  * 캐롤에프정 368.9mg/PTP
  * 디오반필름코팅정 320mg
  * 엑스포지정 5/160mg
  * 미니린멜트설하정 60mcg
  * 미니린멜트설하정 120mcg
  * 리피논정 20mg
  * 리피논정 40mg
  * 플라벤정 500mg/PTP
  * 심바스트씨알정(심바스타틴)
  * 심발타캡슐 30mg
  * 임팩타민정 50mg/PTP
  * 우루사정 300mg
  * 제스판골드정 80mg/PTP
  * 프레미나정 0.3mg
  * 프레미나정 0.625mg
  * 맥시부펜이알정 300mg
  * 익수허브콜캡슐 490mg/포
  * 레보펙신정 250mg
  * 리피토정 80mg
  * 보령모사프리드시트르산염수화물정
  * 휴트라돌정
  * 쎄로켈서방정 400mg
  * 쎄로켈서방정 300mg
  * 쎄로켈서방정 50mg
  * 쎄로켈서방정 200mg
  * 자누메트정 50/500mg
  * 자누메트정 50/1000mg
___
## Model
* EfficientNetB0
  * 구글이 2019년에 발표한 EfficientNet 모델군 중 가장 작은 버전으로, 이미지 분류(Classification) 및 특징 추출(Feature Extraction)에 최적화된 CNN 모델
  * 기존 ResNet, Inception보다 더 적은 연산량으로 더 높은 성능을 냄
  * 복잡한 네트워크 구조 대신 단순한 Scaling 사용
  * EfficientNetB0 ~ B7까지 존재 (EfficientNetB0 이 가장 작은 모델)
___
## Steps
1. download data
2. Run ```python train_pill_model.py``` to train the model
3. Run ```python test_pill_model.py``` to test the model
___
## Results
![image](https://github.com/user-attachments/assets/acc04ec6-c626-43ab-8e78-2a91393d551a)

![image](https://github.com/user-attachments/assets/1db9d60b-1828-4f46-8444-d6ac22b2edcd)

___
## Reference
[1] 약학정보원 [https://www.health.kr/main.asp]

