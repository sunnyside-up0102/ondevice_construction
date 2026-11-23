# 작업자 개인보호구(PPE) 착용 상태 자동 점검 시스템

Raspberry Pi 5 + DEEPX M1 NPU에서 작업자의 보호구 착용/미착용을 실시간으로 탐지하는 온디바이스 AI 프로젝트입니다.

- 팀원: 곽도혁(2021124006), 박경훈(2021128012)
- 과목: 온디바이스 AI 기말 프로젝트

## 데이터셋

- Ultralytics **Construction-PPE** 데이터셋에서 forklift 이미지(247장)를 제외하고 사용
- 753장, 11개 클래스 / train 599장(박스 9,001개), val 154장(박스 2,247개)
- 클래스: `helmet` `gloves` `vest` `boots` `goggles` `Person` / `no_helmet` `no_goggle` `no_gloves` `no_boots` `no_vest`
- 희소 클래스 판단, 반복 횟수, 클래스 가중치는 **train 통계만** 사용하고, val은 모든 실험에서 그대로 사용

## 실험

모든 실험은 같은 데이터 분할, 입력 크기(640), 증강, 시드(0), 검증 방법을 사용하고, 조건마다 원래 사전학습 가중치에서 새로 시작했습니다.

| 실험   | 내용                                        |
| ------ | ------------------------------------------- |
| exp000 | 데이터 증강만 (기준)                        |
| exp001 | 증강 + 희소 클래스 이미지 반복              |
| exp002 | 증강 + 클래스 가중치(cls_pw=0.25)           |
| exp003 | 증강 + 반복 + 클래스 가중치                 |
| exp004 | exp001의 v8n_full을 100 epoch로 학습 (최종) |

exp000~003은 각각 4개 모델 조건(YOLOv8n / YOLO11n × 헤드만 학습 / 전체 미세조정)으로 50 epoch 학습했습니다.

**데이터 증강**: Ultralytics 기본값을 그대로 사용했습니다. 좌우 반전(0.5), 색상·밝기 변화(HSV), 크기·위치 변화(scale, translate), 모자이크를 쓰고, 상하 반전과 회전은 쓰지 않았습니다. 뒤집힌 작업자는 현실에 없고, 회전하면 정답 박스가 부정확해지기 때문입니다.

**희소 클래스 이미지 반복**: 박스 수가 클래스 평균(818개)의 50% 미만인 클래스를 희소 클래스로 보고, `round(sqrt(평균/박스 수))`만큼 그 클래스가 있는 이미지를 반복했습니다. 그 결과 `no_goggle`과 `no_vest`가 2회 반복되어 학습 이미지가 599장에서 884장이 되었습니다. 이미지를 복사하지 않고, 이미지 경로 목록(txt)에 같은 경로를 여러 줄 넣는 방식입니다.

## 결과 (검증셋 mAP50)

| 조건        | exp000 | exp001 | exp002 | exp003 |
| ----------- | ------ | ------ | ------ | ------ |
| v8n 헤드만  | 0.653  | 0.670  | 0.657  | 0.675  |
| v8n 전체    | 0.767  | 0.783  | 0.764  | 0.785  |
| v11n 헤드만 | 0.599  | 0.611  | 0.597  | 0.614  |
| v11n 전체   | 0.773  | 0.778  | 0.778  | 0.768  |

- **전체 미세조정**이 헤드만 학습보다 0.11~0.18 높음
- 전체 미세조정에서 **YOLOv8n과 YOLO11n은 거의 같음** (차이 0.02 미만)
- **반복 학습**은 가장 희소한 `no_goggle`을 크게 개선 (v8n 전체 기준 AP50-95 0.100 → 0.136, F1 0.355 → 0.485)
- 클래스 가중치(cls_pw=0.25)는 효과가 거의 없음
- 100 epoch(exp004)는 73 epoch에서 성능이 멈추고 mAP50이 +0.003만 올라, 한계는 학습 횟수보다 데이터(작은 객체, 라벨 불일치)에 있음

## 최종 모델: exp004 (YOLOv8n 전체 미세조정 + 희소 클래스 반복, 100 epoch)

| mAP50 | mAP50-95 | best epoch |
| ----- | -------- | ---------- |
| 0.786 | 0.546    | 73         |

YOLOv8n을 고른 이유는 정확도가 YOLO11n과 같으면서 구조가 단순해 NPU(INT8) 변환에 유리하다고 판단했기 때문입니다.

## 실행 방법

최종 모델만 다시 학습하려면 `exp004.py`만 실행하면 됩니다.

1. 설치

   ```bash
   pip install -r requirements.txt
   ```

   GPU를 쓰려면 PyTorch CUDA 버전을 먼저 설치합니다. (예: `pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121`)

2. 데이터 준비: 데이터셋은 용량 때문에 [Releases](https://github.com/sunnyside-up0102/ondevice_construction/releases)의 `dataset-v1`에 zip으로 올려 두었습니다. 받아서 프로젝트 폴더에 압축을 풀면 아래 구조가 됩니다.

   ```
   construction-ppe-ultralytics-11/
   ├── data.yaml
   ├── images/train, images/val
   └── labels/train, labels/val
   ```

3. 학습
   ```bash
   python experiments/exp004.py
   ```
   결과는 `runs/exp004/`에 저장됩니다. (RTX 3080 기준 약 52분)

학습한 최종 모델은 `runs/exp004/v8n_full/weights/best.pt`에 포함되어 있어, 다시 학습하지 않고 바로 평가할 수 있습니다.

## 팀 영상으로 평가하기

최종 모델(best.pt)을 팀이 찍은 영상 프레임으로 평가합니다. 코드 없이 Ultralytics 명령어로 실행합니다.

1. 폴더 구성: 프레임 단위가 아니라 영상 단위로 train/val/test를 나눕니다. (이웃 프레임은 거의 같은 사진이라 섞으면 점수가 부풀려짐)
   ```
   team_video/
   ├── data.yaml
   ├── images/train, images/val, images/test
   └── labels/train, labels/val, labels/test
   ```

2. `team_video/data.yaml`: 클래스 번호와 순서는 원본 데이터셋과 반드시 같아야 합니다.
   ```yaml
   path: team_video
   train: images/train
   val: images/val
   test: images/test
   names:
     0: helmet
     1: gloves
     2: vest
     3: boots
     4: goggles
     5: Person
     6: no_helmet
     7: no_goggle
     8: no_gloves
     9: no_boots
     10: no_vest
   ```

3. 평가 (라벨 필요, mAP와 클래스별 결과 출력)
   ```bash
   yolo val model=runs/exp004/v8n_full/weights/best.pt data=team_video/data.yaml split=test imgsz=640
   ```
   test는 미세조정 전(exp004)과 후를 같은 기준으로 비교하는 데 사용합니다.

4. 예측 결과 보기 (라벨 없이도 가능, mp4 영상도 가능)
   ```bash
   yolo predict model=runs/exp004/v8n_full/weights/best.pt source=team_video/images/test imgsz=640
   ```
   결과는 `runs/detect/`에 저장됩니다.

## 폴더 구조

```
experiments/
├── common.py        # 공통 코드 (데이터 확인, 반복 목록, 증강 확인, 학습, 고정 확인, 검증)
├── exp000.py ~ exp003.py
└── exp004.py        # 최종 모델
runs/expXXX/
├── summary.csv          # 조건별 mAP, 학습 시간, optimizer, 백본 고정 여부
├── class_metrics.csv    # 클래스별 P, R, F1, AP
└── {조건}/results.csv   # epoch별 loss, mAP
```

데이터셋은 Releases에 있고, 가중치는 최종 모델(`runs/exp004/v8n_full/weights/best.pt`)만 저장소에 포함했습니다.

## 환경

- 학습: Ubuntu, RTX 3080, PyTorch 2.5.1+cu121, Ultralytics 8.4.216
- 배포: Raspberry Pi 5 + DEEPX M1 NPU (진행 예정)
