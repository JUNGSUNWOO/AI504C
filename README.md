# AI504c Midterm Project — MNIST PCA / AE / VAE

- 과목: 머신러닝을 위한 수학 (AI504c)
- 학번: 202628030
- 실험: MNIST 이미지의 PCA, Autoencoder(AE), Variational Autoencoder(VAE) 차원 축소·복원 성능 비교
- 보고서: 별도 제출한 `Midterm Project_202628030_정선우.pdf` (12페이지)

## 1. 제출 파일 구성

```text
project_제출용/
├── README.md
├── data_preprocess.py       # MNIST 다운로드·전처리 및 공통 표본 선택
├── problem1.py             # PCA: Problem 1.1~1.5
├── problem2.py             # AE: Problem 2.1~2.2 및 PCA 비교
├── problem3.py             # VAE: Problem 3.1~3.2
├── compare_latent.py       # 세 모델의 동일 표본·테스트 데이터 비교
├── checkpoints/
│   ├── autoencoder_M2.pth
│   ├── autoencoder_M5.pth
│   ├── autoencoder_M10.pth
│   └── vae_M2.pth
└── result/
    ├── problem1/           # PCA 주성분·복원·오차·잠재공간 그림
    ├── problem2/           # AE 학습곡선·잠재공간 그림
    ├── problem3/           # VAE 잠재공간 그림·학습 기록 CSV
    └── comparison/         # 세 모델 비교 그림·복원오차 CSV
```

`data/`에 저장되는 MNIST 원본 데이터는 제출 파일에 포함하지 않았습니다. 코드 실행 시 자동 다운로드합니다. `__pycache__/`는 Python 실행 중 생성되는 캐시이므로 재현에 필요하지 않습니다.

**참고:** `data_preprocess.py`를 직접 실행하면 표본 인덱스 기록을 **`results/sample_indices.json`** 에 생성합니다. 이 `results/`는 위의 그림 출력용 `result/`와 다른 경로이며, 현재 코드의 동작을 그대로 기재한 것입니다. ZIP에는 해당 JSON이 포함되어 있지 않습니다.

## 2. 실행 환경 및 설치

Python 3 환경에서 실행합니다. 필요한 주요 패키지는 `torch`, `torchvision`, `numpy`, `matplotlib`이며, 표준 라이브러리 `csv`, `pathlib` 등을 사용합니다. PyTorch/torchvision은 운영체제 및 GPU 환경에 맞는 호환 버전을 설치하십시오.

```bash
python -m pip install torch torchvision numpy matplotlib
```

실제 사용 환경의 버전은 다음 명령으로 확인할 수 있습니다. **현재 ZIP에는 실행 당시의 정확한 패키지 버전 기록이 포함되어 있지 않으므로, 버전을 임의로 단정하지 않습니다.**

```bash
python -c "import sys,torch,torchvision,numpy,matplotlib; print('Python:',sys.version); print('torch:',torch.__version__); print('torchvision:',torchvision.__version__); print('numpy:',numpy.__version__); print('matplotlib:',matplotlib.__version__)"
```

최초 MNIST 다운로드 시 인터넷 연결이 필요합니다. `data_preprocess.py`에는 다운로드 오류 회피를 위해 SSL 인증서 검증을 비활성화하는 설정이 포함되어 있으므로, 신뢰할 수 있는 네트워크에서 실행하고 가능하면 정상적인 인증서 검증 설정으로 대체하는 것을 권장합니다.

## 3. 실행 순서

ZIP을 압축 해제한 후 **`project_제출용` 폴더에서** 아래 명령을 순서대로 실행합니다.

```bash
python data_preprocess.py
python problem1.py
python problem2.py
python problem3.py
python compare_latent.py
```

- `data_preprocess.py`: MNIST 데이터 다운로드, 정규화·PCA 표준화, 고정 표본 선택 및 표본 인덱스 JSON 생성.
- `problem1.py`: PCA 고유벡터, M별 복원 이미지, 학습·테스트 복원오차 및 2차원 잠재공간 시각화.
- `problem2.py`: 잠재 차원 M=2, 5, 10의 AE를 각각 학습하고 PCA와 비교. `checkpoints/autoencoder_M*.pth` 저장.
- `problem3.py`: M=2 VAE를 학습하고 학습 기록·그림 및 `checkpoints/vae_M2.pth` 저장.
- `compare_latent.py`: 위 체크포인트를 읽어 M=2에서 PCA·AE·VAE를 비교. **AE/VAE 학습 후 실행해야 합니다.**

`problem1.py`와 `problem2.py`의 그래프 창은 실행 환경에 따라 닫아야 다음 단계로 진행할 수 있습니다. 코드 실행 시 `result/` 하위에 결과가 저장되며, 체크포인트는 재학습 시 갱신될 수 있습니다.

## 4. 데이터 및 실험 설정

- 데이터: MNIST 28×28 회색조 이미지, 학습 60,000장 및 테스트 10,000장.
- 공통 입력: 784차원 행 우선 펼침, 픽셀값 [0,1].
- PCA: 학습 데이터의 픽셀별 평균 및 모집단 표준편차로 표준화; 테스트 데이터에도 학습 통계만 적용. 표준편차 0인 픽셀은 1로 대체. 복원오차는 역표준화한 **원본 픽셀 공간**에서 계산하며, 수치 평가 시 복원값을 클리핑하지 않음.
- AE/VAE: 별도 표준화 없이 [0,1] 픽셀 입력 및 Sigmoid 출력 사용.
- PCA 잠재 차원: M=2, 5, 10, 20, 50, 200.
- AE: 784–128–M–128–784, M=2, 5, 10; ReLU/Sigmoid, Adam(lr=0.001), batch 256, 20 epochs.
- VAE: M=2, H=128; 평균과 로그분산의 별도 출력, reparameterization, Adam(lr=0.001), batch 256, 20 epochs.
- 난수 시드: 42. PCA 1.2 단일 표본 인덱스: **46435**. 2차원 잠재공간 비교는 학습 데이터에서 숫자 0·1·9를 각각 100개씩 동일하게 선정. 표본 선택 방식과 나머지 인덱스는 `data_preprocess.py`에 구현되어 있으며 직접 실행 시 `results/sample_indices.json`에 저장됨.
- 숫자 라벨은 산점도용 표본 선택·색상 표시 등에 사용하며 PCA/AE/VAE의 복원 학습 목표에는 사용하지 않음.

## 5. 평가 기준과 결과

모든 모델의 복원 성능은 이미지별 **784개 픽셀의 제곱오차 합**을 구한 뒤 평가 이미지 전체에 대해 평균한 값으로 비교합니다.

```text
J_M = (1/N) × Σ_n Σ_i (x[n,i] - x_hat[n,i])²
```

M=2, 동일한 MNIST 테스트 이미지 10,000장 기준:

| 모델 | 평균 제곱 L2 복원오차 |
|---|---:|
| PCA | 45.2113 |
| AE | 34.9667 |
| VAE (`decoder(mu)`) | 34.7531 |

**VAE 주의사항:** 비교표의 VAE 복원오차는 인코더가 출력한 잠재 평균 `mu`를 디코더에 넣은 **결정론적 복원 결과**입니다. VAE 학습 중 샘플링한 `z`로 계산하는 기대 복원오차와 동일한 수치는 아닙니다.

VAE 학습 손실은 이미지별 복원 제곱오차 합과 KL 발산을 합산하고 배치 평균을 취합니다. KL 항은 2개 잠재 차원에 대해 합산하며, 가중치 β=1입니다. **KL 항이 포함된 VAE 총 학습손실을 PCA/AE의 복원오차와 직접 비교하지 않습니다.**

## 6. 결과 파일 위치

- `result/problem1/`: PCA 첫 5개 주성분, 단일 이미지의 주성분 기여도 및 복원오차, M별 복원 이미지, 학습/테스트 J_M 그래프, PCA 잠재공간.
- `result/problem2/`: AE M=2 학습/테스트 곡선 및 AE 잠재공간. PCA와 AE의 M별 복원오차는 코드 실행 출력 및 보고서에 정리.
- `result/problem3/`: VAE 잠재 평균 산점도, 학습 손실 기록 CSV. VAE 손실 곡선은 코드 재실행으로 생성 가능.
- `result/comparison/comparison_latent.png`: 동일한 숫자별 100개 학습 표본으로 PCA·AE·VAE 2D 잠재공간 비교.
- `result/comparison/comparison_test_reconstruction.png`: 테스트 10,000장 M=2 평균 복원오차 비교.
- `result/comparison/comparison_reconstruction_examples.png`: 동일한 테스트 이미지 5장의 복원 예시.
- `result/comparison/comparison_metrics.csv`: 비교 지표 및 VAE 평가 방식 기록.

## 7. 재현성 및 AI 활용

동일한 MNIST 분할·표본 선택·전처리·평가식을 사용하며, 난수 시드를 42로 설정하였습니다. 다만 GPU/라이브러리 버전 등 실행 환경에 따라 신경망 학습 결과가 소폭 달라질 수 있습니다. 제출된 `.pth` 파일은 보고서의 학습 결과를 재확인하기 위한 체크포인트입니다.

Python 코드 구현, 오류 수정 및 보고서 초안 작성에 생성형 AI(ChatGPT)를 활용했습니다. 제안된 코드를 직접 실행하여 정상 동작을 확인하고, 학습 손실 추이·이미지 복원·잠재공간 시각화 및 공통 평가식에 따른 복원오차 비교로 결과를 검토했습니다.

데이터 출처: MNIST (LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, 1998), `torchvision.datasets.MNIST`를 통해 로드.
