"""
exp004: exp001에서 확정한 모델(v8n_full + 희소 클래스 이미지 반복)을 100 epoch로 학습
- exp001 v8n_full의 best epoch가 49/50으로 학습이 덜 끝난 상태였음 → epoch만 50 → 100으로 늘림
- 나머지(데이터, 반복 목록, 증강, 시드, 검증)는 exp001과 동일 → 결과 차이는 epoch 효과
- 학습률은 epoch 수에 맞춰 줄어들고(lrf), 마지막 10 epoch는 모자이크를 끔(close_mosaic)
"""
import common
from common import run_experiment

if __name__ == "__main__":  # 윈도우 학습 데이터로더(worker)가 이 파일을 다시 import해도 학습이 반복되지 않게 함
    common.EPOCHS = 100
    common.MODEL_CONDITIONS = {"v8n_full": common.MODEL_CONDITIONS["v8n_full"]}
    run_experiment(exp_name="exp004", use_repeat=True, cls_pw=0.0)
