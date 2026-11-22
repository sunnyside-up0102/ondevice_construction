"""
exp002: 데이터 증강 + 클래스 가중치(cls_pw)
- 희소 클래스 이미지를 별도로 반복하지 않음
- cls_pw=0.25: 클래스 가중치 = (1/학습 박스 수)^0.25 를 평균 1로 정규화 → 분류 손실(BCE)에 곱함
  0.25는 초기 후보값이며 확정된 최적값은 아님 (원본 분포 기준 가중치 약 0.78~1.32)
- 네 조건: v8n_head / v8n_full / v11n_head / v11n_full
"""
from common import run_experiment

if __name__ == "__main__":  # 윈도우 학습 데이터로더(worker)가 이 파일을 다시 import해도 학습이 반복되지 않게 함
    run_experiment(exp_name="exp002", use_repeat=False, cls_pw=0.25)
