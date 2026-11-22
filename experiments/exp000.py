"""
exp000: 데이터 증강만 적용한 기준(baseline) 실험
- 희소 클래스 이미지 반복 X, 클래스 가중치 X
- exp001~003의 개선 폭(반복 효과, 가중치 효과)을 판단하기 위한 기준점
- 네 조건: v8n_head / v8n_full / v11n_head / v11n_full
- 공통 설정과 진행 순서는 common.py 참고
"""
from common import run_experiment

if __name__ == "__main__":  # 윈도우 학습 데이터로더(worker)가 이 파일을 다시 import해도 학습이 반복되지 않게 함
    run_experiment(exp_name="exp000", use_repeat=False, cls_pw=0.0)
