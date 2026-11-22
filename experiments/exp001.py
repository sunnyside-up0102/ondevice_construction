"""
exp001: 데이터 증강 + 희소 클래스 이미지 반복 학습
- 클래스 가중치 X
- 희소 클래스 기준과 반복 횟수 계산은 common.py의 make_repeat_list 참고
- 네 조건: v8n_head / v8n_full / v11n_head / v11n_full
"""
from common import run_experiment

if __name__ == "__main__":  # 윈도우 학습 데이터로더(worker)가 이 파일을 다시 import해도 학습이 반복되지 않게 함
    run_experiment(exp_name="exp001", use_repeat=True, cls_pw=0.0)
