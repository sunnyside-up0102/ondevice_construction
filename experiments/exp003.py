"""
exp003: 데이터 증강 + 희소 클래스 이미지 반복 학습 + 클래스 가중치(cls_pw=0.25)
- Ultralytics는 학습 데이터로더의 라벨로 클래스 가중치를 계산하므로,
  반복 후 분포가 가중치에 자동으로 반영된다 (학습 로그의 'Class weights:' 줄로 확인)
- 반복과 가중치가 겹쳐 희소 클래스의 오탐(Precision 하락)이 늘어나는지 exp001, exp002와 비교
- 네 조건: v8n_head / v8n_full / v11n_head / v11n_full
"""
from common import run_experiment

if __name__ == "__main__":  
    run_experiment(exp_name="exp003", use_repeat=True, cls_pw=0.25)
