import os
import torch
from ultralytics import YOLO
import traceback # <-- [추가] 오류의 상세 내용을 보기 위해 traceback 모듈을 불러옵니다.

""" 
두개의 다른 YOLO 나노 모델 (yolov11n, yolov8n)을 
동일한 설정으로 순차적으로 훈련시키는 스크립트입니다.
(상세 오류 진단 기능 추가 버전)
"""

# --- 1. 경로 및 설정 ---
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
#PROJECT_ROOT = os.path.join(CURRENT_FILE_PATH, '..', '..', '..')

# 훈련할 모델 목록 정의
MODELS_TO_TRAIN = {
    #'yolov8n_run_all': 'yolov8n.pt',
    'yolov8n_run_all': os.path.join(PROJECT_ROOT, 'model', 'yolov8n.pt'),
    'yolov11n_run_all': os.path.join(PROJECT_ROOT, 'model', 'yolov11n.pt')
}

DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')

# --- 2. 공통 훈련 설정 ---
TRAIN_CONFIG = {
    'data': YAML_PATH,
    'epochs': 50,
    'imgsz': 640,
    'batch': 16
}

# --- 3. 메인 훈련 로직 ---
def main():
    if not torch.cuda.is_available():
        print("경고: CUDA GPU를 사용할 수 없습니다. CPU로 훈련을 시작합니다.")
    
    print(f"총 {len(MODELS_TO_TRAIN)}개의 모델에 대한 훈련을 시작합니다.")

    for run_name, model_path_or_name in MODELS_TO_TRAIN.items():
        print("\n" + "="*50)
        print(f"--- 훈련 시작: {run_name} ---")
        print(f"사용 모델: {model_path_or_name}")
        print("="*50)

        try:
            model = YOLO(model_path_or_name)
            model.train(
                **TRAIN_CONFIG,
                name=run_name
            )
            print(f"--- 훈련 완료: {run_name} ---")
            print(f"결과는 'runs/detect/{run_name}' 폴더에 저장되었습니다.")
        
        except Exception as e:
            print(f"[오류] '{run_name}' 모델 훈련 중 예상치 못한 오류가 발생했습니다.")
            # 자세한 오류 내용을 확인하기 위해 traceback을 출력합니다.
            print("--- TRACEBACK START ---")
            traceback.print_exc()
            print("--- TRACEBACK END ---")
            continue

    print("\n\n 모든 모델의 훈련이 완료되었습니다.")

if __name__ == '__main__':
    main()