# --- 1. 필요한 라이브러리 불러오기 ---
# (os, torch, YOLO, traceback)
import os
import torch
from ultralytics import YOLO
import traceback
import time

""" 
두개의 다른 YOLO 나노 모델 (yolov11n, yolov8n)을 
동일한 설정으로 순차적으로 훈련시키는 스크립트입니다.
(상세 오류 진단 기능 추가)
(훈련 시간 기록 추가)
"""

# --- 2. 경로 및 설정 정의 ---
# (현재 파일 경로, 프로젝트 루트 경로, 데이터셋 경로, )
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_PATH = os.path.join(CURRENT_FILE_PATH, '..', '..', '..')
DATASET_PATH = os.path.join(PROJECT_ROOT_PATH, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_PATH, 'data.yaml')

# (훈련시킬 모델 목록 딕셔너리 만들기 {'결과 폴더명': '모델 파일 경로'})
MODELS_TO_TRAIN = {
    #'yolov8n_run_all': os.path.join(PROJECT_ROOT_PATH, 'model', 'yolov8n.pt'),
    'yolov11n_run_all': os.path.join(PROJECT_ROOT_PATH, 'model', 'yolov11n.pt')
}

# (공통 훈련 설정 딕셔너리 만들기 {'설정 목록': 설정값})
TRAIN_CONFIG = {
    'data': YAML_PATH,
    'epochs': 50,
    'imgsz': 640,
    'batch': 16
}
# --- 3. 메인 훈련 로직 함수 정의 ---
# def main():
def main():
    # GPU 체크 먼저 하고
    if not torch.cuda.is_available():
        print("경고: CUDA GPU를 사용할 수 없습니다. CPU로 훈련을 시작합니다.")

    print(f"총 {len(MODELS_TO_TRAIN)}개의 모델에 대한 훈련을 시작합니다.")    
    
    # 모델 목록을 반복문으로 돌리기
    for run_name, model_path in MODELS_TO_TRAIN.items():
        # 훈련 시작하고 사용 모델 알려주고
        print("\n" + "="*50)
        print(f"--- 훈련 시작: {run_name} ---")
        print(f"사용 모델: {model_path}")
        print("="*50)

        # try-except로 에러 대비하기
        try:
            # 훈련 시작 시간 기록
            start_time = time.time()
            # YOLO 모델 객체 만들고
            model = YOLO(model_path)
            # 1. 모델 훈련: train() 함수 호출해서 훈련 시작
            model.train(
                **TRAIN_CONFIG, # **는 Dictionary Unpacking : 딕셔너리 안에 있는 모든 키-값 쌍을 함수의 인자(argument)로 풀어헤져서 전달하는 역할
                name=run_name
            )
            # 훈련 끝나면 완료 메시지 출력
            print(f"--- 훈련 완료: {run_name} ---")
            # 훈련 종료 시간 기록 및 소요 시간 계산/출력
            end_time = time.time()
            elapsed_time = end_time - start_time
            # 보기 좋게 분/초 단위로 변환
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            print(f"총 소요 시간: {minutes}분 {seconds}초")
            print(f"결과는 'runs/detect/{run_name}' 폴더에 저장되었습니다.")

        # except 부분:
        except Exception as e:
            # 에러 났다고 메시지 출력하고
            print(f"[오류] '{run_name}' 모델 훈련 중 예상치 못한 오류가 발생했습니다.")
            # traceback으로 상세 내용 보여주기
            print("--- TRACEBACK START ---")
            traceback.print_exc()
            print("--- TRACEBACK END ---")
            continue
        
# --- 4. 스크립트 실행 ---
if __name__ == '__main__':
    main()
