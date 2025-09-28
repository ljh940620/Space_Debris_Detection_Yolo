import os
from ultralytics import YOLO

"""
훈련된 최종 모델(best.pt)을 사용하여
테스트 데이터셋(test set)으로 성능을 평가하는 스크립트입니다.
"""
# --- 1. 경로 설정 ---
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_FILE_PATH, '..', '..', '..')

# [중요] 평가할 모델의 경로를 정확하게 지정해주세요.
# 이전에 훈련한 결과 폴더('yolo11_hazard_run5' 등) 이름을 확인하고 수정해야 합니다.
#MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo8_hazard_all', 'weights', 'best.pt')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo11_hazard_all', 'weights', 'best.pt')

# test 경로가 포함된 data.yaml 파일의 경로를 지정합니다.
# (이전 훈련과 동일한 데이터셋 구성을 사용한다고 가정)
YAML_PATH = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2', 'data.yaml')

# --- 2. 메인 평가 로직 ---
def evaluate_model():
    # 모델 파일이 존재하는지 확인
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("MODEL_PATH 변수의 경로가 올바른지 확인해주세요.")
        return

    # YAML 파일이 존재하는지 확인
    if not os.path.exists(YAML_PATH):
        print(f"[오류] YAML 파일을 찾을 수 없습니다: {YAML_PATH}")
        print("YAML_PATH 변수의 경로가 올바른지 확인해주세요.")
        return

    print(f"모델 로드 중: {MODEL_PATH}")
    # 1. 훈련된 'best.pt' 모델 로드
    model = YOLO(MODEL_PATH)

    print("\n테스트 데이터셋으로 최종 성능 평가를 시작합니다...")
    
    # --- YOLOv11 모델 최종 평가 ---
    #print("--- [YOLOv8n 최종 성능 평가 시작] ---")
    print("--- [YOLOv11n 최종 성능 평가 시작] ---")
    results = model.val(
        data=YAML_PATH,
        split='test',  # 'val'이 아닌 'test' 데이터셋으로 평가하도록 지정!
        project='evaluate_results',
        #name='all_classes_yolo8'
        name='all_classes_yolo11'
    )
   
    print("\n--- 최종 성능 평가 결과 (mAP) ---")
    # mAP 점수 등 성능 지표가 여기에 표시됩니다.
    # (results 객체에 모든 정보가 들어있습니다)


if __name__ == '__main__':
    evaluate_model()

