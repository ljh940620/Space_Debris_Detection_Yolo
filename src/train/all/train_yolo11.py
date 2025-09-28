""" 1. 라이브러리 불러오기 - "YOLO 전문가 초빙하기"
ultralytics 라이브러리에서 YOLO라는 이름의 전문가(핵심 부품, 클래스)를 초빙합니다. 이 전문가 YOLO는 모델을 만들고, 훈련시키고, 예측하는 모든 방법을 알고 있습니다. 이 한 줄로 모든 준비가 끝납니다. """
import os
from ultralytics import YOLO

""" 2. 모델 로드 - "선수 선발 및 기본 교육"
YOLO('yolov11n.pt'): YOLO 전문가에게 'YOLOv11' 모델 중에서 가장 작고 빠른 'nano' 버전인 yolov11n.pt 선수를 선발하라고 지시합니다.
여기서 중요한 점은, 이 모델이 완전한 신입이 아니라는 것입니다. 수많은 일반 이미지(사람, 자동차, 동물 등)를 통해 '세상의 사물이 어떻게 생겼는지'를 이미 학습한 사전 훈련된(pre-trained) 모델입니다.
우리는 이 똑똑한 선수를 데려와서 '우주 쓰레기'만 전문적으로 알아보도록 추가 훈련시키는, **전이 학습(Transfer Learning)**을 사용할 것입니다. 훨씬 적은 데이터로도 빠르고 좋은 성능을 낼 수 있는 비결이죠. """

# --- 1. 경로 설정 ---
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(CURRENT_FILE_PATH, '..', '..', '..')

# --- 2. 모델 로드 ('s' 모델로 변경) ---
model_path = os.path.join(PROJECT_ROOT, 'model', 'yolov11n.pt')
#model_path = os.path.join(PROJECT_ROOT, 'model', 'yolov11s.pt') # yolov11n.pt -> yolov11s.pt
model = YOLO(model_path)


""" 3. 데이터 설정 - "학습 자료(시간표) 지정하기"
모델에게 "어떤 데이터로 공부해야 하는지" 알려주는 '설정 지도(data.yaml)' 파일의 위치를 yaml_path라는 변수에 저장합니다.
여기서는 파일 이름만 적었는데, 이것은 이 스크립트(train_yolo.py)와 data_hazard_only.yaml 파일이 같은 폴더에 있을 때 사용 가능한 상대 경로 방식입니다. """
# --- 3. 데이터 설정 ---
yaml_path = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2', 'data.yaml') # 운석 클래스 D2 추가

""" 4. 모델 학습 - "본격적인 훈련 캠프 시작!"
if __name__ == '__main__':: 이 스크립트가 메인 프로그램으로 실행될 때만 아래의 훈련 코드를 작동시키라는 의미입니다. 항상 써주는 파이썬의 관용구라고 생각하시면 편합니다.
results = model.train(...): 드디어 model 선수에게 훈련을 시작하라고 명령하는 부분입니다. 괄호 안의 설정값들이 바로 훈련의 방식과 조건을 결정하는 '훈련 메뉴얼'입니다.
data=yaml_path: "학습에 사용할 데이터 정보는 yaml_path 변수에 지정된 (data_hazard_only.yaml) 파일을 참고해라."
epochs=50: "전체 훈련 데이터를 처음부터 끝까지 총 50번 반복해서 학습해라." (훈련 강도)
imgsz=640: "모든 이미지를 640x640 픽셀 크기로 통일해서 학습해라." (학습할 이미지 크기)
batch=8: "한 번에 8장의 이미지를 묶어서 모델에게 보여주고 학습시켜라." (한 번에 푸는 문제집 페이지 수. GPU 메모리가 부족하면 이 숫자를 4나 2로 줄여야 합니다.)
name='hazard_yolov11_hazard_only_run': "이번 훈련의 모든 결과물(성적표, 성장일지, 최종 결과물 등)은 runs/detect/ 폴더 아래에 hazard_yolov11_hazard_only_run 이라는 이름의 폴더를 만들어 저장해라." """
# 3. 모델 학습 시작!

# [훈련 가속]
# RTX 4060은 더 빠른 연산을 위해 'FP16' (혼합 정밀도)을 지원합니다.
# 'half=True' 옵션을 추가하면 VRAM을 덜 사용하고 훈련 속도가 빨라질 수 있습니다.
use_half_precision = True

if __name__ == '__main__':
    results = model.train(
        data = yaml_path,
        epochs = 50,
        imgsz = 640,
        #batch = 8, # 's' 모델은 'n'보다 크므로 batch를 16->8로 줄여서 시작하는 것이 안전합니다.
        
        # [해결책 1] batch 크기 늘리기 (가장 효과적)
        # VRAM이 넉넉하므로, CPU가 한 번에 8장 대신 16장(또는 32장)을 준비하게 합니다.
        # GPU도 한 번에 더 많은 양을 받아 더 오래 일하게 됩니다.
        batch = 16,  # 8 -> 16 (32까지도 시도해볼 만합니다)

        # [해결책 2] workers 수 늘리기
        # 데이터를 불러오는 CPU 일꾼(스레드) 수를 늘려 병목을 줄입니다.
        #workers = 12, # (CPU 코어 수에 맞춰 현재 사용하는 pc의 CPU는 AMD Ryzen 5 5600G로, 6코어 12스레드(worker))

        # [최적화 3: 핵심] 48GB RAM에 데이터셋 캐시
        # CPU 병목 현상을 획기적으로 줄여줍니다.
        #cache = True,

        name = 'yolo11_hazard_all',
        #name = 'yolo11_hazard_all_optimization',
        #half=use_half_precision  # FP16 사용으로 속도 향상 (권장)
    )
    """ 5. 완료 메시지 - "훈련 캠프 종료 알림"
    50번의 epoch 훈련이 모두 끝나면, 터미널에 이 메시지를 출력하여 모든 과정이 성공적으로 끝났음을 알려줍니다.
    한 문장으로 요약하자면, 이 스크립트는 "똑똑한 YOLOv11n 학생을 데려와서(YOLO(...)), data.yaml이라는 시간표를 주고(data=...), 50일 동안(epochs=50) 8쪽씩(batch=8) 공부시켜서, 그 결과를 hazard_yolov11... 라는 이름의 노트에 기록하는" 과정이라고 할 수 있습니다. """
    print("모델 학습이 완료되었습니다!")
