import os
import cv2
import yaml
from tqdm import tqdm

# --- 1. 경로 및 설정 ---

# [수정됨]
# 이 스크립트(src/visualize_labels.py)의 위치를 기준으로 상위 폴더(프로젝트 루트)를 찾습니다.
# os.path.dirname()을 두 번 사용하여 한 단계 위 폴더로 이동합니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 데이터셋 관련 경로 (이하 코드는 수정할 필요 없음)
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATASET_DIR, 'test', 'labels')

# 시각화된 결과 이미지를 저장할 폴더
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'visualization_results_labels')

# 클래스별 색상 지정을 위한 색상 리스트 (BGR 순서)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255)
]

# --- 2. 클래스 이름 불러오기 ---
try:
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        CLASS_NAMES = data_yaml['names']
        print(f"총 {len(CLASS_NAMES)}개의 클래스를 찾았습니다: {CLASS_NAMES}")
except Exception as e:
    print(f"오류: {YAML_PATH} 파일을 읽을 수 없습니다. 경로를 확인하세요.")
    print(e)
    exit()

# --- 3. 메인 시각화 로직 ---
def visualize_and_save():
    print(f"\n라벨 시각화를 시작합니다. 결과는 '{OUTPUT_DIR}' 폴더에 저장됩니다.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    label_files = [f for f in os.listdir(TEST_LABELS_DIR) if f.endswith('.txt')]

    if not label_files:
        print(f"경고: '{TEST_LABELS_DIR}' 폴더에 라벨 파일(.txt)이 없습니다.")
        return

    # tqdm을 사용하여 진행 상황 표시
    for label_file in tqdm(label_files, desc="라벨 파일 처리 중"):
        base_name = os.path.splitext(label_file)[0]
        
        # 해당 라벨에 맞는 이미지 파일 찾기 (다양한 확장자 고려)
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = os.path.join(TEST_IMAGES_DIR, base_name + ext)
            if os.path.exists(potential_path):
                image_path = potential_path
                break
        
        if image_path is None:
            continue

        # 이미지 불러오기
        image = cv2.imread(image_path)
        if image is None:
            continue
            
        h, w, _ = image.shape
        
        # 라벨 파일 읽고 처리하기
        label_path = os.path.join(TEST_LABELS_DIR, label_file)
        classes_in_image = set()

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                
                class_id, x_center, y_center, width, height = map(float, parts)
                class_id = int(class_id)
                
                box_w = width * w
                box_h = height * h
                x1 = int((x_center * w) - (box_w / 2))
                y1 = int((y_center * h) - (box_h / 2))
                x2 = int(x1 + box_w)
                y2 = int(y1 + box_h)

                class_name = CLASS_NAMES[class_id]
                classes_in_image.add(class_name)
                color = COLORS[class_id % len(COLORS)]
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                label_text = f"{class_name}"
                (text_w, text_h), _ = cv2.getTextSize