import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import random

# --- (1. 경로 설정, 2. 클래스 이름 불러오기 등은 이전과 동일) ---
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo8_hazard_all', 'weights', 'best.pt')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATASET_DIR, 'test', 'labels')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'prediction__results', 'results')

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255)
]

try:
    with open(YAML_PATH, 'r', encoding='utf-8') as f:
        data_yaml = yaml.safe_load(f)
        CLASS_NAMES = data_yaml['names']
        print(f"총 {len(CLASS_NAMES)}개의 클래스를 찾았습니다: {CLASS_NAMES}")
except Exception as e:
    print(f"오류: {YAML_PATH} 파일을 읽을 수 없습니다. 경로를 확인하세요.")
    exit()

# --- 3. 메인 비교 로직 ---
def compare_predictions():
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print("모델 로드 완료.")

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"\n총 {len(image_files)}개의 테스트 이미지에 대한 비교를 시작합니다.")
    print(f"결과는 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

    # ############################################################# #
    # ############## [수정 1: 클래스별 카운터 초기화] ################# #
    # ############################################################# #
    # 각 클래스 이름으로 카운터를 만들고 0으로 설정
    class_counters = {name: 0 for name in CLASS_NAMES}
    no_label_counter = 0
    # ############################################################# #

    results_generator = model.predict(source=TEST_IMAGES_DIR, batch=16, stream=True, verbose=False)

    for results in tqdm(results_generator, total=len(image_files), desc="이미지 처리 중"):
        prediction_image = results.plot()
        
        image_path = results.path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(TEST_LABELS_DIR, base_name + '.txt')

        original_image = cv2.imread(image_path)
        h, w, _ = original_image.shape
        ground_truth_image = original_image.copy()
        
        classes_in_image = set()

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id, x_c, y_c, box_w, box_h = map(float, parts)
                    class_id = int(class_id)
                    
                    x1 = int((x_c - box_w / 2) * w)
                    y1 = int((y_c - box_h / 2) * h)
                    x2 = int((x_c + box_w / 2) * w)
                    y2 = int((y_c + box_h / 2) * h)

                    class_name = CLASS_NAMES[class_id]
                    classes_in_image.add(class_name)
                    color = COLORS[class_id % len(COLORS)]

                    cv2.rectangle(ground_truth_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(ground_truth_image, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
             cv2.putText(ground_truth_image, "No Ground Truth", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        std_size = (640, 640)
        prediction_image_resized = cv2.resize(prediction_image, std_size)
        ground_truth_image_resized = cv2.resize(ground_truth_image, std_size)

        top_margin = 50
        h_std, w_std, _ = prediction_image_resized.shape
        combined_image = np.zeros((h_std + top_margin, w_std * 2, 3), dtype=np.uint8)
        
        cv2.putText(combined_image, "Label", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(combined_image, "Prediction", (w_std + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        combined_image[top_margin:, :w_std] = ground_truth_image_resized
        combined_image[top_margin:, w_std:] = prediction_image_resized

        # ############################################################# #
        # ############## [수정 2: 파일 저장 로직 변경] ################## #
        # ############################################################# #
        if not classes_in_image:
            class_output_dir = os.path.join(OUTPUT_DIR, "no_label_images")
            os.makedirs(class_output_dir, exist_ok=True)
            
            # no_label 카운터 1 증가
            no_label_counter += 1
            # 새 파일 이름 생성 (예: no_label_1.jpg)
            new_filename = f"no_label_{no_label_counter}.jpg"
            output_path = os.path.join(class_output_dir, new_filename)
            cv2.imwrite(output_path, combined_image)
        else:
            for class_name in classes_in_image:
                class_output_dir = os.path.join(OUTPUT_DIR, class_name)
                os.makedirs(class_output_dir, exist_ok=True)
                
                # 해당 클래스의 카운터 1 증가
                class_counters[class_name] += 1
                # 새 파일 이름 생성 (예: cheops_1.jpg, cheops_2.jpg)
                new_filename = f"{class_name}_{class_counters[class_name]}.jpg"
                output_path = os.path.join(class_output_dir, new_filename)
                cv2.imwrite(output_path, combined_image)
        # ############################################################# #
                
    print("\n모든 비교 이미지 생성이 완료되었습니다.")

if __name__ == '__main__':
    compare_predictions()