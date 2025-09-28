import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import random
from tqdm import tqdm

"""
목적: yolov11n과 yolov8n 두 모델의 성능을 시각적으로 비교.

작동 방식:
1. test 폴더의 모든 이미지를 대상으로 합니다.
2. 먼저 yolov8n 모델로 모든 이미지에 대한 [정답 | 예측 | 정보창] 3단 비교 이미지를 생성합니다.
3. 생성된 이미지들을 정답 라벨에 포함된 클래스를 기준으로 클래스별 폴더에 나누어 저장합니다.
4. yolov8n 작업이 끝나면, 자동으로 yolov11n 모델에 대해 2~3번 과정을 반복합니다.

최종 결과물: label_predict_comparison_results 폴더 안에 yolov8n_results와 yolov11n_results 두 개의 폴더가 생성되고, 그 안에 각각 클래스별로 분류된 비교 이미지들이 저장됩니다.
"""

# --- 1. 경로 및 설정 ---
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))

# 비교할 두 모델의 경로를 각각 지정
MODELS_TO_PROCESS = {
    "yolov8n_results": os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo8_hazard_all', 'weights', 'best.pt'),
    "yolov11n_results": os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo11_hazard_all', 'weights', 'best.pt')
}

DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATASET_DIR, 'test', 'labels')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'label_predict_comparison_results')

# 시각화 스타일 설정
IMG_SIZE = (600, 600)
LEGEND_WIDTH = 450
FONT = cv2.FONT_HERSHEY_SIMPLEX
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
except Exception as e:
    print(f"오류: {YAML_PATH} 파일을 읽을 수 없습니다.")
    exit()

# --- 3. 헬퍼 함수 --- (이전과 동일)
def draw_boxes_with_numbers(image, boxes, is_prediction=False):
    for i, box_data in enumerate(boxes):
        if is_prediction:
            x1, y1, x2, y2, _, class_id = box_data
        else: # Label
            class_id, xc, yc, bw, bh = box_data
            h, w, _ = image.shape
            x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
            x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        class_id = int(class_id)
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        label = str(i + 1)
        (w_text, h_text), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        cv2.rectangle(image, (int(x1), int(y1) - h_text - 10), (int(x1) + w_text + 5, int(y1)), color, -1)
        cv2.putText(image, label, (int(x1) + 2, int(y1) - 5), FONT, 0.8, (255, 255, 255), 2)
    return image

def create_legend_panel(title, boxes, is_prediction=False):
    title_h = 60
    line_h = 35
    total_h = title_h + (len(boxes) * line_h) + 20
    panel = np.full((total_h, LEGEND_WIDTH, 3), 40, dtype=np.uint8)
    cv2.putText(panel, title, (20, 40), FONT, 1, (255, 255, 255), 2)
    y_pos = 80
    for i, box_data in enumerate(boxes):
        if is_prediction:
            _, _, _, _, conf, class_id = box_data
            text = f"{i+1}. {CLASS_NAMES[int(class_id)]} ({conf:.2f})"
        else:
            class_id = box_data[0]
            text = f"{i+1}. {CLASS_NAMES[int(class_id)]}"
        class_id = int(class_id)
        color = COLORS[class_id % len(COLORS)]
        cv2.rectangle(panel, (20, y_pos - 20), (40, y_pos), color, -1)
        cv2.putText(panel, text, (50, y_pos), FONT, 0.7, (255, 255, 255), 2)
        y_pos += line_h
    return panel

# --- 4. 메인 로직 ---
def main():
    # 모든 테스트 이미지 경로 리스트업
    all_image_files = [os.path.join(TEST_IMAGES_DIR, f) for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 두 모델 파일 모두 확인
    # 모델 목록을 순회하며 처리
    for model_name, model_path in MODELS_TO_PROCESS.items():
        if not os.path.exists(model_path):
            print(f"[오류] {model_name} 모델 파일을 찾을 수 없습니다: {model_path}")
            continue

        model = YOLO(model_path)
        print(f"\n--- '{model_name}' 모델 처리 시작 ---")
        
         # 3. 전체 이미지에 대해 예측 실행
        results_generator = model.predict(source=all_image_files, batch=16, stream=True, verbose=False)

        for results in tqdm(results_generator, total=len(all_image_files), desc=f"[{model_name}] 예측 중"):
            image_path = results.path
            original_image = cv2.imread(image_path)
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # Label 데이터 로드
            label_path = os.path.join(TEST_LABELS_DIR, base_name + '.txt')
            gt_boxes = []
            classes_in_image = set()
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        gt_boxes.append(parts)
                        classes_in_image.add(CLASS_NAMES[int(parts[0])])

            # 예측 결과
            pred_boxes = results.boxes.data.cpu().numpy()

            # [정답 | 예측 | 정보창] 3단 이미지 생성
            gt_image_panel = draw_boxes_with_numbers(original_image.copy(), gt_boxes, is_prediction=False)
            pred_image_panel = draw_boxes_with_numbers(original_image.copy(), pred_boxes, is_prediction=True)
            
            gt_legend_panel = create_legend_panel("label", gt_boxes, is_prediction=False)
            pred_legend_panel = create_legend_panel(f"Prediction ({model_name.split('_')[0]})", pred_boxes, is_prediction=True)

            full_legend_panel = np.vstack([gt_legend_panel, pred_legend_panel])
            
            legend_for_comparison = cv2.resize(full_legend_panel, (LEGEND_WIDTH, IMG_SIZE[1]))
            gt_image_panel_resized = cv2.resize(gt_image_panel, IMG_SIZE)
            pred_image_panel_resized = cv2.resize(pred_image_panel, IMG_SIZE)
            
            comparison_image = np.hstack([gt_image_panel_resized, pred_image_panel_resized, legend_for_comparison])

            # 클래스별 폴더에 저장
            if not classes_in_image:
                class_output_dir = os.path.join(OUTPUT_DIR, model_name, "no_label_images")
                os.makedirs(class_output_dir, exist_ok=True)
                save_path = os.path.join(class_output_dir, f"comparison_{base_name}.jpg")
                cv2.imwrite(save_path, comparison_image)
            else:
                for class_name in classes_in_image:
                    class_output_dir = os.path.join(OUTPUT_DIR, model_name, class_name)
                    os.makedirs(class_output_dir, exist_ok=True)
                    save_path = os.path.join(class_output_dir, f"comparison_{base_name}.jpg")
                    cv2.imwrite(save_path, comparison_image)

if __name__ == '__main__':
    main()
    print(f"\n모든 작업이 완료되었습니다. '{OUTPUT_DIR}' 폴더를 확인해주세요.")