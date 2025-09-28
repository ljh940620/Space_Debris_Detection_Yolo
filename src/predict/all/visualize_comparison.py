import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import argparse

"""
단일 이미지에 대해 [정답 | 예측 | 정보창] 3단 레이아웃으로
결과를 시각화하고 저장하는 스크립트입니다.
"""

# --- 1. 경로 및 설정 ---
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo8_hazard_all', 'weights', 'best.pt')
#MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo11_hazard_all', 'weights', 'best.pt')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TEST_LABELS_DIR = os.path.join(DATASET_DIR, 'test', 'labels')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'summary_montage_results', 'results')

# 시각화 스타일 설정
IMG_SIZE = (800, 800)
LEGEND_WIDTH = 400
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

# --- 3. 헬퍼 함수 ---
def draw_boxes_with_labels(image, boxes, is_prediction=False):
    """이미지에 바운딩 박스와 번호표를 그립니다."""
    for i, box_data in enumerate(boxes):
        if is_prediction:
            x1, y1, x2, y2, _, class_id = box_data
        else: # Ground Truth
            class_id, xc, yc, bw, bh = box_data
            h, w, _ = image.shape
            x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
            x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        
        class_id = int(class_id)
        color = COLORS[class_id % len(COLORS)]
        
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        
        # 번호표 그리기
        label = str(i + 1)
        (w_text, h_text), _ = cv2.getTextSize(label, FONT, 0.8, 2)
        cv2.rectangle(image, (int(x1), int(y1) - h_text - 10), (int(x1) + w_text, int(y1)-5), color, -1)
        cv2.putText(image, label, (int(x1), int(y1) - 5), FONT, 0.8, (255, 255, 255), 2)
    return image

def create_legend_panel(title, boxes, is_prediction=False):
    """정보창(Legend) 이미지를 생성합니다."""
    panel = np.full((IMG_SIZE[1], LEGEND_WIDTH, 3), 40, dtype=np.uint8)
    
    # 타이틀
    cv2.putText(panel, title, (20, 50), FONT, 1.2, (255, 255, 255), 2)
    
    y_pos = 100
    for i, box_data in enumerate(boxes):
        if is_prediction:
            _, _, _, _, conf, class_id = box_data
            text = f"{i+1}. {CLASS_NAMES[int(class_id)]} ({conf:.2f})"
        else: # Ground Truth
            class_id = box_data[0]
            text = f"{i+1}. {CLASS_NAMES[int(class_id)]}"
            
        class_id = int(class_id)
        color = COLORS[class_id % len(COLORS)]
        
        cv2.rectangle(panel, (20, y_pos - 20), (40, y_pos), color, -1)
        cv2.putText(panel, text, (50, y_pos), FONT, 0.8, (255, 255, 255), 2)
        y_pos += 40
        
    return panel

# --- 4. 메인 로직 ---
def main(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return
    if not os.path.exists(image_path):
        print(f"[오류] 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
        
    model = YOLO(MODEL_PATH)
    original_image = cv2.imread(image_path)
    
    # Ground Truth 데이터 로드
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(TEST_LABELS_DIR, base_name + '.txt')
    gt_boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                gt_boxes.append(list(map(float, line.strip().split())))

    # 예측 실행
    results = model.predict(source=image_path, verbose=False)
    pred_boxes = results[0].boxes.data.cpu().numpy()

    # 각 패널 생성
    gt_image_panel = draw_boxes_with_labels(original_image.copy(), gt_boxes, is_prediction=False)
    pred_image_panel = draw_boxes_with_labels(original_image.copy(), pred_boxes, is_prediction=True)
    
    gt_legend_panel = create_legend_panel("Label", gt_boxes, is_prediction=False)
    pred_legend_panel = create_legend_panel("Prediction", pred_boxes, is_prediction=True)

    # 정보창 합치기
    legend_panel = np.vstack([gt_legend_panel, pred_legend_panel])
    legend_panel = cv2.resize(legend_panel, (LEGEND_WIDTH, IMG_SIZE[1]))

    # 모든 패널 최종 합치기
    gt_image_panel = cv2.resize(gt_image_panel, IMG_SIZE)
    pred_image_panel = cv2.resize(pred_image_panel, IMG_SIZE)
    final_image = np.hstack([gt_image_panel, pred_image_panel, legend_panel])

    # 결과 저장 및 표시
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"comparison_{base_name}.jpg")
    cv2.imwrite(save_path, final_image)
    print(f"결과 이미지가 다음 경로에 저장되었습니다: {save_path}")

    cv2.imshow("Ground Truth vs Prediction", final_image)
    print("결과 창이 열렸습니다. 아무 키나 누르면 종료됩니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLOv8 모델의 예측과 정답을 나란히 시각화합니다.")
    parser.add_argument('--image', type=str, required=True, help="분석할 이미지 파일의 경로")
    args = parser.parse_args()
    
    main(image_path=args.image)