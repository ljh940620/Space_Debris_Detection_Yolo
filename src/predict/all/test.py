import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import random

"""
실행 시 테스트셋에서 임의의 이미지를 자동으로 선택하여
[정답 | 예측 | 정보창] 3단 레이아웃으로 결과를 시각화하고 저장합니다.
"""
# --- 1. 경로 및 설정 --- (이전과 동일)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo8_hazard_all', 'weights', 'best.pt')
#MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo11_hazard_all', 'weights', 'best.pt')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATASET_DIR, 'test', 'labels')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'summary_montage_results', 'results')
# 타일 및 그리드 스타일 설정
IMG_SIZE = (400, 400)
LEGEND_WIDTH = 300
TILE_SIZE = (IMG_SIZE[0] + LEGEND_WIDTH, IMG_SIZE[1])
GRID_SHAPE = (4, 4)
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
        print(f"총 {len(CLASS_NAMES)}개의 클래스를 찾았습니다: {CLASS_NAMES}")
except Exception as e:
    print(f"오류: {YAML_PATH} 파일을 읽을 수 없습니다.")
    exit()

# --- 3. 헬퍼 함수 ---

# ############################################################# #
# ############# [최종 수정된 create_tile_with_legend 함수] ####### #
# ############################################################# #
def create_tile_with_legend(image, boxes, is_prediction=False):
    """[이미지 + 정보창] 형태의 타일 하나를 생성합니다. (2줄 레이아웃 적용)"""
    tile = np.full((TILE_SIZE[1], TILE_SIZE[0], 3), 30, dtype=np.uint8)
    img_resized = cv2.resize(image, IMG_SIZE)
    h, w, _ = image.shape
    
    # 이미지에 바운딩 박스 그리기
    for i, box_data in enumerate(boxes):
        if is_prediction:
            x1, y1, x2, y2, _, class_id = box_data
        else: # Ground Truth
            class_id = box_data[0]
            xc, yc, bw, bh = box_data[1:]
            x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
            x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
        
        class_id = int(class_id)
        color = COLORS[class_id % len(COLORS)]
        # 이미지 크기 비율에 맞춰 바운딩 박스 좌표 변환
        rx1, ry1 = int(x1 * IMG_SIZE[0] / w), int(y1 * IMG_SIZE[1] / h)
        rx2, ry2 = int(x2 * IMG_SIZE[0] / w), int(y2 * IMG_SIZE[1] / h)
        cv2.rectangle(img_resized, (rx1, ry1), (rx2, ry2), color, 2)

    tile[:, :IMG_SIZE[0]] = img_resized

    # 정보창에 텍스트 쓰기
    y_pos = 40
    for i, box_data in enumerate(boxes):
        # 텍스트 스타일
        font_scale_class = 0.7
        font_scale_conf = 0.6
        font_thickness = 2
        
        if is_prediction:
            _, _, _, _, conf, class_id = box_data
        else: # Ground Truth
            class_id = box_data[0]
            
        class_id = int(class_id)
        class_name = CLASS_NAMES[class_id]
        color = COLORS[class_id % len(COLORS)]

        # 첫 번째 줄: 번호, 색상 박스, 클래스 이름
        text_line1 = f"{i+1}. {class_name}"
        (w1, h1), _ = cv2.getTextSize(text_line1, FONT, font_scale_class, font_thickness)
        cv2.rectangle(tile, (IMG_SIZE[0] + 15, y_pos - h1), (IMG_SIZE[0] + 15 + 20, y_pos+5), color, -1)
        cv2.putText(tile, text_line1, (IMG_SIZE[0] + 45, y_pos), FONT, font_scale_class, (255, 255, 255), font_thickness)
        y_pos += h1 + 5

        # 두 번째 줄: 신뢰도 점수 (예측일 경우에만)
        if is_prediction:
            text_line2 = f"   (Conf: {conf:.2f})"
            (w2, h2), _ = cv2.getTextSize(text_line2, FONT, font_scale_conf, font_thickness)
            cv2.putText(tile, text_line2, (IMG_SIZE[0] + 45, y_pos), FONT, font_scale_conf, (200, 200, 200), font_thickness)
            y_pos += h2 + 20 # 다음 객체와의 간격
        else:
            y_pos += 20 # 다음 객체와의 간격

        if y_pos > TILE_SIZE[1] - 20: break
            
    return tile

def create_montage(image_list, grid_shape, gap=5):
    """타일 리스트를 받아서 최종 그리드 이미지를 생성합니다."""
    grid_h, grid_w = grid_shape
    tile_w, tile_h = TILE_SIZE
    montage_h = grid_h * tile_h + (grid_h - 1) * gap
    montage_w = grid_w * tile_w + (grid_w - 1) * gap
    montage = np.full((montage_h, montage_w, 3), 50, dtype=np.uint8)
    for i, img in enumerate(image_list):
        if i >= grid_h * grid_w:
            break
        row, col = divmod(i, grid_w)
        y_offset = row * (tile_h + gap)
        x_offset = col * (tile_w + gap)
        montage[y_offset:y_offset+tile_h, x_offset:x_offset+tile_w] = img
    return montage

# --- 4. 메인 로직 ---
def main():
    # ... (이하 메인 로직은 이전 코드와 동일) ...
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다.")
        return
    model = YOLO(MODEL_PATH)
    print("모델 로드 완료.")
    print("테스트셋 전체에 대한 예측을 시작합니다...")
    all_results = model.predict(source=TEST_IMAGES_DIR, batch=16, stream=True, verbose=False)

    image_data = {}
    for result in tqdm(all_results, total=len(os.listdir(TEST_IMAGES_DIR)), desc="예측 결과 수집 중"):
        image_path = result.path
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(TEST_LABELS_DIR, base_name + '.txt')
        image_data[image_path] = {'preds': result.boxes.data.cpu().numpy(), 'gts': []}
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    image_data[image_path]['gts'].append(list(map(float, line.strip().split())))

    class_images = {name: [] for name in CLASS_NAMES}
    for image_path, data in image_data.items():
        gt_classes = {int(gt[0]) for gt in data['gts']}
        for class_id in gt_classes:
            class_images[CLASS_NAMES[class_id]].append(image_path)
    
    print("\n클래스별 요약 이미지 생성을 시작합니다...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for class_id, class_name in enumerate(tqdm(CLASS_NAMES, desc="클래스별 처리 중")):
        image_paths_for_class = random.sample(class_images[class_name], min(len(class_images[class_name]), 16))
        if not image_paths_for_class:
            continue

        gt_tiles, pred_tiles = [], []
        for image_path in image_paths_for_class:
            data = image_data[image_path]
            original_image = cv2.imread(image_path)
            
            gt_tile = create_tile_with_legend(original_image, data['gts'], is_prediction=False)
            pred_tile = create_tile_with_legend(original_image, data['preds'], is_prediction=True)
            
            gt_count = sum(1 for gt in data['gts'] if int(gt[0]) == class_id)
            pred_count = sum(1 for pred in data['preds'] if int(pred[5]) == class_id)
            
            border_color = None
            if pred_count < gt_count: border_color = (0, 255, 255)
            elif pred_count > gt_count: border_color = (0, 0, 255)
            
            if border_color:
                cv2.rectangle(pred_tile, (0, 0), (TILE_SIZE[0]-1, TILE_SIZE[1]-1), border_color, 10)
            
            gt_tiles.append(gt_tile)
            pred_tiles.append(pred_tile)
            
        gt_montage = create_montage(gt_tiles, GRID_SHAPE)
        pred_montage = create_montage(pred_tiles, GRID_SHAPE)
        
        class_specific_output_dir = os.path.join(OUTPUT_DIR, f"{class_id}_{class_name}")
        os.makedirs(class_specific_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(class_specific_output_dir, "Label_yolo8.jpg"), gt_montage)
        cv2.imwrite(os.path.join(class_specific_output_dir, "Prediction_yolo8.jpg"), pred_montage)

    print(f"\n모든 작업이 완료되었습니다. '{OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == '__main__':
    main()