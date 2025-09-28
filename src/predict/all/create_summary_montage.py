import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import random

"""
테스트셋의 정답과 예측 결과를 바탕으로,
클래스별 4x4 그리드 요약 이미지를 생성하고,
결과를 클래스별 하위 폴더에 나누어 저장하는 스크립트입니다.
(가독성 개선 최종 버전)
"""

# --- 1. 경로 및 설정 --- (이전과 동일)
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
#MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo8_hazard_all', 'weights', 'best.pt')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'runs', 'detect', 'yolo11_hazard_all', 'weights', 'best.pt')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'data', 'Space-debris-2')
YAML_PATH = os.path.join(DATASET_DIR, 'data.yaml')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test', 'images')
TEST_LABELS_DIR = os.path.join(DATASET_DIR, 'test', 'labels')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'summary_montage_results', 'results')
TILE_SIZE = (476, 476)
GRID_SHAPE = (4, 4)

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (0, 128, 128), (128, 0, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255)
]

# --- 2. 클래스 이름 불러오기 --- (이전과 동일)
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
# ############# [가독성이 개선된 draw_label 함수] ################ #
# ############################################################# #
def draw_label(image, label_text, box, color):
    x1, y1 = int(box[0]), int(box[1])
    
    # [수정] 폰트 크기, 굵기, 여백을 늘려 가독성 대폭 향상
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9  # 폰트 크기 증가
    font_thickness = 2  # 폰트 굵기
    padding = 5         # 텍스트 주변 여백

    (text_w, text_h), baseline = cv2.getTextSize(label_text, font_face, font_scale, font_thickness)
    
    # 배경 상자 위치 계산 (기본값: 박스 위)
    rect_start_point = (x1, y1 - text_h - (padding * 2) - baseline)
    rect_end_point = (x1 + text_w + (padding * 2), y1)
    
    # 텍스트 위치 계산
    text_start_point = (x1 + padding, y1 - padding - baseline)
    
    # 만약 텍스트가 이미지 상단 밖으로 나간다면, 박스 안쪽 상단으로 위치 조정
    if rect_start_point[1] < 0:
        rect_start_point = (x1, y1)
        rect_end_point = (x1 + text_w + (padding * 2), y1 + text_h + (padding * 2) + baseline)
        text_start_point = (x1 + padding, y1 + text_h + padding)

    # 텍스트 배경과 텍스트 그리기
    cv2.rectangle(image, rect_start_point, rect_end_point, color, -1)
    cv2.putText(image, label_text, text_start_point, font_face, font_scale, (255, 255, 255), font_thickness)
    
    return image

def create_montage(image_list, grid_shape, gap=5):
    grid_h, grid_w = grid_shape
    tile_h, tile_w = TILE_SIZE
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

# --- 4. 메인 로직 --- (이전과 동일)
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        return

    model = YOLO(MODEL_PATH)
    print("모델 로드 완료.")

    print("테스트셋 전체에 대한 예측을 시작합니다 (시간이 걸릴 수 있습니다)...")
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
            h, w, _ = original_image.shape

            gt_tile = original_image.copy()
            for gt in data['gts']:
                gid, xc, yc, bw, bh = gt
                gid = int(gid)
                color = COLORS[gid % len(COLORS)]
                x1, y1 = int((xc - bw/2) * w), int((yc - bh/2) * h)
                x2, y2 = int((xc + bw/2) * w), int((yc + bh/2) * h)
                cv2.rectangle(gt_tile, (x1, y1), (x2, y2), color, 2)
                gt_tile = draw_label(gt_tile, CLASS_NAMES[gid], (x1, y1), color)

            gt_tiles.append(cv2.resize(gt_tile, TILE_SIZE))

            pred_tile = original_image.copy()
            gt_count_this_class = sum(1 for gt in data['gts'] if int(gt[0]) == class_id)
            pred_count_this_class = sum(1 for pred in data['preds'] if int(pred[5]) == class_id)
            
            border_color = None
            if pred_count_this_class < gt_count_this_class:
                border_color = (0, 255, 255)
            elif pred_count_this_class > gt_count_this_class:
                border_color = (0, 0, 255)

            for pred in data['preds']:
                x1, y1, x2, y2, conf, pid = pred
                pid = int(pid)
                color = COLORS[pid % len(COLORS)]
                cv2.rectangle(pred_tile, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label_text = f"{CLASS_NAMES[pid]} {conf:.2f}"
                pred_tile = draw_label(pred_tile, label_text, (x1, y1), color)
            
            pred_tile = cv2.resize(pred_tile, TILE_SIZE)
            if border_color:
                cv2.rectangle(pred_tile, (0, 0), (TILE_SIZE[0]-1, TILE_SIZE[1]-1), border_color, 5)
            pred_tiles.append(pred_tile)
        
        gt_montage = create_montage(gt_tiles, GRID_SHAPE)
        pred_montage = create_montage(pred_tiles, GRID_SHAPE)
        
        class_specific_output_dir = os.path.join(OUTPUT_DIR, f"{class_id}_{class_name}")
        os.makedirs(class_specific_output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(class_specific_output_dir, "Label.jpg"), gt_montage)
        cv2.imwrite(os.path.join(class_specific_output_dir, "Prediction.jpg"), pred_montage)

    print(f"\n모든 작업이 완료되었습니다. '{OUTPUT_DIR}' 폴더를 확인해주세요.")

if __name__ == '__main__':
    main()