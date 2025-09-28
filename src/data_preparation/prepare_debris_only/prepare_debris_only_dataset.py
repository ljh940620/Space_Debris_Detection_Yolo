import os
import shutil

# --- 설정 (이 두 경로만 수정해주세요!) ---
# 1. 원본 11클래스 데이터셋이 있는 폴더 경로
# '새로운_데이터셋_폴더명'을 실제 폴더 이름으로 수정!
original_dataset_path = r'C:\Users\이진호\Desktop\PythonWorkspace\space_hazard_Project_Yolo\data\dataset'

# 2. 'debris' 클래스만 담을 새로운 데이터셋 폴더 경로 (이 이름 그대로 사용 권장)
new_dataset_path = r'C:\Users\이진호\Desktop\PythonWorkspace\space_hazard_Project_Yolo\data\debris_only_dataset'
# -----------------------------------------

# 'debris' 클래스의 원본 클래스 번호 (names 리스트에서 'debris'는 1번째, 즉 index=1)
DEBRIS_CLASS_ID = 1

def process_labels(split):
    """
    'train', 'valid', 'test' 폴더에 대해 라벨 파일을 처리하는 함수
    """
    print(f"'{split}' 폴더 처리 시작...")

    # 원본 이미지/라벨 폴더 경로
    original_image_dir = os.path.join(original_dataset_path, split, 'images')
    original_label_dir = os.path.join(original_dataset_path, split, 'labels')

    # 새로운 이미지/라벨 폴더 경로
    new_image_dir = os.path.join(new_dataset_path, split, 'images')
    new_label_dir = os.path.join(new_dataset_path, split, 'labels')

    # 새 폴더 생성
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    # 1. 모든 이미지 복사
    for filename in os.listdir(original_image_dir):
        shutil.copy(os.path.join(original_image_dir, filename), new_image_dir)

    # 2. 라벨 파일 처리
    for filename in os.listdir(original_label_dir):
        if not filename.endswith('.txt'):
            continue

        new_label_content = []
        with open(os.path.join(original_label_dir, filename), 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])

                # 'debris' 클래스(ID=1)인 경우에만, 클래스 ID를 0으로 바꿔서 저장
                if class_id == DEBRIS_CLASS_ID:
                    new_line = f"0 {' '.join(parts[1:])}\n"
                    new_label_content.append(new_line)

        # debris 라벨이 하나라도 있었다면, 새 라벨 파일 생성
        if new_label_content:
            with open(os.path.join(new_label_dir, filename), 'w') as f:
                f.writelines(new_label_content)

    print(f"'{split}' 폴더 처리 완료!")

if __name__ == '__main__':
    if not os.path.exists(original_dataset_path):
        print(f"오류: 원본 데이터셋 경로를 찾을 수 없습니다! -> {original_dataset_path}")
    else:
        # train, valid, test 폴더에 대해 각각 처리 실행
        for split_name in ['train', 'valid', 'test']:
            process_labels(split_name)
        print("\n모든 작업이 완료되었습니다! 'debris_only_dataset' 폴더를 확인하세요.")