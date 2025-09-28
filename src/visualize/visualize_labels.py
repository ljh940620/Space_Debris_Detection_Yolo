""" 이 스크립트의 목적은 YOLO 형식으로 만들어진 데이터셋의 정답 라벨을 시각화하여 데이터의 품질을 검증하기 위한 도구입니다. 
텍스트 파일(.txt)에 숫자로만 적혀 있는 라벨 좌표를 실제 이미지 위에 바운딩 박스로 그려서 보여줌으로써,
데이터 오류를 사전에 찾아낼 수 있습니다. """ 

# import os: 운영체제와 상호작용하는 기능을 제공합니다. 주로 파일 경로를 만들거나(os.path.join), 파일 이름을 추출하는(os.path.basename)데 사용됩니다.
# import glob: 특정 패턴과 일치하는 파일들의 목록을 쉽게 찾아주는 도구입니다. 여기서는 'images' 폴더 안에 있는 모든 .jpg나 .png 파일을 찾는 데 사용됩니다.
# import cv2: OpenCV 라이브러리로, 이미지 처리의 핵심 도구입니다. 이미지를 읽고(imread), 이미지 위에 도형을 그리고(rectangle), 텍스트를 쓰고(putText), 최종적으로 이미지를 파일로 저장(imwrite)하는 모든 시각화 작업을 담당합니다.
# from tqdm import tqdm: 긴 반복문 작업이 얼마나 진행되었는지 예쁜 진행률 막대(Progress Bar)로 보여주는 라이브러리입니다. 수백, 수천 장의 이미지를 처리할 때 작업이 멈춘 것이 아니라 잘 돌아가고 있다는 것을 알려주어 편리합니다.
import os
import glob
import cv2
from tqdm import tqdm

# --- 설정 (이 부분만 사용 목적에 맞게 수정해주세요) ---

# 1. 원본 이미지가 있는 폴더 경로
image_dir = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\debris_only_dataset\test\images'

# 2. yolo 라벨(.txt) 파일이 있는 폴더 경로
label_dir = r'C:\Users\leeji\Desktop\PythonWorkspace\Space_Debris_Project_Yolo\data\debris_only_dataset\test\labels'

# 3. 시각화된 결과 이미지를 저장할 폴더 이름
output_dir = 'visualization_results_labels_only_debris'

# 4. 클래스 이름 목록 (data.yaml 파일의 names와 동일하게)
class_names = ['debris']

# 5. 바운딩 박스 색상 (B, G, R 순서 - 여기서는 밝은 녹색)
box_color = (0, 255, 0)
text_color = (0, 0, 0) # 검은색 텍스트

# --------------------------------------------------------------------------------------------------------------------

# 핵심 기능 함수: visualize_labels
# 이 함수는 이미지 한 장을 받아 처리하는 실제 작업의 핵심입니다.
""" 하나의 이미지와 라벨 파일을 읽어 바운딩 박스를 그린 후 저장하는 함수 """
def visualize_labels(image_path, label_path, output_path):
    # 1. 이미지 읽기
    """ cv2.imread로 이미지를 불러와 h(높이), w(너비) 값을 얻습니다. 이 값은 YOLO의 정규화된(normalized) 좌표를 실제 픽셀 위치로 되돌리는 데 필수적입니다. """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: 이미지를 불러올 수 없습니다: {image_path}")
        return
    """ 숫자 배열의 모양(차원)을 알려줍니다. 예를 들어 (480, 640, 3)과 같은 튜플(tuple) 형태로 반환되는데, 이는 각각 (높이, 너비, 채널수)를 의미합니다.
    이 튜플의 값을 h에는 높이, w에는 너비를 각각 저장합니다. 색상 채널 정보(3)는 당장 필요 없으므로 _ 라는 변수명으로 무시하고 받지 않겠다는 관례적인 표현입니다.
    이 h와 w 값은 YOLO의 상대 좌표를 실제 픽셀 좌표로 변환하는 데 사용되므로 이 함수에서 가장 중요한 정보 중 하나입니다. """
    h, w, _ = image.shape 
    
    """ 라벨 파일이 없는 이미지는 '배경 이미지(Negative Sample)'이므로, 아무것도 그리지 않고 함수를 종료합니다. """
    if not os.path.exists(label_path):  # 3. 라벨 파일이 없는 경우
        # cv2.imwrite(output_path, image) # 라벨 파일이 없는 경우(배경 이미지), 원본 이미지만 저장
        return
    
    """ with open... 구문으로 라벨 파일을 안전하게 엽니다. for line in f:를 통해 파일 안의 각 줄(각 바운딩 박스 정보)을 순서대로 처리합니다. """
    # 라벨 파일 읽기
    with open(label_path, 'r') as f: # 4. 라벨 파일 열기
        for line in f:               #    한 줄씩 읽기 (객체 하나당 한 줄)
            # --- 5. 라벨 정보 파싱 및 좌표 변환 ---
            """가장 중요한 좌표 변환 부분입니다.
            1. YOLO 라벨 0 0.5 0.5 0.2 0.2 와 같은 한 줄을 공백 기준으로 나눕니다. (['0', '0.5', '0.5', '0.2', '0.2']) 
            2. 첫 번째는 클래스 ID, 나머지는 중심 x, 중심 y, 너비, 높이입니다.
            3. 이 값들은 이미지 전체 너비와 높이를 1로 봤을 때의 상대적인 비율이므로, * w 또는 * h를 곱하여 실제 픽셀 위치로 변환합니다.
            4. cv2.rectangle 함수는 중심점이 아닌 좌상단, 우하단 좌표를 사용하므로, 계산을 통해 이 두 점의 좌표(x1, y1, x2, y2)를 구합니다. """
            try:
                # YOLO 형식: class_id, x_center, y_center, width, height (모두 0~1 사이 값)
                parts = line.strip().split() 
                # line.strip(): 줄 끝의 눈에 보이지 않는 줄바꿈 문자(\n)나 공백을 제거합니다.
                # 예: "0 0.5 0.5 0.2 0.2\n" -> "0 0.5 0.5 0.2 0.2"
                # .split(): 공백을 기준으로 문자열을 잘라 리스트로 만듭니다. 예: "0 0.5 0.5 0.2 0.2" -> ['0', '0.5', '0.5', '0.2', '0.2']
                class_id = int(parts[0])
                # class_id = int(parts[0]): 리스트의 첫 번째 요소('0')를 정수(int)로 변환하여 class_id에 저장합니다.
                x_center_norm, y_center_norm, width_norm, height_norm = map(float, parts[1:])
                # map(float, parts[1:]): 리스트의 두 번째 요소부터 끝까지(parts[1:])에 대해, float 함수를 각각 적용하여 모든 문자열을 실수(소수점이 있는 숫자)로 변환합니다. 그 결과를 각 변수에 차례대로 저장합니다.

                # 정규화된 좌표를 실제 픽셀 좌표로 변환 (De-normalization)
                # YOLO의 상대 좌표(0~1 사이의 비율 값)에 이미지의 실제 너비(w)와 높이(h)를 곱하여, 실제 픽셀 단위의 좌표와 크기로 변환하는 과정입니다.
                x_center = x_center_norm * w
                y_center = y_center_norm * h
                box_width = width_norm * w
                box_height = height_norm * h

                # 중심좌표/너비 -> 박스의 좌상단(x1, y1), 우하단(x2, y2) 좌표로 변환
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)
                # OpenCV의 사각형 그리기 함수(cv2.rectangle)는 박스의 '왼쪽 위 꼭짓점'과 '오른쪽 아래 꼭짓점'의 좌표를 필요로 합니다.
                # YOLO 좌표(중심점, 너비, 높이)를 cv2가 사용하기 편한 형식으로 변환하는 계산입니다. int()는 픽셀 좌표가 정수여야 하므로 소수점을 버리기 위해 사용됩니다.
                """왜 이 변환이 필요한가?
                가장 큰 이유는 YOLO가 좌표를 저장하는 방식과 OpenCV(cv2)가 사각형을 그리는 방식이 서로 다르기 때문입니다.
                1. YOLO의 방식 (라벨 파일에 저장된 정보): 박스의 중심점 좌표 와 전체 너비/높이 로 객체를 표현합니다.
                (x_center, y_center, width, height)
                2. OpenCV의 방식(cv2.rectangle이 요구하는 정보): 박스의 왼쪽 위 꼭짓점 과 오른쪽 아래 꼭짓점 의 좌표로 객체를 표현합니다.
                (x1, y1) 과 (x2, y2)
                따라서 우리는 YOLO의 정보('중심점'과 '크기')를 가지고 OpenCV가 알아들을 수 있는 정보('양쪽 끝 꼭짓점')로 변환해 주어야 합니다.

                좌표 변환 과정 (그림으로 이해하기)
                아래 그림은 하나의 바운딩 박스와 관련된 모든 좌표 정보를 보여줍니다.

                (x1, y1) <--- 이 점을 찾아야 함
                    *---------------------------*
                    |                           |
                    |                           |
                    |         (x_center, y_center) ● 중심점 (YOLO가 알려줌)
                    |                           |
                    |                           |
                    *---------------------------*
                                              (x2, y2) <--- 이 점을 찾아야 함

                    <---------- width ---------->
                                (YOLO가 알려줌)
                이제 이 그림을 보면서 코드 한 줄 한 줄을 따라가 보겠습니다.

                1. x1 (박스의 왼쪽 X좌표) 계산하기
                x1 = int(x_center - box_width / 2)

                우리는 박스의 가장 왼쪽 끝(x1)의 위치를 알고 싶습니다.

                우리가 아는 정보는 박스의 정중앙(x_center)과 박스의 전체 너비(box_width)입니다.

                정중앙(x_center)에서 전체 너비의 절반(box_width / 2) 만큼 왼쪽으로 이동하면 정확히 가장 왼쪽 끝(x1)에 도달하게 됩니다.

                따라서 계산식은 x_center - (box_width / 2) 가 됩니다.

                2. y1 (박스의 위쪽 Y좌표) 계산하기
                y1 = int(y_center - box_height / 2)
                박스의 가장 위쪽 끝(y1)의 위치를 계산하는 것도 동일한 원리입니다.

                정중앙(y_center)에서 전체 높이의 절반(box_height / 2) 만큼 위쪽으로 이동하면 가장 위쪽 끝(y1)에 도달합니다.

                (컴퓨터 이미지에서는 좌표가 왼쪽 위에서 시작하므로, 위로 가는 것이 빼기(-) 방향입니다.)

                따라서 계산식은 y_center - (box_height / 2) 입니다.

                3. x2, y2 (박스의 오른쪽 아래 좌표) 계산하기
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)
                오른쪽 아래 꼭짓점도 마찬가지입니다.

                가장 오른쪽 끝(x2)은 중심(x_center)에서 너비의 절반만큼 오른쪽(+ box_width / 2) 으로 이동한 지점입니다.

                가장 아래쪽 끝(y2)은 중심(y_center)에서 높이의 절반만큼 아래쪽(+ box_height / 2) 으로 이동한 지점입니다.

                int(...)는 왜 필요한가?
                계산 결과는 123.5 와 같은 소수점이 있는 실수(float)일 수 있습니다.

                하지만 이미지의 픽셀 위치는 123번 픽셀, 124번 픽셀처럼 정수(integer)여야 합니다.

                int() 함수는 계산된 실수 값의 소수점 이하를 버리고 정수로 만들어주는 역할을 합니다.

                이 과정을 거치면 비로소 OpenCV의 cv2.rectangle 함수가 "아, (x1, y1)에서 시작해서 (x2, y2)에서 끝나는 사각형을 그리면 되는구나!" 라고 이해할 수 있게 되는 것입니다. """
                # 이미지에 사각형(바운딩 박스) 그리기
                cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2) # image 위에, (x1, y1)부터 (x2, y2)까지, box_color 색상으로, 두께 2짜리 사각형을 그립니다.

                # 클래스 이름 텍스트 추가
                label_text = class_names[class_id]
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # cv2.putText로 글씨를 쓰기 전에, 그 글씨가 차지할 공간의 너비(text_w)와 높이(text_h)를 미리 계산하는 과정입니다. 텍스트 배경을 정확한 크기로 그리기 위해 꼭 필요합니다.
                cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w, y1), box_color, -1)
                cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                # cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2: 각각 글꼴, 글자 크기, 색상, 굵기를 의미합니다.
                """try...except는 오류 처리 구문입니다. 만약 라벨 파일의 특정 줄이 비어있거나 형식이 잘못된 경우,
                int()나 float() 변환 시 오류가 발생하여 프로그램이 멈추게 됩니다. 이 except 구문은 그런 오류를 잡아내어 프로그램을 멈추는 대신,
                어떤 파일의 어떤 줄에 문제가 있는지 경고 메시지를 출력하고 다음 작업을 계속하도록 해주는 안전장치입니다. """
            except (ValueError, IndexError) as e:
                print(f"Warning: '{label_path}' 파일의 라인 형식이 잘못되었습니다: {line.strip()} | 오류: {e}")

    # 7. 결과 이미지 저장
    cv2.imwrite(output_path, image)        

if __name__ == '__main__':
    # 결과 저장 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과 이미지는 '{output_dir}' 폴더에 저장됩니다.")

    # 이미지 파일 목록 가져오기
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    """ image_extensions: 찾아낼 파일 확장자 패턴들을 미리 정의합니다. *는 "모든 문자"를 의미하는 와일드카드입니다. """
    """ image_paths = []: 찾아낸 이미지 파일들의 전체 경로를 저장할 빈 리스트를 만듭니다.
    for ext in ...: *.jpg, *.jpeg, *.png 패턴을 하나씩 돌아가며 작업을 반복합니다.
    glob.glob(os.path.join(image_dir, ext)): image_dir 경로와 패턴(예: *.jpg)을 합쳐서 (예: C:\...\images\*.jpg), 이 패턴과 일치하는 모든 파일의 경로를 찾아 리스트로 반환합니다.
    image_paths.extend(...): glob이 찾아낸 파일 경로 리스트를 image_paths 리스트의 맨 뒤에 추가합니다. 이 과정을 모든 확장자에 대해 반복하여, 최종적으로 image_paths는 모든 이미지 파일의 전체 경로를 담게 됩니다. """
    if not image_paths:
        print(f"오류: '{image_dir}' 폴더에서 이미지를 찾을 수 없습니다.")
    else:
        # 각 이미지에 대해 시각화 작업 수행
        for img_path in tqdm(image_paths, desc="라벨 시각화 진행 중"):
            # 이미지 파일 이름에 맞는 라벨 파일 경로 생성 (예: image1.jpg -> image1.txt)
            base_filename = os.path.basename(img_path)
            label_filename = os.path.splitext(base_filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)
            """ 설명: 현재 처리 중인 이미지(img_path)에 짝이 되는 라벨 파일의 전체 경로를 만들어내는 과정입니다.
            base_filename = os.path.basename(img_path): 전체 경로에서 파일 이름(예: photo_01.jpg)만 추출합니다.
            label_filename = os.path.splitext(...)[0] + '.txt': 파일 이름에서 확장자(.jpg)를 제거하고(.splitext), 그 자리에 .txt를 붙여 라벨 파일 이름(예: photo_01.txt)을 완성합니다.
            label_path = os.path.join(...): 라벨 폴더 경로(label_dir)와 방금 만든 라벨 파일 이름을 합쳐 라벨 파일의 전체 경로를 만듭니다. """
            # 결과 파일이 저장될 경로
            """ 설명: 결과 저장 폴더 경로(output_dir)와 원본 파일 이름(base_filename)을 합쳐, 시각화된 결과 이미지가 저장될 최종 경로를 만듭니다. """
            output_path = os.path.join(output_dir, base_filename)
            """ 설명: 이 루프의 핵심 실행문입니다.
            위에서 준비한 세 가지 정보(①원본 이미지 경로, ②라벨 파일 경로, ③결과 저장 경로)를 visualize_labels 함수에 전달하여 호출합니다.
            이 함수가 호출되면, 이미지 한 장에 대한 모든 그리기 및 저장 작업이 수행됩니다. """
            visualize_labels(img_path, label_path, output_path)
        """ 설명: for 반복문이 모든 이미지를 처리하고 끝난 뒤, 사용자에게 모든 작업이 성공적으로 완료되었음을 알리는 최종 메시지를 출력합니다. """
        print("\n모든 라벨 시각화 작업이 완료되었습니다.")


