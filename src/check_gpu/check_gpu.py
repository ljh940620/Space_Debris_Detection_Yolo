# check_gpu.py
import torch

print(f"PyTorch 버전: {torch.__version__}")
print("-" * 20)

is_available = torch.cuda.is_available()
print(f"GPU 사용 가능 여부: {is_available}")

if is_available:
    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    
    print(f"사용 가능한 GPU 개수: {device_count}")
    print(f"현재 GPU 이름: {device_name}")
    print("\n>>> GPU가 성공적으로 인식되었습니다! 이제 훈련을 시작하셔도 좋습니다.")
else:
    print("\n>>> GPU를 인식할 수 없습니다. PyTorch가 CPU 버전으로 설치되어 있습니다.")
    print("이전 답변을 참고하여 GPU 버전을 재설치해주세요.")