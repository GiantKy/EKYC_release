"""
Config module for E-KYC Server Module.
Chứa các thiết lập đường dẫn weights, ngưỡng nhận diện và tham số thuật toán.
"""

import os
from pathlib import Path

# Đường dẫn thư mục gốc
SERVER_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SERVER_MODULE_DIR)

# =====================================================================
# THƯ MỤC MODELS: Chỉ sử dụng server_module/models/ (tự chứa)
# =====================================================================
MODELS_DIR = os.path.join(SERVER_MODULE_DIR, "models")

# Đường dẫn các mô hình AI — tất cả nằm trong server_module/models/
FACE_DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection", "yolo_face_detection_official.pt") if os.path.exists(os.path.join(MODELS_DIR, "face_detection", "yolo_face_detection_official.pt")) else os.path.join(MODELS_DIR, "Face_Detection.pt")
ANTI_SPOOF_YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "Anti_Spoof_YOLO.pt")
FACE_LANDMARKER_MODEL_PATH = os.path.join(MODELS_DIR, "landmarks", "mediapipe_face_landmarker_official.task") if os.path.exists(os.path.join(MODELS_DIR, "landmarks", "mediapipe_face_landmarker_official.task")) else os.path.join(MODELS_DIR, "face_landmarker.task")

# =====================================================================
# ENSEMBLE ANTI-SPOOF: YOLO_4 + RF-DETR Small
# =====================================================================
# Model 1: YOLO_4 (Anti_Spoof_YOLO_4.pt) — Object Detection local
ANTI_SPOOF_YOLO4_MODEL_PATH = os.path.join(MODELS_DIR, "anti_spoof", "yolo", "yolo_anti_spoof_v4_official.pt") if os.path.exists(os.path.join(MODELS_DIR, "anti_spoof", "yolo", "yolo_anti_spoof_v4_official.pt")) else os.path.join(MODELS_DIR, "Anti_Spoof_YOLO_4.pt")

# Model 2: RF-DETR Small — Local ONNX Weights (100% Offline)
RFDETR_ONNX_PATH = os.path.join(MODELS_DIR, "anti_spoof", "rf_detr", "rfdetr_small_official", "weights.onnx")
if not os.path.exists(RFDETR_ONNX_PATH):
    RFDETR_ONNX_PATH = os.path.join(MODELS_DIR, "roboflow", "k-thi-gia-s-workspace", "face-spoof-detection-liika-owgrl-1-rfdetr-small-t1", "weights.onnx")

RFDETR_MODEL_ID = "k-thi-gia-s-workspace/face-spoof-detection-liika-owgrl-1-rfdetr-small-t1"
RFDETR_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# Face Occlusion: YOLO26n Glass & Mask — Local ONNX Weights (100% Offline)
OCCLUSION_ONNX_PATH = os.path.join(MODELS_DIR, "face_occlusion", "yolo26n_glass_and_mask_official", "weights.onnx")
if not os.path.exists(OCCLUSION_ONNX_PATH):
    OCCLUSION_ONNX_PATH = os.path.join(MODELS_DIR, "roboflow", "glass-and-mask-q5de1", "2", "weights.onnx")

# Roboflow model cache — nằm trong server_module/models/roboflow/
ROBOFLOW_CACHE_DIR = os.path.join(MODELS_DIR, "roboflow")

# Ensemble fusion parameters
ENSEMBLE_CONF_THRESHOLD = 0.30       # Ngưỡng confidence tối thiểu cho mỗi model
ENSEMBLE_IOU_THRESHOLD = 0.40        # Ngưỡng IoU để ghép cặp detection giữa 2 model
ENSEMBLE_W_YOLO = 0.5                # Trọng số YOLO trong Soft Voting
ENSEMBLE_W_RFDETR = 0.5              # Trọng số RF-DETR trong Soft Voting
ENSEMBLE_SPOOF_VETO_THRESHOLD = 0.65 # Ngưỡng Spoof Veto: nếu 1 model phát hiện SPOOF >= ngưỡng → VETO

# Thư mục lưu kết quả mặc định
DEFAULT_DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data_raw")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Ngưỡng Face Detection
CONF_THRESHOLD_FACE = 0.5

# Ngưỡng Anti-Spoofing YOLO
CONF_THRESHOLD_ANTI_SPOOF = 0.25

# Ngưỡng tư thế 3D Pose (Euler Angles: Yaw, Pitch, Roll)
POSE_MAX_YAW = 35.0       # Độ xoay ngang tối đa cho phép (nới lỏng từ 32.0 -> 35.0 tránh bắt bẻ khi ngồi lệch webcam)
POSE_MAX_PITCH = 30.0     # Độ ngước lên/cúi xuống tối đa cho phép (nới lỏng từ 24.0 -> 30.0 phù hợp góc chiếu laptop)
POSE_MAX_ROLL = 22.0      # Độ nghiêng đầu tối đa cho phép (nới lỏng từ 18.0 -> 22.0)
MIN_FACE_HEIGHT = 75      # Chiều cao khuôn mặt tối thiểu trong khung hình (hạ từ 105 -> 75px cho người ngồi xa 60-70cm)
OVAL_FIT_MIN_RATIO = 0.25  # Tỷ lệ tối thiểu của chiều cao mặt so với chiều cao khung Oval (nới lỏng từ 30% -> 25%)
OVAL_FIT_MAX_RATIO = 0.92  # Tỷ lệ tối đa của chiều cao mặt so với chiều cao khung Oval (92%)

# Ngưỡng Liveness Blink (Eye Aspect Ratio - EAR)
EAR_EYE_CLOSED_THRESHOLD = 0.20   # Dưới ngưỡng này coi như mắt nhắm (bắt trọn chớp mắt tự nhiên)
EAR_EYE_OPEN_THRESHOLD = 0.22     # Trên ngưỡng này coi như mắt mở
MIN_BLINKS_REQUIRED = 1

# Ngưỡng thử thách chuyển động đầu (Head Movement Challenge)
HEAD_YAW_THRESHOLD = 14.0         # Ngưỡng góc quay trái/phải phân loại tĩnh
HEAD_PITCH_THRESHOLD = 18.0       # Ngưỡng góc cúi/ngước cho phép khi nhìn thẳng
HEAD_DELTA_YAW_THRESHOLD = 6.5    # Ngưỡng nhích nhẹ đầu thực tế từ mốc ban đầu (6.5 độ: tự nhiên, nhạy bén, chống rung camera/chớp mắt)
HEAD_DELTA_PITCH_THRESHOLD = 7.0  # Ngưỡng nhích nhẹ gật đầu tối thiểu
CHALLENGE_TIMEOUT_SECONDS = 10.0

# =====================================================================
# CHÍNH SÁCH KIỂM TRA VẬT CHE MẶT (OCCLUSION DEFENSE - POLICY A)
# =====================================================================
# Chính sách A (Strict Policy): Bắt buộc tháo TOÀN BỘ mọi loại kính (kính cận, kính râm) và khẩu trang.
STRICT_GLASSES_POLICY = True        # True: Cấm toàn bộ kính (kính cận trong suốt, kính thuốc, kính râm)
CHECK_CLEAR_GLASSES = True          # Bật thuật toán dò gọng kính cận trong suốt (Nose bridge edge & rims)
CHECK_GLASSES_GLARE = True          # Bật thuật toán dò lóa sáng tròng kính
SUNGLASSES_RATIO_THRESH = 0.58      # Tỷ lệ độ sáng hốc mắt / trán (< 0.58 coi là kính râm/kính màu)
GLASSES_BRIDGE_EDGE_THRESH = 16.0   # Ngưỡng năng lượng cạnh Sobel tại cầu sống mũi phát hiện gọng kính
MASK_DELTA_E_THRESH = 28.0          # Ngưỡng sai lệch màu LAB trán vs cằm phát hiện khẩu trang
MASK_LIP_CONTRAST_THRESH = 12.0     # Ngưỡng tương phản màu môi trên vs nhân trung phát hiện khẩu trang nude

# =====================================================================
# NODE.JS BACKEND INTEGRATION & WEBHOOK
# =====================================================================
NODEJS_WEBHOOK_URL = os.environ.get("NODEJS_WEBHOOK_URL", "http://127.0.0.1:3000/api/ekyc/result")
NODEJS_WEBHOOK_TIMEOUT = float(os.environ.get("NODEJS_WEBHOOK_TIMEOUT", "5.0"))
