"""
Local ONNX Model Runners for RF-DETR Small and YOLO26n Occlusion.
Chạy trực tiếp file ONNX trong thư mục models cục bộ, KHÔNG gọi lên cloud Roboflow.
100% Offline, không phụ thuộc internet hay API key.
"""

import os
import sys
from types import SimpleNamespace
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np


def _patch_onnxruntime_compatibility():
    """
    Khắc phục triệt để lỗi Version Mismatch trong ONNX Runtime trên Windows:
    'onnxruntime.capi.onnxruntime_pybind11_state.InferenceSession' object has no attribute 'is_webgpu_graph_capture_enabled'
    Xảy ra khi các file Python của onnxruntime mới hơn so với thư viện nhị phân C++ pybind11 (.pyd).
    """
    try:
        import onnxruntime as ort
        # 1. Bypass hàm kiểm tra WebGPU graph capture trên Python InferenceSession
        if hasattr(ort.InferenceSession, "_validate_graph_capture_run_api"):
            ort.InferenceSession._validate_graph_capture_run_api = lambda *args, **kwargs: None
            
        # 2. Bổ sung fallback method vào C++ pybind11 class
        try:
            capi_sess = ort.capi.onnxruntime_pybind11_state.InferenceSession
            if not hasattr(capi_sess, "is_webgpu_graph_capture_enabled"):
                setattr(capi_sess, "is_webgpu_graph_capture_enabled", lambda *args, **kwargs: False)
        except Exception:
            pass
    except Exception:
        pass

_patch_onnxruntime_compatibility()


def _safe_ort_run(session, input_feed, output_names=None):
    """
    Chạy inference session an toàn, tự động bắt lỗi và fallback sang C++ _sess.run
    nếu gặp lỗi AttributeError 'is_webgpu_graph_capture_enabled'.
    """
    try:
        return session.run(output_names, input_feed)
    except AttributeError as ae:
        if "is_webgpu_graph_capture_enabled" in str(ae) and hasattr(session, "_sess"):
            return session._sess.run(output_names, input_feed)
        raise
    except Exception as e:
        if "is_webgpu_graph_capture_enabled" in str(e) and hasattr(session, "_sess"):
            return session._sess.run(output_names, input_feed)
        raise


class RFDETROnnxRunner:
    """
    Trình chạy suy luận trực tiếp cho mô hình RF-DETR Small từ file weights.onnx cục bộ.
    Không phụ thuộc vào cloud Roboflow hay API key.
    """
    def __init__(self, onnx_path: str, class_names_path: Optional[str] = None):
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.input_shape = (512, 512)
        self.class_names = ["background_class83422", "real", "spoof"]
        
        # Load danh sách nhãn lớp từ file class_names.txt nếu có
        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                if lines:
                    self.class_names = lines
        else:
            dir_path = os.path.dirname(onnx_path)
            candidate_cls = os.path.join(dir_path, "class_names.txt")
            if os.path.exists(candidate_cls):
                with open(candidate_cls, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        self.class_names = lines

        self._init_session()

    def _init_session(self):
        _patch_onnxruntime_compatibility()
        try:
            import onnxruntime as ort
        except ImportError:
            print("\n" + "="*60)
            print("[THÔNG BÁO QUAN TRỌNG] Hệ thống cần thư viện 'onnxruntime' để chạy model ONNX.")
            print("Vui lòng mở terminal và chạy lệnh cài đặt:")
            print("    pip install onnxruntime")
            print("Hoặc (tăng tốc GPU Windows):")
            print("    pip install onnxruntime-directml")
            print("="*60 + "\n")
            raise ImportError("Thiếu thư viện onnxruntime. Vui lòng chạy 'pip install onnxruntime'.")

        # Tối ưu Execution Providers: ưu tiên GPU DirectML / CUDA, fallback CPU
        available_providers = ort.get_available_providers()
        providers = []
        for p in ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
            if p in available_providers:
                providers.append(p)
        if not providers:
            providers = ["CPUExecutionProvider"]

        # Session options để tăng tốc độ suy luận
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)

        self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
        # Lấy kích thước đầu vào nếu ONNX khai báo shape tĩnh
        shape = self.session.get_inputs()[0].shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_shape = (shape[3], shape[2])  # (w, h)
        print(f"[RFDETROnnxRunner] Đã nạp thành công file ONNX cục bộ: {self.onnx_path}")
        print(f"[RFDETROnnxRunner] Input: '{self.input_name}' {self.input_shape} | Providers: {providers} | Classes: {self.class_names}")

    def infer(self, image: np.ndarray, conf_threshold: float = 0.20) -> SimpleNamespace:
        """
        Giao diện infer tương thích 100% với output của Roboflow Inference SDK.
        Trả về: SimpleNamespace(predictions=[SimpleNamespace(x, y, width, height, confidence, class_name), ...])
        """
        if self.session is None or image is None or image.size == 0:
            return SimpleNamespace(predictions=[])

        h_orig, w_orig = image.shape[:2]
        target_w, target_h = self.input_shape

        # Preprocessing: BGR -> RGB, Resize 512x512, chuẩn hóa [0, 1], NCHW float32
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        blob = (resized.transpose(2, 0, 1) / 255.0).astype(np.float32)[None, ...]

        # Chạy suy luận ONNX (sử dụng _safe_ort_run để tự động khắc phục lỗi version mismatch)
        outputs = _safe_ort_run(self.session, {self.input_name: blob})

        # Phân tách boxes và class logits từ output
        boxes = None
        logits = None
        for out in outputs:
            if len(out.shape) == 3:
                if out.shape[-1] == 4:
                    boxes = out[0]
                elif out.shape[-1] >= len(self.class_names) or out.shape[-1] in (2, 3, 80, 91):
                    logits = out[0]

        if boxes is None or logits is None:
            if len(outputs) >= 2:
                if outputs[0].shape[-1] == 4:
                    boxes = outputs[0][0]
                    logits = outputs[1][0]
                else:
                    logits = outputs[0][0]
                    boxes = outputs[1][0]

        if boxes is None or logits is None:
            return SimpleNamespace(predictions=[])

        # Áp dụng Sigmoid cho class logits của DETR
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

        predictions = []
        num_queries = len(boxes)
        for i in range(num_queries):
            query_scores = scores[i]
            # Nếu 3 classes: [background, real, spoof]
            if len(query_scores) == 3:
                p_real = float(query_scores[1])
                p_spoof = float(query_scores[2])
                if p_real >= p_spoof:
                    best_conf = p_real
                    cls_name = "real"
                else:
                    best_conf = p_spoof
                    cls_name = "spoof"
            else:
                best_cls_idx = int(np.argmax(query_scores))
                best_conf = float(query_scores[best_cls_idx])
                cls_name = self.class_names[best_cls_idx] if best_cls_idx < len(self.class_names) else f"class_{best_cls_idx}"
                if "background" in cls_name.lower():
                    continue

            if best_conf < conf_threshold:
                continue

            cx, cy, pw, ph = boxes[i]
            # Chuyển đổi về tọa độ pixel trên khung hình gốc
            px = float(cx * w_orig)
            py = float(cy * h_orig)
            p_width = float(pw * w_orig)
            p_height = float(ph * h_orig)

            predictions.append(SimpleNamespace(
                x=px,
                y=py,
                width=p_width,
                height=p_height,
                confidence=best_conf,
                class_name=cls_name
            ))

        return SimpleNamespace(predictions=predictions)


class YOLOOcclusionOnnxRunner:
    """
    Trình chạy suy luận trực tiếp cho mô hình YOLO26n Glass & Mask từ file weights.onnx cục bộ.
    Chạy 100% OFFLINE trực tiếp qua ONNXRuntime, KHÔNG dùng Ultralytics (tránh lỗi AutoUpdate),
    KHÔNG gọi lên cloud Roboflow hay phụ thuộc internet.
    """
    def __init__(self, onnx_path: str, class_names_path: Optional[str] = None):
        self.onnx_path = onnx_path
        self.class_names = {0: "glass", 1: "mask", 2: "no_glass", 3: "no_mask"}
        self.ort_session = None
        self.input_name = None
        self.input_shape = (640, 640)

        if class_names_path and os.path.exists(class_names_path):
            with open(class_names_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
                if lines:
                    self.class_names = {i: name for i, name in enumerate(lines)}
        else:
            candidate_cls = os.path.join(os.path.dirname(onnx_path), "class_names.txt")
            if os.path.exists(candidate_cls):
                with open(candidate_cls, "r", encoding="utf-8") as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        self.class_names = {i: name for i, name in enumerate(lines)}

        self._init_model()

    def _init_model(self):
        _patch_onnxruntime_compatibility()
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Thiếu thư viện onnxruntime. Vui lòng chạy 'pip install onnxruntime'.")

        available = ort.get_available_providers()
        providers = [p for p in ["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"] if p in available] or ["CPUExecutionProvider"]
        
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = max(1, os.cpu_count() // 2 if os.cpu_count() else 2)

        self.ort_session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=providers)
        self.input_name = self.ort_session.get_inputs()[0].name
        
        shape = self.ort_session.get_inputs()[0].shape
        if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
            self.input_shape = (shape[3], shape[2])
        print(f"[YOLOOcclusionOnnxRunner] Đã nạp thành công file ONNX trực tiếp qua ONNXRuntime (100% Offline): {self.onnx_path}")
        print(f"[YOLOOcclusionOnnxRunner] Input: '{self.input_name}' {self.input_shape} | Providers: {providers} | Classes: {self.class_names}")

    def infer(self, image: np.ndarray, conf_threshold: float = 0.25) -> SimpleNamespace:
        """
        Giao diện infer tương thích 100% với Roboflow Inference SDK.
        """
        if image is None or image.size == 0 or self.ort_session is None:
            return SimpleNamespace(predictions=[])

        try:
            h_orig, w_orig = image.shape[:2]
            target_w, target_h = self.input_shape
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            blob = (resized.transpose(2, 0, 1) / 255.0).astype(np.float32)[None, ...]

            outputs = _safe_ort_run(self.ort_session, {self.input_name: blob})
            out = outputs[0]
            predictions = []
            scale_x = w_orig / float(target_w)
            scale_y = h_orig / float(target_h)

            # Dạng End-to-End ONNX [1, 300, 6] -> [x1, y1, x2, y2, conf, cls_id]
            if len(out.shape) == 3 and out.shape[-1] == 6:
                for det in out[0]:
                    x1, y1, x2, y2, conf, cls_id = det
                    if conf < conf_threshold:
                        continue
                    cx = ((float(x1) + float(x2)) / 2.0) * scale_x
                    cy = ((float(y1) + float(y2)) / 2.0) * scale_y
                    w = (float(x2) - float(x1)) * scale_x
                    h = (float(y2) - float(y1)) * scale_y
                    cls_name = self.class_names.get(int(cls_id), f"class_{int(cls_id)}").lower().strip()
                    predictions.append(SimpleNamespace(
                        x=float(cx),
                        y=float(cy),
                        width=float(w),
                        height=float(h),
                        confidence=float(conf),
                        class_name=cls_name
                    ))
                return SimpleNamespace(predictions=predictions)

            # Fallback dạng standard YOLO [1, 8, 8400] hoặc [1, 8400, 8]
            if len(out.shape) == 3:
                preds = out[0]
                if preds.shape[0] < preds.shape[1]:
                    preds = preds.T  # shape: [8400, 4 + num_classes]
                for row in preds:
                    box = row[:4]
                    scores = row[4:]
                    cls_id = int(np.argmax(scores))
                    conf = float(scores[cls_id])
                    if conf < conf_threshold:
                        continue
                    cx, cy, w, h = box
                    px = float(cx * scale_x)
                    py = float(cy * scale_y)
                    pw = float(w * scale_x)
                    ph = float(h * scale_y)
                    cls_name = self.class_names.get(cls_id, f"class_{cls_id}").lower().strip()
                    predictions.append(SimpleNamespace(
                        x=px,
                        y=py,
                        width=pw,
                        height=ph,
                        confidence=conf,
                        class_name=cls_name
                    ))
                return SimpleNamespace(predictions=predictions)

        except Exception as e:
            print(f"[YOLOOcclusionOnnxRunner] ORT infer error: {e}")

        return SimpleNamespace(predictions=[])
