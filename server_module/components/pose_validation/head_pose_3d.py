import cv2
import numpy as np


class HeadPoseEstimator:

    def __init__(self, w=640, h=480):

        self.w = w
        self.h = h

        # Mô hình nhân trắc học 3D chuẩn hóa đồng bộ với hệ tọa độ Camera OpenCV:
        # - Trục X: hướng sang phải (khóe mắt bên trái ảnh X âm, bên phải ảnh X dương)
        # - Trục Y: hướng xuống dưới (Mắt ở trên: Y âm; Mũi: Y=0; Cằm ở dưới: Y dương)
        # - Trục Z: hướng ra trước khuôn mặt (Mũi nhô cao nhất: Z=0; Mắt, Miệng, Cằm lùi ra sau: Z âm)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 1: Mũi (Nose tip)
            (0.0, 330.0, -65.0),         # 152: Cằm (Chin) - Y DƯƠNG (dưới mũi)
            (-225.0, -170.0, -135.0),    # 33: Khóe mắt phải người dùng (bên trái ảnh) - X âm, Y âm (trên mũi)
            (225.0, -170.0, -135.0),     # 263: Khóe mắt trái người dùng (bên phải ảnh) - X dương, Y âm (trên mũi)
            (-150.0, 150.0, -125.0),     # 61: Khóe miệng phải người dùng (bên trái ảnh) - X âm, Y dương (dưới mũi)
            (150.0, 150.0, -125.0)       # 291: Khóe miệng trái người dùng (bên phải ảnh) - X dương, Y dương (dưới mũi)
        ], dtype=np.float64)

        focal = w
        center = (w / 2, h / 2)

        self.camera_matrix = np.array([
            [focal, 0, center[0]],
            [0, focal, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((4, 1))

    def estimate(self, landmarks, get_point, img_w=None, img_h=None):

        try:
            nose = get_point(landmarks, 1)
            chin = get_point(landmarks, 152)
            le = get_point(landmarks, 33)   # Khóe mắt bên trái ảnh (mắt phải người dùng)
            re = get_point(landmarks, 263)  # Khóe mắt bên phải ảnh (mắt trái người dùng)
            lm = get_point(landmarks, 61)   # Khóe miệng bên trái ảnh
            rm = get_point(landmarks, 291)  # Khóe miệng bên phải ảnh

            if not all([nose, chin, le, re, lm, rm]):
                return None

            image_points = np.array([
                nose, chin, le, re, lm, rm
            ], dtype=np.float64)

            # Tự động điều chỉnh camera matrix theo kích thước khung hình thực tế
            cam_mat = self.camera_matrix
            if img_w is not None and img_h is not None and img_w > 0 and img_h > 0:
                focal = float(img_w)
                cam_mat = np.array([
                    [focal, 0.0, float(img_w) / 2.0],
                    [0.0, focal, float(img_h) / 2.0],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)

            success, rvec, tvec = cv2.solvePnP(
                self.model_points,
                image_points,
                cam_mat,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            rmat, _ = cv2.Rodrigues(rvec)

            # Phân rã Euler từ ma trận quay R đồng bộ trực tiếp với Camera Frame:
            # R xấp xỉ ma trận đơn vị khi nhìn thẳng (không bị lệch 180 độ)
            sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
            singular = sy < 1e-6

            if not singular:
                x = np.arctan2(rmat[2, 1], rmat[2, 2])
                y = np.arctan2(-rmat[2, 0], sy)
                z = np.arctan2(rmat[1, 0], rmat[0, 0])
            else:
                x = np.arctan2(-rmat[1, 2], rmat[1, 1])
                y = np.arctan2(-rmat[2, 0], sy)
                z = 0.0

            pitch = float(np.degrees(x))
            yaw = float(np.degrees(y))
            roll = float(np.degrees(z))

            # Quy ước dấu chuẩn ngân hàng & eKYC Challenge:
            # - Yaw > 0: Quay mặt sang phải người dùng (Turn Right)
            # - Yaw < 0: Quay mặt sang trái người dùng (Turn Left)
            # - Pitch > 0: Cúi mặt xuống (Head Down)
            # - Pitch < 0: Ngước mặt lên (Head Up)
            # - Roll > 0: Nghiêng đầu sang vai phải; Roll < 0: Nghiêng sang vai trái

            return {
                "yaw": round(yaw, 2),
                "pitch": round(pitch, 2),
                "roll": round(roll, 2)
            }

        except Exception:
            return None