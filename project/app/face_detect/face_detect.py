"""
人脸检测模块 —— YuNet (OpenCV DNN)
比 Haar Cascade 更准更快，支持侧脸、遮挡、暗光
"""
import os
import cv2
import numpy as np

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATH = os.path.join(_MODEL_DIR, "face_detection_yunet.onnx")

# 全局单例
_detector = None
_detector_backend = None
_input_size = None
_haar_detector = None


def _load_haar_detector():
    """当 YuNet 不可用时，回退到 OpenCV 自带 Haar 级联。"""
    global _haar_detector
    if _haar_detector is not None:
        return _haar_detector if not _haar_detector.empty() else None

    haar_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    if not os.path.exists(haar_path):
        return None

    _haar_detector = cv2.CascadeClassifier(haar_path)
    return _haar_detector if not _haar_detector.empty() else None


def init_detector(width=640, height=480):
    """初始化检测器（首次调用或画面尺寸变化时）"""
    global _detector, _detector_backend, _input_size
    _input_size = (width, height)
    _detector = None
    _detector_backend = None

    if hasattr(cv2, "FaceDetectorYN") and os.path.exists(_MODEL_PATH):
        try:
            _detector = cv2.FaceDetectorYN.create(
                _MODEL_PATH, "", (width, height),
                score_threshold=0.6,      # 置信度阈值
                nms_threshold=0.3,        # 重叠抑制
                top_k=5000,
            )
            _detector_backend = "yunet"
            return
        except Exception:
            _detector = None
            _detector_backend = None

    if _load_haar_detector() is not None:
        _detector_backend = "haar"


def load_img(path):
    """安全读图（兼容中文路径、PNG透明图）"""
    arr = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def find_face(img):
    """返回最大人脸坐标 (x, y, w, h)，未检测到返回 None"""
    global _detector, _detector_backend, _input_size
    h, w = img.shape[:2]
    if _detector is None or _input_size != (w, h):
        init_detector(w, h)

    if _detector_backend == "yunet" and _detector is not None:
        _, faces = _detector.detect(img)
        if faces is None or len(faces) == 0:
            return None
        # 取置信度最高的人脸
        best = max(faces, key=lambda f: f[-1])
        x, y, w_box, h_box = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        if w_box < 20 or h_box < 20:  # 太小的忽略
            return None
        return (x, y, w_box, h_box)

    haar = _load_haar_detector()
    if haar is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    faces = haar.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    x, y, w_box, h_box = max(faces, key=lambda f: f[2] * f[3])
    if w_box < 20 or h_box < 20:
        return None
    return (int(x), int(y), int(w_box), int(h_box))


def get_face(img):
    """检测人脸 → 裁剪 + 留白 10% → Resize 224×224"""
    face = find_face(img)
    if face is None:
        return None
    x, y, w, h = face
    pad = int(min(w, h) * 0.10)
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (224, 224))


def draw_box(img, face, label="", color=(0, 255, 0)):
    """在图上画人脸框 + 标签"""
    if face is None: return img
    x, y, w, h = face
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    if label:
        cv2.putText(img, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


# ====================== 批量测试 ======================
if __name__ == "__main__":
    INPUT = "my_images"
    OUTPUT = "aligned_faces"
    os.makedirs(INPUT, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    for f in os.listdir(INPUT):
        try:
            img = load_img(os.path.join(INPUT, f))
            if img is None:
                print(f"⚠️ 无法读取：{f}")
                continue
            face = get_face(img)
            if face is not None:
                out_path = os.path.join(OUTPUT, f)
                cv2.imencode(".jpg", face)[1].tofile(out_path)
                print(f"✅ {f}")
            else:
                print(f"❌ 未检测到人脸：{f}")
        except Exception as e:
            print(f"❌ 失败：{f} ({e})")
    print("\n🎉 全部完成！")
