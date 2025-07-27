import os
import time
from collections import deque

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import insightface

# ====== Configuração de desempenho =======
RESOLUCAO = (640, 390)   # (width, height) - Reduza para ganhar FPS (ex: 320x192 ou 416x256)
global yolo_model_path
# =========================================
yolo_model_path = "yolo11n.pt"  # Use yolov8n.pt para máxima velocidade.
# Se quiser, pode testar yolov8n-seg.pt ou yolov8n-int8.pt (quantizado)
# =========================================

# Checagem do dispositivo
try:
    import torch
    yolo_uses_gpu = torch.cuda.is_available()
    if yolo_uses_gpu:
        print("[INFO] YOLO (PyTorch) está usando GPU:", torch.cuda.get_device_name(0))
    else:
        print("[INFO] YOLO (PyTorch) está usando CPU.")
except ImportError:
    yolo_uses_gpu = False
    print("[ALERTA] PyTorch não instalado.")

# Checagem do InsightFace/MXNet
try:
    import mxnet
    n_gpus = mxnet.context.num_gpus()
    if n_gpus > 0:
        print(f"[INFO] InsightFace (MXNet) detectou {n_gpus} GPU(s).")
        insightface_uses_gpu = True
    else:
        print("[INFO] InsightFace (MXNet) está usando CPU.")
        insightface_uses_gpu = False
except ImportError:
    insightface_uses_gpu = False

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    epsilon = 1e-8
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + epsilon)

def detect_fall_ultimate(history: deque, fallen_already=False, debug=False) -> bool:
    N = len(history)
    if N < 16: return False
    aspect_ratios = np.array([h/(w+1e-6) for x, y, w, h in history])
    cy = np.array([y + h/2 for x, y, w, h in history])
    areas = np.array([w*h for x, y, w, h in history])
    ar_start = np.mean(aspect_ratios[:5])
    ar_end = np.mean(aspect_ratios[-5:])
    delta_ar = ar_start - ar_end
    criteria_ar_drop = (ar_start > 1.3) and (delta_ar > (ar_start * 0.28))
    max_dy = np.max(cy[8:] - cy[:-8])
    criteria_fast_fall = max_dy > 9.5
    area_start = np.mean(areas[:5])
    area_end = np.mean(areas[-5:])
    area_ratio = area_start / (area_end + 1e-6)
    criteria_area = area_ratio > 1.20
    std_cy_post = np.std(cy[-7:])
    std_ar_post = np.std(aspect_ratios[-7:])
    criteria_immobile = std_cy_post < 7 and std_ar_post < 0.13
    n_flat = np.sum(aspect_ratios[-7:] < 1.2)
    criteria_flat = n_flat >= 4
    criteria = [criteria_ar_drop, criteria_fast_fall, criteria_area, criteria_immobile, criteria_flat]
    fall_detected = (sum(criteria) >= 3) and not fallen_already
    return fall_detected

class FallFaceDetectorDeepSort:
    def __init__(self, video_path, face_db_path="./trainer/face_db.pickle", fall_history=18):
        self.video_path = video_path
        self.fall_history = fall_history
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.yolo_model = YOLO(yolo_model_path)
        print(f"[INFO] YOLO modelo carregado ({yolo_model_path}). Dispositivo: {'GPU' if yolo_uses_gpu else 'CPU'}.")

        self.yolo_conf_threshold = 0.5
        self.tracker = DeepSort(max_age=8, n_init=2, nms_max_overlap=1.0, embedder="mobilenet", half=True)

        self.face_db_path = face_db_path
        if os.path.exists(self.face_db_path):
            with open(self.face_db_path, "rb") as f:
                self.face_db = pickle.load(f)
            print("[INFO] Banco de dados facial carregado.")
        else:
            self.face_db = {}
            print("[ALERTA] Banco de dados facial não encontrado!")

        self.recognizer = insightface.app.FaceAnalysis()
        try:
            self.recognizer.prepare(ctx_id=0)
            print("[INFO] InsightFace está usando GPU para reconhecimento facial.")
        except Exception:
            print("[ERRO] InsightFace GPU não disponível, usando CPU.")
            self.recognizer.prepare(ctx_id=-1)

        self.track_histories = {}
        self.track_fallen = {}
        self.last_fall_time = {}
        self.track_face_label = {}

    def get_label_from_embedding(self, embedding: np.ndarray):
        best_label = "Desconhecido"
        best_sim = -1.0
        for label, db_emb in self.face_db.items():
            sim = cosine_similarity(embedding, db_emb)
            if sim > best_sim:
                best_sim = sim
                best_label = label
        if best_sim >= 0.5:
            return best_label, best_sim
        return "Desconhecido", best_sim

    def process(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("[ERRO] Não foi possível abrir o vídeo.")
            return

        w1, h1 = RESOLUCAO
        print(f"[INFO] Resolução de trabalho otimizada: {w1}x{h1}")

        frame_count = 0
        last_time = time.time()
        FPS_SMOOTH = 0
        show_fall_alert = False
        fall_alert_timer = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Fim do vídeo ou erro.")
                break
            if frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
                print("[ERRO] Frame inválido.")
                break
            # Redimensiona frame para RESOLUCAO
            frame = cv2.resize(frame, (w1, h1))
            frame_count += 1

            now = time.time()
            elapsed = now - last_time
            last_time = now
            fps = 1.0 / (elapsed + 1e-8)
            FPS_SMOOTH = FPS_SMOOTH * 0.95 + fps * 0.05 if frame_count > 10 else fps

            # YOLO só para 1 frame. Para batch, veja abaixo.
            results = self.yolo_model(frame, classes=[0])[0] # classes=[0] para detectar apenas pessoas
            detections = []
            for box in results.boxes:
                if int(box.cls[0]) == 0:  # Só pessoa!
                    xyxy = box.xyxy[0]
                    if hasattr(xyxy, "cpu"):
                        xyxy = xyxy.cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(box.conf[0])
                    if conf < self.yolo_conf_threshold:
                        continue
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

            tracks = self.tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                w, h = x2 - x1, y2 - y1
                box = (x1, y1, w, h)

                if track_id not in self.track_histories:
                    self.track_histories[track_id] = deque(maxlen=self.fall_history)
                    self.track_fallen[track_id] = False
                self.track_histories[track_id].append(box)

                h_face = int(h * 0.55)
                face_roi = frame[y1:y1 + h_face, x1:x2]
                label = "Desconhecido"
                similarity = 0.0
                if face_roi.shape[0] > 30 and face_roi.shape[1] > 30:
                    faces = self.recognizer.get(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                    if faces:
                        face = faces[0]
                        label, similarity = self.get_label_from_embedding(face.embedding)
                        fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                        abs_fx1 = x1 + fx1
                        abs_fy1 = y1 + fy1
                        abs_fx2 = x1 + fx2
                        abs_fy2 = y1 + fy2
                        cv2.rectangle(frame, (abs_fx1, abs_fy1), (abs_fx2, abs_fy2), (0, 255, 255), 2)
                self.track_face_label[track_id] = label

                fall = detect_fall_ultimate(self.track_histories[track_id], self.track_fallen[track_id], debug=False)
                if fall:
                    if not self.track_fallen[track_id]:
                        print(f"[EVENTO] Queda detectada para ID {track_id}!")
                        self.last_fall_time[track_id] = time.time()
                    self.track_fallen[track_id] = True
                else:
                    self.track_fallen[track_id] = False

                color = (0, 0, 255) if self.track_fallen[track_id] else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID:{track_id}", (x1, y1-8), self.font, 0.8, color, 2)
                cv2.putText(frame, f"{label}:{similarity:.2f}", (x1, y1-28), self.font, 0.8, (0, 255, 255), 2)
                if self.track_fallen[track_id]:
                    cv2.putText(frame, "QUEDA!", (x1, y2+25), self.font, 0.9, (0,0,255), 2)

            if any(time.time() - t < 2 for t in self.last_fall_time.values()):
                show_fall_alert = True
                fall_alert_timer = time.time()
            if show_fall_alert:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (0,0,255), -1)
                cv2.putText(frame, "QUEDA DETECTADA!", (30, 38), self.font, 1.4, (255,255,255), 4)
                if time.time() - fall_alert_timer > 2.0:
                    show_fall_alert = False

            cv2.putText(frame, f"FPS: {FPS_SMOOTH:.2f}", (10, frame.shape[0]-10), self.font, 0.9, (20,200,255), 2)

            cv2.imshow("YOLO + DeepSORT: Face, Corpos, Queda", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        print("\n[RESUMO FINAL]")
        print(f"YOLO rodou usando: {'GPU' if yolo_uses_gpu else 'CPU'}")
        print(f"InsightFace rodou usando: {'GPU' if insightface_uses_gpu else 'CPU'}")

def main(video_path=0, model_path=None):
    # Define o modelo do YOLO globalmente
    global yolo_model_path

    if model_path:
        yolo_model_path = model_path

    print("="*70)
    print("===== OTIMIZADO PARA MÁXIMO DESEMPENHO (FPS) =====")
    print(f"Resolução reduzida: {RESOLUCAO[0]}x{RESOLUCAO[1]}")
    print(f"Modelo YOLO: {yolo_model_path}")
    print("Se for testar outros modelos, troque o caminho da variável yolo_model_path.")
    print("Aumente ainda mais o FPS diminuindo a resolução ou rodando em placa de vídeo (GPU).")
    print("="*70)
    print("Precisa ser instalado: Drivers NVIDIA instalados,  CUDA Toolkit compatível, cuDNN compatível, mxnet-cu121 (ou ajuste para o CUDA instalado), torch, torchvision, torchaudio (PyTorch com CUDA)")
    detector = FallFaceDetectorDeepSort(
        video_path=video_path,
        face_db_path="./trainer/face_db.pickle",
        fall_history=18
    )
    detector.process()

if __name__ == "__main__":
    main()
