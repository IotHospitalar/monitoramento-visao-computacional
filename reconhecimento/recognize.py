import os
import time
import threading
from queue import Queue

import torch
import cv2
import numpy as np
import pickle

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import insightface
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

# === Configurações de GPU e desempenho ===
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True  # otimiza conv/cuBLAS
print(f"[INFO] Usando dispositivo: {DEVICE}")

RESOLUTION      = (640, 390)    # inferência
BATCH_SIZE      = 4             # batch para YOLO
DETECT_INTERVAL = 3             # detecta a cada N frames
EMO_LABELS      = ["Anger","Contempt","Disgust","Fear","Happiness","Neutral","Sadness","Surprise"]

# === Filas para pipeline paralelo ===
frame_queue = Queue(maxsize=16)
batch_queue = Queue(maxsize=8)

class FallFaceEmotionDetector:
    def __init__(self,
                 video_path,
                 yolo_model_path='yolo11n.pt',
                 face_db_path='./trainer/face_db.pickle',
                 fall_history=18,
                 emotion_model='enet_b0_8_best_afew',
                 emotion_smooth=0.6,
                 output_path=None):
        self.video_path       = video_path
        self.yolo_conf_thresh = 0.5
        self.fall_history     = fall_history
        self.emotion_smooth   = emotion_smooth
        self.output_path      = output_path
        self.font             = cv2.FONT_HERSHEY_SIMPLEX

        # YOLO FP16 + fusão
        self.yolo = YOLO(yolo_model_path, device=DEVICE)
        self.yolo.model.half()
        self.yolo.model.fuse()

        # DeepSort tracker
        self.tracker = DeepSort(max_age=8, n_init=2,
                                nms_max_overlap=1.0,
                                embedder="mobilenet",
                                half=True)

        # Face DB
        if os.path.exists(face_db_path):
            with open(face_db_path, 'rb') as f:
                self.face_db = pickle.load(f)
        else:
            self.face_db = {}

        # InsightFace (GPU/CPU)
        providers = ["CUDAExecutionProvider"] if DEVICE.startswith("cuda") else ["CPUExecutionProvider"]
        self.face_app = insightface.app.FaceAnalysis(providers=providers)
        self.face_app.prepare(ctx_id=0 if DEVICE.startswith("cuda") else -1)

        # Emotion recognizer (ONNXRuntime GPU/CPU)
        self.emo_rec = HSEmotionRecognizer(model_name=emotion_model, providers=providers)

        # States
        self.histories     = {}
        self.fallen        = {}
        self.last_fall     = {}
        self.labels        = {}
        self.sims          = {}
        self.emos          = {}
        self.smooth_hist   = {}

    def cosine_sim(self, a, b):
        return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

    def detect_fall(self, history, already):
        if len(history) < 16: return False
        ar    = np.array([h/w for x,y,w,h in history])
        cy    = np.array([y+h/2 for x,y,w,h in history])
        area  = np.array([w*h for x,y,w,h in history])
        c1 = ar[:5].mean() > ar[-5:].mean() + ar[:5].mean()*0.28
        c2 = (cy[8:]-cy[:-8]).max()>9.5
        c3 = area[:5].mean() > area[-5:].mean()*1.20
        return (c1+c2+c3)>=2 and not already

    def reader(self):
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((idx, frame))
            idx += 1
        # sinal de fim
        frame_queue.put((None, None))
        cap.release()

    def batcher(self):
        buffer = []
        idxs   = []
        while True:
            idx, frame = frame_queue.get()
            if idx is None:
                break
            buffer.append(frame)
            idxs.append(idx)
            # empacota ao alcançar BATCH_SIZE ou DETECT_INTERVAL
            if len(buffer) >= BATCH_SIZE or (idx % DETECT_INTERVAL == 0):
                batch_queue.put((idxs.copy(), buffer.copy()))
                buffer.clear()
                idxs.clear()
        # sinal de fim
        batch_queue.put((None, None))

    def worker(self):
        writer = None
        if self.output_path:
            cap_temp = cv2.VideoCapture(self.video_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap_temp.get(cv2.CAP_PROP_FPS) or 25
            w0 = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
            h0 = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(self.output_path, fourcc, fps, (w0,h0))
            cap_temp.release()

        FPS_hist = []
        last_time = time.time()

        while True:
            idxs, batch = batch_queue.get()
            if idxs is None:
                break

            # inferência YOLO em batch
            smalls = [cv2.resize(f, RESOLUTION) for f in batch]
            with torch.cuda.amp.autocast(enabled=True):
                results = self.yolo(smalls, classes=[0])

            # processa cada frame do batch
            for (idx, frame), res in zip(zip(idxs, batch), results):
                # tracking e face/emotion/queda
                h, w = frame.shape[:2]
                dets = []
                for box in res.boxes:
                    if float(box.conf[0]) < self.yolo_conf_thresh: continue
                    x1,y1,x2,y2 = box.xyxy[0].cpu().numpy().astype(int)
                    x1 = int(x1*w/RESOLUTION[0]); y1 = int(y1*h/RESOLUTION[1])
                    x2 = int(x2*w/RESOLUTION[0]); y2 = int(y2*h/RESOLUTION[1])
                    dets.append(([x1,y1,x2-x1,y2-y1], float(box.conf[0]), 'person'))

                tracks = self.tracker.update_tracks(dets, frame=frame)
                for track in tracks:
                    if not track.is_confirmed(): continue
                    tid = track.track_id
                    x1,y1,x2,y2 = map(int, track.to_ltrb())
                    w_box,h_box = x2-x1, y2-y1
                    # histórico de queda
                    self.histories.setdefault(tid, []).append((x1,y1,w_box,h_box))
                    if len(self.histories[tid])>self.fall_history:
                        self.histories[tid].pop(0)

                    # ROI e face embedding
                    face_roi = frame[y1:y1+int(0.55*h_box), x1:x2]
                    if face_roi.size>0:
                        faces = self.face_app.get(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
                        if faces:
                            emb = faces[0].embedding
                            best_sim,best_lbl = max(
                                ((self.cosine_sim(emb,db),lbl) for lbl,db in self.face_db.items()),
                                default=(0,'Desconhecido'))
                            self.labels[tid] = best_lbl if best_sim>=0.5 else 'Desconhecido'
                            self.sims[tid]   = best_sim

                    # emoção
                    if face_roi.size>0:
                        _,scores = self.emo_rec.predict_emotions(face_roi, logits=False)
                        probs = np.array(scores, dtype=np.float32)
                        prev = self.smooth_hist.get(tid)
                        self.smooth_hist[tid] = probs if prev is None else self.emotion_smooth*probs + (1-self.emotion_smooth)*prev
                        self.emos[tid] = EMO_LABELS[int(np.argmax(self.smooth_hist[tid]))]

                    # queda
                    fall = self.detect_fall(self.histories[tid], self.track_fallen.get(tid,False))
                    if fall:
                        self.last_fall[tid] = time.time()
                    self.track_fallen[tid] = fall

                    # desenha anotações
                    color = (0,0,255) if self.track_fallen[tid] else (0,255,0)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), color,2)
                    cv2.putText(frame, f"ID:{tid}", (x1,y1-8), self.font,0.6,color,2)
                    cv2.putText(frame, f"{self.labels.get(tid,'?')}:{self.sims.get(tid,0):.2f}",
                                (x1,y1-28),self.font,0.6,(0,255,255),2)
                    cv2.putText(frame, self.emos.get(tid,''), (x1,y2+20),self.font,0.6,(255,255,0),2)
                    if self.track_fallen[tid]:
                        cv2.putText(frame, "QUEDA!", (x1,y2+40),self.font,0.8,(0,0,255),2)

                # overlay global e FPS
                alert = any(time.time()-t<2 for t in self.last_fall.values())
                if alert:
                    cv2.rectangle(frame,(0,0),(w,50),(0,0,255),-1)
                    cv2.putText(frame,"QUEDA DETECTADA!",(30,38),self.font,1.2,(255,255,255),3)
                now = time.time()
                fps = 1/(now-last_time+1e-8)
                last_time = now
                cv2.putText(frame,f"FPS:{fps:.2f}",(10,h-10),self.font,0.6,(20,200,255),2)

                # mostra e salva
                cv2.imshow("Deteccao Otimizada", frame)
                if writer:
                    writer.write(frame)
                if cv2.waitKey(1)&0xFF==27:
                    break

        cv2.destroyAllWindows()

    def run(self):
        threads = [
            threading.Thread(target=self.reader, daemon=True),
            threading.Thread(target=self.batcher, daemon=True),
            threading.Thread(target=self.worker, daemon=True)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

if __name__ == "__main__":
    detector = FallFaceEmotionDetector(video_path='video.mp4', output_path=None)
    detector.run()