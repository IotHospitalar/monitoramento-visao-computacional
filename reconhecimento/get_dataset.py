import cv2
import os
import csv
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Inicializa o MediaPipe Face Detection.
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def save_landmarks_csv(dataset_dir, image_filename, keypoints):
    """
    Salva (ou adiciona) os landmarks em um arquivo CSV chamado landmarks.csv.
    Caso o arquivo não exista, escreve o cabeçalho.
    """
    csv_file = os.path.join(dataset_dir, "landmarks.csv")
    header = ("filename,left_eye_x,left_eye_y,right_eye_x,right_eye_y,"
              "nose_x,nose_y,mouth_left_x,mouth_left_y,mouth_right_x,mouth_right_y\n")
    row = (f"{image_filename},"
           f"{keypoints['left_eye'][0]},{keypoints['left_eye'][1]},"
           f"{keypoints['right_eye'][0]},{keypoints['right_eye'][1]},"
           f"{keypoints['nose'][0]},{keypoints['nose'][1]},"
           f"{keypoints['mouth_left'][0]},{keypoints['mouth_left'][1]},"
           f"{keypoints['mouth_right'][0]},{keypoints['mouth_right'][1]}\n")

    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write(header)
    with open(csv_file, "a") as f:
        f.write(row)


def get_existing_count(dataset_dir, person_name):
    """
    Retorna a quantidade de arquivos que já existem no diretório,
    considerando arquivos com nomes iniciados com person_name.
    """
    count = 0
    if os.path.exists(dataset_dir):
        for f in os.listdir(dataset_dir):
            if f.startswith(f"{person_name}.") and f.lower().endswith(".jpg"):
                count += 1
    return count


def show_image(title, image, pause_time=1.0):
    """
    Exibe uma imagem usando Matplotlib.
    Converte a imagem BGR para RGB, define o título e aguarda um determinado tempo.
    """
    plt.clf()  # Limpa a figura atual
    plt.title(title)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.draw()
    plt.pause(pause_time)


class DataCaptureNeural:
    def __init__(self, source, mode="video", dataset_dir="dataset", max_width=960, max_images=20,
                 min_detection_confidence=0.5):
        """
        Inicializa a captura usando MediaPipe para detecção facial.
          - source: índice da câmera ou caminho do vídeo.
          - dataset_dir: pasta onde serão salvos os rostos.
          - max_images: quantidade máxima de imagens para serem capturadas.
        """
        self.source = source
        self.mode = mode
        self.dataset_dir = dataset_dir
        self.max_width = max_width
        self.max_images = max_images
        self.detector = mp_face_detection.FaceDetection(model_selection=0,
                                                        min_detection_confidence=min_detection_confidence)
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)

    def resize_frame(self, frame):
        height, width = frame.shape[:2]
        if width > self.max_width:
            scale = self.max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(frame, (new_width, new_height))
        return frame

    def capture_faces_from_video(self, person_name):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print("Erro: Não foi possível abrir o vídeo ou câmera.")
            return

        # Define a numeração começando a partir do que já existe
        global_count = get_existing_count(self.dataset_dir, person_name)
        print(f"[INFO] Imagens já existentes para '{person_name}': {global_count}")

        plt.ion()  # Ativa o modo interativo do Matplotlib
        fig = plt.figure("Captura de Rosto")
        print("[INFO] Iniciando captura de rostos. Feche a janela Matplotlib para encerrar.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Vídeo finalizado ou erro na leitura.")
                break

            frame = self.resize_frame(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detector.process(rgb_frame)
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    # Aplica margem de 20%
                    margin = 0.2
                    x_new = max(0, int(x - margin * w))
                    y_new = max(0, int(y - margin * h))
                    w_new = int(w * (1 + 2 * margin))
                    h_new = int(h * (1 + 2 * margin))
                    if x_new + w_new > iw:  w_new = iw - x_new
                    if y_new + h_new > ih:  h_new = ih - y_new

                    face_img = frame[y_new:y_new + h_new, x_new:x_new + w_new]
                    cv2.rectangle(frame, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 0, 0), 2)

                    global_count += 1
                    filename = f"{person_name}.{global_count}.jpg"
                    file_path = os.path.join(self.dataset_dir, filename)
                    cv2.imwrite(file_path, face_img)

                    # Extração dos keypoints via MediaPipe.
                    keypoints = {}
                    if detection.location_data.relative_keypoints:
                        kp = detection.location_data.relative_keypoints
                        left_eye = (int(kp[0].x * iw), int(kp[0].y * ih))
                        right_eye = (int(kp[1].x * iw), int(kp[1].y * ih))
                        nose = (int(kp[2].x * iw), int(kp[2].y * ih))
                        mouth_center = (int(kp[3].x * iw), int(kp[3].y * ih))
                        mouth_left = (mouth_center[0] - int(0.05 * w), mouth_center[1])
                        mouth_right = (mouth_center[0] + int(0.05 * w), mouth_center[1])
                        keypoints = {
                            "left_eye": left_eye,
                            "right_eye": right_eye,
                            "nose": nose,
                            "mouth_left": mouth_left,
                            "mouth_right": mouth_right
                        }
                        save_landmarks_csv(self.dataset_dir, filename, keypoints)

                    cv2.putText(frame, f"Capturado: {filename}", (x_new, y_new - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Se atingiu a quantidade máxima, encerra
                    if global_count >= self.max_images:
                        break

            show_image("Captura de Rosto", frame, pause_time=0.01)
            # Se o usuário fechar a janela, encerra o loop
            if not plt.fignum_exists(fig.number):
                break
        cap.release()
        plt.ioff()
        plt.close(fig)
        print(f"[INFO] Captura finalizada. Total de novas imagens salvas: {global_count}")


def process_image(image_path, person_name, dataset_dir="dataset", max_width=960):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem:", image_path)
        return
    height, width = image.shape[:2]
    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    results = detector.process(rgb_image)

    # Obtém contagem já existente para esse usuário
    global_count = get_existing_count(dataset_dir, person_name)
    count = 0

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            margin = 0.2
            x_new = max(0, int(x - margin * w))
            y_new = max(0, int(y - margin * h))
            w_new = int(w * (1 + 2 * margin))
            h_new = int(h * (1 + 2 * margin))
            if x_new + w_new > iw:  w_new = iw - x_new
            if y_new + h_new > ih:  h_new = ih - y_new

            face_img = image[y_new:y_new + h_new, x_new:x_new + w_new]
            cv2.rectangle(image, (x_new, y_new), (x_new + w_new, y_new + h_new), (255, 0, 0), 2)
            count += 1
            global_count += 1
            filename = f"{person_name}.{global_count}.jpg"
            file_path = os.path.join(dataset_dir, filename)
            cv2.imwrite(file_path, face_img)

            # Extração dos keypoints
            keypoints = {}
            if detection.location_data.relative_keypoints:
                kp = detection.location_data.relative_keypoints
                left_eye = (int(kp[0].x * iw), int(kp[0].y * ih))
                right_eye = (int(kp[1].x * iw), int(kp[1].y * ih))
                nose = (int(kp[2].x * iw), int(kp[2].y * ih))
                mouth_center = (int(kp[3].x * iw), int(kp[3].y * ih))
                mouth_left = (mouth_center[0] - int(0.05 * w), mouth_center[1])
                mouth_right = (mouth_center[0] + int(0.05 * w), mouth_center[1])
                keypoints = {
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "nose": nose,
                    "mouth_left": mouth_left,
                    "mouth_right": mouth_right
                }
                save_landmarks_csv(dataset_dir, filename, keypoints)
            cv2.putText(image, f"Capturado: {filename}", (x_new, y_new - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Exibe a imagem usando Matplotlib por 1 segundo
            plt.figure("Captura de Rosto - Imagem")
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Captura de Rosto - Imagem")
            plt.axis("off")
            plt.show(block=False)
            plt.pause(1.0)
            plt.close()
    else:
        print("Nenhum rosto detectado na imagem.")

    print(f"[INFO] Processamento concluído. Total de rostos salvos nesta imagem: {count}")


def main():
    print("==== CAPTURA DE DATASET DE ROSTOS MULTIMODAL ====")
    print("Escolha a fonte de captura:")
    print("1 - Upload de foto(s) do sistema")
    print("2 - Captura ao vivo da câmera")
    print("3 - Upload de um vídeo")
    option = input("Digite o número da opção: ").strip()
    person_name = input("Digite o nome da pessoa (ex: João): ").strip()

    if option == "1":
        # Permite informar múltiplos caminhos separados por vírgula
        image_paths = input("Digite os caminhos das fotos separados por vírgula: ").strip().split(",")
        for path in image_paths:
            process_image(path.strip(), person_name, dataset_dir="dataset")
    elif option == "2":
        capture = DataCaptureNeural(source=0, mode="camera", dataset_dir="dataset")
        capture.capture_faces_from_video(person_name)
    elif option == "3":
        video_path = input("Digite o caminho do vídeo (ex: videos/meu_video.mp4): ").strip()
        capture = DataCaptureNeural(source=video_path, mode="video", dataset_dir="dataset")
        capture.capture_faces_from_video(person_name)
    else:
        print("Opção inválida.")


if __name__ == "__main__":
    main()
