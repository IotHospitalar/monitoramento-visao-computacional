import os
import random
import torch
import argparse

# GPU detection snippet
try:
    import mxnet
    n_gpus = mxnet.context.num_gpus()
    if n_gpus > 0:
        print(f"[INFO] InsightFace (MXNet) detectou {n_gpus} GPU(s).")
except ImportError:
    print("[INFO] MXNet não está instalado.")

def check_gpu():
    torch_gpu = torch.cuda.is_available()
    print(f"[INFO] PyTorch GPU available: {torch_gpu}")
    print()

# Importações relativas ao pacote (reconhecimento)
from .get_dataset import main as capture_main
from .train import enroll_faces as train_main
from .recognize import main as recognize_main

VIDEOS_DIR = "./videos"

def interactive_recognition_menu(model_path=None, video=None, random_choice=False):
    if video or random_choice:
        recognize_main_cli(video=video, model_path=model_path)
        return
    print("Reconhecimento: escolha uma opção:")
    print("[1] - Webcam")
    print("[2] - Vídeo (caminho)")
    print("[3] - Escolher amostra disponível")
    print("[4] - Amostra aleatória")
    choice = input("Opção: ").strip()
    if choice == '1':
        recognize_main(yolo_model_path=model_path)
    elif choice == '2':
        path = input("Caminho do vídeo: ").strip()
        recognize_main_cli(video=path, model_path=model_path)
    elif choice == '3':
        samples = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi'))]
        for i, s in enumerate(samples, 1): print(f"{i}. {s}")
        idx = int(input("Escolha o número: ")) - 1
        recognize_main_cli(video=os.path.join(VIDEOS_DIR, samples[idx]), model_path=model_path)
    elif choice == '4':
        samples = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi'))]
        choice = random.choice(samples)
        print(f"Vídeo aleatório selecionado: {choice}")
        recognize_main_cli(video=os.path.join(VIDEOS_DIR, choice), model_path=model_path)
    else:
        print("Opção inválida.")

def recognize_main_cli(video=None, model_path=None):
    if model_path:
        os.environ['YOLO_MODEL_PATH'] = model_path
    recognize_main(video_path=video, yolo_model_path=model_path)


def parse_args():
    parser = argparse.ArgumentParser(prog='python -m reconhecimento')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--capture', action='store_true', help='Capturar imagens (arquivo1)')
    group.add_argument('--train', action='store_true', help='Treinar dataset (arquivo2)')
    group.add_argument('--recognize', action='store_true', help='Executar reconhecimento (arquivo3)')
    parser.add_argument('--video', type=str, help='Caminho de vídeo para reconhecimento')
    parser.add_argument('--random', action='store_true', help='Selecionar vídeo aleatório para reconhecimento')
    parser.add_argument('--model', type=str, help='Caminho do modelo YOLO (ex.: yolov11n.pt)')

    # Novas opções para emoções
    parser.add_argument("--input", help="vídeo de entrada (ex: emotions.mp4)")
    parser.add_argument("--output", default=None, help="salvar vídeo anotado (mp4)")
    parser.add_argument("--emotion-model", default="enet_b0_8_best_afew", help="modelo HSEmotion-ONNX (veja README)")
    parser.add_argument("--min_conf", type=float, default=0.5, help="confiança mínima detecção de rosto")
    parser.add_argument("--smooth", type=float, default=0.6, help="suavização temporal 0-1 (0 desliga)")

    return parser.parse_args()


def main():
    args = parse_args()
    check_gpu()

    if args.capture:
        capture_main()
    elif args.train:
        train_main()
    elif args.recognize:
        video = None
        if args.random:
            samples = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith(('.mp4', '.avi'))]
            video = os.path.join(VIDEOS_DIR, random.choice(samples))
        else:
            video = args.video
        recognize_main_cli(video=video, model_path=args.model)
    else:
        print("=== Reconhecimento Facial e Detecção de Quedas ===")
        print("[1] - Capturar imagens")
        print("[2] - Treinar dataset")
        print("[3] - Executar reconhecimento")
        opt = input("Escolha uma opção: ").strip()
        if opt == '1':
            capture_main()
        elif opt == '2':
            train_main()
        elif opt == '3':
            interactive_recognition_menu(model_path=args.model)
        else:
            print("Opção inválida. Saindo.")

if __name__ == '__main__':
    main()
