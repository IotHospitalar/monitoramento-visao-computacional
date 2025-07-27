import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import pickle
import insightface


def augment_image(image):
    """
    Gera versões aumentadas da imagem para melhorar a robustez dos embeddings.
    São aplicadas as seguintes transformações:
      - Flip horizontal
      - Aumento de brilho
      - Redução de brilho
      - Rotação leve (10º)
    Retorna uma lista de imagens (incluindo a original).
    """
    augmentations = [image]  # Inclui imagem original

    # Flip horizontal
    flip = cv2.flip(image, 1)
    augmentations.append(flip)

    # Aumento de brilho (+30)
    bright_up = cv2.convertScaleAbs(image, alpha=1.0, beta=30)
    augmentations.append(bright_up)

    # Redução de brilho (-30)
    bright_down = cv2.convertScaleAbs(image, alpha=1.0, beta=-30)
    augmentations.append(bright_down)

    # Rotação leve (10 graus)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmentations.append(rotated)

    return augmentations


def enroll_faces(dataset_dir="dataset", output_db="trainer/face_db.pickle", embedding_threshold=0.0):
    """
    Processa recursivamente as imagens do dataset para extrair embeddings faciais usando InsightFace,
    aplicando data augmentation para aumentar a robustez do embedding.

    Se as imagens estiverem em subpastas (ex.: dataset/cr7/...), o nome da subpasta é usado como rótulo.
    Caso estejam na raiz do diretório, o rótulo é extraído a partir do nome do arquivo (ex.: "Joao.1.jpg").

    Os embeddings de cada pessoa são agrupados (média) e salvos em um arquivo pickle.
    """
    if not os.path.exists(dataset_dir):
        print(f"[ERRO] Diretório {dataset_dir} não encontrado!")
        return

    valid_extensions = (".jpg", ".jpeg", ".png")
    print(f"[INFO] Iniciando o enrollment utilizando o dataset: {dataset_dir}")

    # Inicializa o InsightFace para detecção, alinhamento e extração de embeddings.
    recognizer = insightface.app.FaceAnalysis()
    recognizer.prepare(ctx_id=0)

    embeddings_dict = {}  # dicionário: label -> lista de embeddings

    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue

            # Define o rótulo: se a imagem estiver na raiz, usa a parte antes do primeiro ponto;
            # caso contrário, utiliza o nome da subpasta.
            if os.path.abspath(root) == os.path.abspath(dataset_dir):
                label = file.split(".")[0]
            else:
                label = os.path.basename(root)

            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[AVISO] Não foi possível carregar a imagem: {img_path}")
                continue

            # Aplica data augmentation (incluindo a versão original)
            images_aug = augment_image(img)

            for idx, img_variant in enumerate(images_aug):
                rgb_img = cv2.cvtColor(img_variant, cv2.COLOR_BGR2RGB)
                faces = recognizer.get(rgb_img)
                if faces:
                    face = faces[0]  # usa a primeira face detectada
                    if hasattr(face, "det_score") and face.det_score < embedding_threshold:
                        print(f"[AVISO] Baixa confiança na detecção em {img_path} (variante {idx}), ignorando.")
                        continue
                    embedding = face.embedding
                    embeddings_dict.setdefault(label, []).append(embedding)
                    print(f"[INFO] Embedding extraído para '{label}' a partir de {img_path} (variante {idx})")
                else:
                    print(f"[AVISO] Nenhuma face detectada em {img_path} (variante {idx})")

    # Calcula a média dos embeddings para cada rótulo
    face_db = {}
    for label, embeds in embeddings_dict.items():
        if embeds:
            face_db[label] = np.mean(embeds, axis=0)
        else:
            print(f"[AVISO] Sem embeddings válidos para '{label}'.")

    # Cria o diretório de saída, se necessário
    output_dir = os.path.dirname(output_db)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Salva o banco de dados facial em formato pickle
    with open(output_db, "wb") as f:
        pickle.dump(face_db, f)
    print(f"[INFO] Banco de dados facial salvo em {output_db}")


if __name__ == "__main__":
    print("==== ENROLLMENT DE ROSTOS ====")
    dataset_dir = input("Digite o caminho do dataset (pressione Enter para 'dataset'): ").strip() or "dataset"
    output_db = input(
        "Digite o caminho para salvar o banco de dados (pressione Enter para 'trainer/face_db.pickle'): ").strip() or "trainer/face_db.pickle"
    enroll_faces(dataset_dir=dataset_dir, output_db=output_db)
