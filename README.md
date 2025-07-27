# Monitoramento com Visão Computacional

## Descrição

Este projeto implementa um sistema de **detecção de quedas** em vídeos, utilizando:

* **YOLO** (v11) para detecção de pessoas
* **Deep SORT** para rastreamento de múltiplos objetos
* **InsightFace** (MXNet) para reconhecimento facial
* **EmotiEffLib** para reconhecimento de expressões

O código está organizado como um **pacote Python** denominado `reconhecimento` e fornece três funcionalidades principais:

1. **Captura de imagens** para geração do dataset de rostos
2. **Treinamento** em cima do dataset gerado
3. **Reconhecimento** em vídeos pré-existentes ou na webcam

## Estrutura de Pastas

O projeto está organizado da seguinte forma:

```
├── reconhecimento/             # Código-fonte do pacote Python
├── videos/                     # Vídeos de entrada para reconhecimento
├── dataset/                    # Imagens brutas para enrollment
│   ├── pessoa1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── pessoa2.jpg
├── trainer/
│   └── face_db.pickle          # Banco de embeddings gerado
├── yolo11n.pt                  # Modelos YOLO pré-treinados
├── yolo8n.pt                   
├── requirements.txt            # Pacotes necessários para rodar o código
└── enet_b0_8_best_afew.pt      # Modelo EmotiEffLib pré-treinado
```

## Dependências

Certifique-se de ter instalado o Python 3.8+ e, de preferência, um ambiente virtual.

```bash
pip install -r requirements.txt
```
ou
```bash
pip install ultralytics deep_sort_realtime opencv-python torch torchvision torchaudio insightface mxnet mediapipe matplotlib pillow numpy onnxruntime hsemotion-onnx
```

## Como Rodar

A partir do diretório raiz do projeto (aquele que contém `reconhecimento/`), você pode executar o sistema de duas formas:

### 1. Modo não interativo (via flags)

* **Capturar imagens para o dataset**:

  ```bash
  python -m reconhecimento --capture
  ```

* **Treinar o reconhecimento facial no dataset**:

  ```bash
  python -m reconhecimento --train
  ```

* **Executar reconhecimento**:

  * Em um arquivo específico:

    ```bash
    python -m reconhecimento --recognize --video ./videos/nome_do_video.mp4 --model yolo11n.pt
    ```
  * Em um vídeo aleatório da pasta `videos/`:

    ```bash
    python -m reconhecimento --recognize --random --model yolov8n.pt
    ```

> **Observação**: o parâmetro `--model` aponta para o arquivo `.pt` do YOLO (v8 ou v11) disponível no diretório raiz. O modelo padrão é `yolo11n.pt`.

### 2. Modo interativo

Simplesmente execute:

```bash
python -m reconhecimento
```

O menu interativo apresentará as opções:

```
[1] - Capturar imagens
[2] - Treinar dataset
[3] - Executar reconhecimento
```

Basta digitar `1`, `2` ou `3` conforme desejado e seguir as instruções na tela.

## 🔬 Suporte a GPU

**YOLOv8 e InsightFace suportam GPU, mas dependem do seu ambiente.**

### Como garantir o uso de GPU:

Instale o pacote `onnxruntime-gpu` no lugar de `onnxruntime`.

1. **Drivers NVIDIA** instalados e CUDA disponível.
2. **PyTorch** instalado com CUDA.

   * Veja seu CUDA com `nvidia-smi`.
   * Exemplo para CUDA 11.7:

     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
     ```
3. **MXNet** instalado na versão CUDA correta:

   * Para CUDA 11.7:

     ```bash
     pip install mxnet-cu117
     ```
   * Para CPU apenas:

     ```bash
     pip install mxnet
     ```
4. Rode seu script normalmente. O console informará se GPU está ativa.

## Observações Finais

* Ajuste os paths dentro dos scripts se a estrutura de pastas for alterada.
* Para melhor desempenho em treinamento e inferência, instale drivers e suporte CUDA adequados.
* Logs de treinamento e arquivos gerados ficarão dentro da pasta `trainer/`.

---