"""Demo na webcam: detecta rostos (Viola-Jones) e classifica a emoção com o modelo treinado.

Uso (a partir da raiz do repositório):

    python -m src.cnn.webcam_demo --checkpoint checkpoints/resnet18.pt --camera 0

Pressione ``q`` na janela para sair. O pré-processamento do rosto reutiliza
``src.cnn.datasets.build_transform`` (a mesma transformação da validação), para casar com o treino.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from src.cnn.datasets import build_transform
from src.common.data import EMOTIONS, IMG_SIZE
from src.common.utils import detect_faces

MARGIN = 0.1  # fração da caixa acrescentada de cada lado (a FER-2013 tem um pouco de contexto)


def preprocess_face(gray: np.ndarray, box, mode: str, size: int | None) -> torch.Tensor:
    """Recorta o rosto, reduz para 48x48 e devolve o tensor de entrada [1, C, H, W]."""
    x, y, w, h = (int(v) for v in box)
    mx, my = int(w * MARGIN), int(h * MARGIN)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, gray.shape[1]), min(y + h + my, gray.shape[0])
    face = cv2.resize(gray[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return build_transform(mode, augment=False, size=size)(face).unsqueeze(0)


def predict_frame(frame_bgr, model, mode, size, device, detector=detect_faces):
    """Devolve [(caixa, classe prevista, probabilidades softmax [7])] para cada rosto do quadro."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    boxes = detector(gray)
    if len(boxes) == 0:
        return []
    batch = torch.cat([preprocess_face(gray, b, mode, size) for b in boxes]).to(device)
    with torch.inference_mode():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
    return [(tuple(int(v) for v in b), int(p.argmax()), p) for b, p in zip(boxes, probs)]


def draw_predictions(frame_bgr, predictions) -> None:
    """Desenha caixa e texto "emocao 87%" no quadro (texto ASCII: cv2.putText não faz acentos)."""
    for (x, y, w, h), cls, probs in predictions:
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{EMOTIONS[cls]} {probs[cls] * 100:.0f}%"
        cv2.putText(frame_bgr, label, (x, max(y - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default="checkpoints/resnet18.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    from src.cnn.models import load_checkpoint  # import tardio

    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"Checkpoint não encontrado: {args.checkpoint}")
    model, ckpt = load_checkpoint(args.checkpoint, device=args.device)
    mode, size = ckpt["mode"], ckpt.get("size")

    cap = cv2.VideoCapture(args.camera)
    try:
        if not cap.isOpened():
            raise SystemExit(f"Não foi possível abrir a câmera {args.camera}")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Falha ao ler quadro da câmera; encerrando.")
                break
            draw_predictions(frame, predict_frame(frame, model, mode, size, args.device))
            cv2.imshow("Emocoes (q para sair)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
