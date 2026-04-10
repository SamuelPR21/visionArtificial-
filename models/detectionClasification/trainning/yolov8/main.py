import os

# DEBE ir antes de cualquier import de torch/ultralytics
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import torch
from dotenv import load_dotenv
from ultralytics import YOLO
from src.hyperparam import resolve_batch_size, scale_learning_rate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _is_memory_error(err: RuntimeError) -> bool:
    msg = str(err).lower()
    return (
        "out of memory" in msg
        or "cuda outofmemoryerror" in msg
        or "cudnn_status_internal_error_host_allocation_failed" in msg
    )


def main():
    load_dotenv("../../../../config/.env")

    data_yaml  = os.getenv("DATA_YAML")
    model_name = os.getenv("MODEL_NAME")
    epochs     = int(os.getenv("EPOCHS"))
    img_size   = int(os.getenv("IMG_SIZE"))

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"YAML no encontrado en: {data_yaml}")
    if not model_name:
        raise ValueError("MODEL_NAME no definido en .env")

    model = YOLO(model_name)

    batch_size = resolve_batch_size(model, img_size)
    attempt_batches = []
    for b in (batch_size, max(8, batch_size // 2)):
        if b not in attempt_batches:
            attempt_batches.append(b)

    base_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        project="runs",
        name="experimento",
        lrf=0.01,
        optimizer="SGD",
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        amp=False,          
        workers=0,
        cache=False,
        resume=True,
    )

    last_error = None
    for i, attempt_batch in enumerate(attempt_batches, start=1):
        lr0 = scale_learning_rate(attempt_batch)
        logger.info(f"Intento {i}/{len(attempt_batches)} → batch={attempt_batch} | lr0={lr0}")
        try:
            model.train(**base_kwargs, batch=attempt_batch, lr0=lr0)
            logger.info("Entrenamiento completado correctamente.")
            return
        except RuntimeError as e:
            if not _is_memory_error(e):
                raise
            last_error = e
            logger.warning(
                f"Fallo por memoria en intento {i} (batch={attempt_batch}). "
                "Liberando caché CUDA."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise RuntimeError(
        "No se pudo iniciar el entrenamiento por memoria GPU insuficiente."
    ) from last_error


if __name__ == '__main__':
    main()