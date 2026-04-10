import logging
import torch
from ultralytics.utils.autobatch import check_train_batch_size

logger = logging.getLogger(__name__)

BASE_LR: float   = 0.01
BASE_BATCH: int  = 16
LR_MIN: float    = 1e-4
LR_MAX: float    = 0.10


def resolve_batch_size(model, img_size: int) -> int:
    if not torch.cuda.is_available():
        logger.warning("GPU no detectada — usando batch=8 (CPU fallback).")
        return 8

    # check_train_batch_size estaba ausente — causa NameError
    batch = check_train_batch_size(
        model=model.model,
        imgsz=img_size,
        amp=False,          # coherente con entrenamiento FP32
    )

    # GTX 1650 4GB FP32: imgsz=416 → batch=16 seguro
    if img_size <= 416:
        resolved = min(int(batch), 16)
    else:
        resolved = 8

    resolved = max(8, resolved)
    logger.info(f"AutoBatch sugirió → {batch} | usando → {resolved} (imgsz={img_size}, FP32)")
    return resolved


def scale_learning_rate(batch_size: int) -> float:
    raw_lr = BASE_LR * (batch_size / BASE_BATCH)
    lr = round(min(max(raw_lr, LR_MIN), LR_MAX), 6)
    logger.info(
        f"Linear Scaling Rule → lr0={lr:.6f}  "
        f"(base_lr={BASE_LR}, batch={batch_size}/{BASE_BATCH})"
    )
    return lr