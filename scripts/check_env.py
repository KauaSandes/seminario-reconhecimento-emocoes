"""Verifica o ambiente PyTorch da Frente A (RTX 5060 / Colab).

Imprime versões e dados da GPU, roda uma multiplicação de matrizes, um passo com
``torch.autocast`` e mede forward + backward de uma ResNet-18 (lote 64, 224x224)
em fp32 e em precisão mista, com o pico de memória.

Uso:
    python scripts/check_env.py                 # GPU, se houver
    python scripts/check_env.py --device cpu --iters 2 --batch-size 8   # só para testar o script

A série RTX 50 (Blackwell, sm_120) exige PyTorch compilado com CUDA 12.8+.
O erro "no kernel image is available for execution on the device" indica versão errada.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time

import torch
import torchvision

KERNEL_IMAGE_HINT = (
    "\n>>> 'no kernel image is available': este PyTorch não tem kernels para esta GPU.\n"
    ">>> RTX 50 (Blackwell) precisa de PyTorch com CUDA 12.8+. Reinstale com:\n"
    ">>>   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128\n"
)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_versions(device: torch.device) -> None:
    section("Versões")
    print(f"Python       : {sys.version.split()[0]} ({platform.system()} {platform.release()})")
    print(f"PyTorch      : {torch.__version__}")
    print(f"torchvision  : {torchvision.__version__}")
    print(f"CUDA (build) : {torch.version.cuda}")
    print(f"cuDNN        : {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None}")
    print(f"CUDA ok      : {torch.cuda.is_available()}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        cap = f"sm_{props.major}{props.minor}"
        print(f"GPU          : {props.name}")
        print(f"VRAM total   : {props.total_memory / 2**30:.2f} GiB")
        print(f"Capacidade   : {cap}")
        archs = torch.cuda.get_arch_list()
        print(f"Arqs do build: {' '.join(archs)}")
        if cap not in archs and not any(a.startswith("compute_") for a in archs):
            print(f"AVISO: {cap} não está entre as arquiteturas deste build." + KERNEL_IMAGE_HINT)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def check_matmul(device: torch.device) -> None:
    section("Multiplicação de matrizes")
    a = torch.randn(2048, 2048, device=device)
    b = torch.randn(2048, 2048, device=device)
    c = a @ b
    sync(device)
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    sync(device)
    dt = (time.perf_counter() - t0) / 10
    ref = (a[:4].cpu() @ b.cpu())
    err = (c[:4].cpu() - ref).abs().max().item()
    print(f"2048x2048 fp32: {dt * 1e3:.2f} ms ({2 * 2048**3 / dt / 1e12:.2f} TFLOP/s), erro máx vs CPU {err:.2e}")


def check_autocast(device: torch.device) -> None:
    section("Passo com torch.autocast")
    dtype = amp_dtype(device)
    model = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.ReLU(), torch.nn.Linear(512, 7)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    x = torch.randn(64, 512, device=device)
    y = torch.randint(0, 7, (64,), device=device)
    with torch.autocast(device_type=device.type, dtype=dtype):
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    sync(device)
    print(f"dtype {str(dtype).replace('torch.', '')}: saída {out.dtype}, perda {loss.item():.4f}, GradScaler={use_scaler}")


def bench_resnet(device: torch.device, batch_size: int, size: int, iters: int, warmup: int, amp: bool) -> dict:
    model = torchvision.models.resnet18(weights=None, num_classes=7).to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    dtype = amp_dtype(device)
    use_scaler = amp and device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)
    x = torch.randn(batch_size, 3, size, size, device=device)
    y = torch.randint(0, 7, (batch_size,), device=device)

    def step() -> None:
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(warmup):
        step()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    sync(device)
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return {"ms_per_step": dt * 1e3, "img_per_s": batch_size / dt, "peak_mib": peak}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # acentos no console do Windows

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA indisponível. Confira o driver NVIDIA e se o PyTorch é o build cu128 (não o +cpu).")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print_versions(device)
    try:
        check_matmul(device)
        check_autocast(device)
        section(f"ResNet-18 forward+backward (lote {args.batch_size}, {args.size}x{args.size})")
        rows = []
        for amp in (False, True):
            label = f"AMP {str(amp_dtype(device)).replace('torch.', '')}" if amp else "fp32"
            r = bench_resnet(device, args.batch_size, args.size, args.iters, args.warmup, amp)
            rows.append((label, r))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    except RuntimeError as exc:
        if "no kernel image" in str(exc):
            print(KERNEL_IMAGE_HINT, file=sys.stderr)
        raise

    print("\n| Precisão | ms/passo | imagens/s | pico de memória (MiB) |")
    print("|---|---|---|---|")
    for label, r in rows:
        print(f"| {label} | {r['ms_per_step']:.1f} | {r['img_per_s']:.0f} | {'n/d' if r['peak_mib'] != r['peak_mib'] else f"{r['peak_mib']:.0f}"} |")
    print("\nOK: ambiente funcionando.")


if __name__ == "__main__":
    main()
