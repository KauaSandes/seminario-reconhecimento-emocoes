"""Baixa e valida os dados do projeto. É idempotente: o que já existe e está válido não é baixado de novo.

Uso (da raiz do repositório, com o .venv ativo)::

    python scripts/download_data.py              # FER-2013 e CK+
    python scripts/download_data.py --only fer2013
    python scripts/download_data.py --only ckplus

FER-2013
    Vem da competição oficial do Kaggle, a única fonte que mantém a divisão
    Training / PublicTest / PrivateTest. Antes de rodar, uma pessoa precisa
    (1) aceitar as regras da competição no site do Kaggle e
    (2) ter o token da API em ``~/.kaggle/kaggle.json`` ou nas variáveis
    ``KAGGLE_USERNAME`` e ``KAGGLE_KEY``.
    O script só confere se o token existe, sem ler nem gravar seu conteúdo. Se o
    download oficial falhar, ele para em vez de trocar de fonte.

CK+
    O acesso é restrito: a equipe pede aos autores, e os zips chegam por e-mail. O
    script cria ``data/raw/ckplus/LEIA-ME.txt``. Se os zips estiverem lá, ele descompacta
    e confere a estrutura. Se não estiverem, avisa e termina sem erro. Nada é baixado de
    espelhos.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.data import (  # noqa: E402
    CKPLUS_RAW_DIR,
    FER2013_COLUMNS,
    FER2013_CSV,
    FER2013_SIZES,
    find_ckplus_dirs,
    read_fer2013_csv,
)
from src.common.utils import ensure_utf8_stdio  # noqa: E402

COMPETITION = "challenges-in-representation-learning-facial-expression-recognition-challenge"
COMPETITION_URL = f"https://www.kaggle.com/c/{COMPETITION}"
#: SHA-256 do fer2013.csv obtido da competição oficial, preenchido após o primeiro download.
#: Se for ``None``, o hash só é gravado e não é comparado.
EXPECTED_SHA256: str | None = None
#: Saída do kaggle que indica token inválido ou regras não aceitas.
AUTH_ERROR = re.compile(r"\b(401|403)\b|unauthori[sz]ed|forbidden", re.IGNORECASE)

CKPLUS_ZIPS: tuple[str, ...] = (
    "extended-cohn-kanade-images.zip",
    "Emotion_labels.zip",
    "FACS_labels.zip",
    "Landmarks.zip",
)
CKPLUS_README = """\
CK+ (Extended Cohn-Kanade), acesso restrito
============================================

O CK+ NÃO tem download público oficial e NÃO deve ser baixado de espelhos (Kaggle,
GitHub, Google Drive de terceiros) por causa da licença.

1. Alguém da equipe pede o acesso aos autores pela página de recursos do laboratório
   do Prof. Jeffrey Cohn (Affect Analysis Group, Universidade de Pittsburgh), citando
   Lucey et al. (2010).
2. Se o pedido for aprovado, os arquivos chegam por e-mail. Coloque nesta pasta
   (data/raw/ckplus/) os zips:
       extended-cohn-kanade-images.zip   (obrigatório: cohn-kanade-images/Sxxx/yyy/*.png)
       Emotion_labels.zip                (obrigatório: Emotion/Sxxx/yyy/*_emotion.txt)
       FACS_labels.zip                   (opcional)
       Landmarks.zip                     (opcional)
3. Rode de novo, a partir da raiz do repositório:
       python scripts/download_data.py --only ckplus
       python -m src.common.prepare_data
   O primeiro comando descompacta e confere a estrutura. O segundo recorta os rostos,
   grava data/processed/ckplus.npz e gera splits/ckplus_folds.csv (GroupKFold por pessoa).
"""


class DownloadError(RuntimeError):
    """Erro que interrompe o script com uma mensagem para a pessoa."""


# ------------------------------------------------------------------ utilidades
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def kaggle_help_message(reason: str) -> str:
    return f"""
[FER-2013] {reason}

Dois passos manuais, feitos por uma pessoa e nunca pelo script:
  1. Entrar no Kaggle e aceitar as regras da competição em
       {COMPETITION_URL}/rules
     (botão "I Understand and Accept"; o Kaggle pode pedir verificação por telefone).
  2. Gerar o token da API em https://www.kaggle.com/settings (seção API,
     "Create New Token") e salvar o kaggle.json baixado em
       {Path.home() / '.kaggle' / 'kaggle.json'}
     ou definir as variáveis de ambiente KAGGLE_USERNAME e KAGGLE_KEY.
     Nunca cole o token no chat, no código ou no repositório (kaggle.json está no .gitignore).

Depois rode de novo: python scripts/download_data.py
Não use versões em pastas de imagens (como "msambare/fer2013"): elas perderam a divisão
PublicTest/PrivateTest de que o projeto precisa.
"""


def find_kaggle_credentials(env: dict[str, str] | None = None, home: Path | None = None) -> str | None:
    """Diz onde está a credencial do Kaggle, sem ler o conteúdo, ou ``None`` se não houver."""
    env = dict(os.environ) if env is None else env
    home = Path.home() if home is None else home
    if env.get("KAGGLE_API_TOKEN"):
        return "variável KAGGLE_API_TOKEN"
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return "variáveis KAGGLE_USERNAME e KAGGLE_KEY"
    config_dir = Path(env["KAGGLE_CONFIG_DIR"]) if env.get("KAGGLE_CONFIG_DIR") else home / ".kaggle"
    for name in ("kaggle.json", "access_token"):
        if (config_dir / name).is_file():
            return str(config_dir / name)
    return None


def find_kaggle_executable() -> list[str]:
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    scripts = Path(sys.executable).parent
    for name in ("kaggle.exe", "kaggle"):
        if (scripts / name).is_file():
            return [str(scripts / name)]
    raise DownloadError("o pacote kaggle não está instalado. Rode: pip install -r requirements.txt")


# -------------------------------------------------------------------- FER-2013
def download_fer2013_zip(fer_dir: Path) -> Path:
    """Roda ``kaggle competitions download`` e devolve o caminho do zip."""
    cmd = [*find_kaggle_executable(), "competitions", "download", "-c", COMPETITION, "-p", str(fer_dir)]
    print(f"[FER-2013] baixando da competição oficial: {COMPETITION}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = f"{proc.stdout}\n{proc.stderr}"
    if AUTH_ERROR.search(output):
        raise DownloadError(kaggle_help_message(
            "O Kaggle recusou o download (401/403): o token é inválido ou as regras da "
            "competição ainda não foram aceitas nesta conta.\n  Saída do kaggle: "
            + output.strip().replace("\n", "\n    ")))
    if proc.returncode != 0:
        raise DownloadError(
            f"[FER-2013] o download oficial falhou (código {proc.returncode}). Não vou trocar de "
            f"fonte; avise a equipe.\n  Saída do kaggle:\n    " + output.strip().replace("\n", "\n    "))
    zip_path = fer_dir / f"{COMPETITION}.zip"
    if not zip_path.is_file():
        zips = sorted(fer_dir.glob("*.zip"))
        if not zips:
            raise DownloadError(f"[FER-2013] o kaggle terminou sem erro, mas nenhum zip apareceu em {fer_dir}")
        zip_path = zips[0]
    return zip_path


def extract_fer2013(zip_path: Path, out_csv: Path) -> str:
    """Extrai ``fer2013.csv`` do zip da competição e devolve a fonte usada.

    Prefere ``fer2013.tar.gz`` (lido por stream, só o membro ``fer2013.csv``, sem
    ``extractall``). Se não houver o tar.gz, usa ``icml_face_data.csv``, normaliza os nomes
    das colunas (que vêm com espaço no começo) e grava na ordem ``emotion,pixels,Usage``.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_csv.with_suffix(".csv.part")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        tar_name = next((n for n in names if n.endswith("fer2013.tar.gz")), None)
        icml_name = next((n for n in names if n.endswith("icml_face_data.csv")), None)
        if tar_name is not None:
            with tarfile.open(fileobj=io.BytesIO(zf.read(tar_name)), mode="r:gz") as tf:
                member = next((m for m in tf.getmembers() if m.isfile() and m.name.endswith("fer2013.csv")), None)
                if member is None:
                    raise DownloadError(f"[FER-2013] {tar_name} não contém fer2013.csv")
                src = tf.extractfile(member)
                assert src is not None
                with open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            source = f"{zip_path.name} -> {tar_name} -> {member.name}"
        elif icml_name is not None:
            import pandas as pd

            with zf.open(icml_name) as f:
                df = pd.read_csv(f, dtype={" pixels": str, " Usage": str, "pixels": str, "Usage": str})
            df.columns = df.columns.str.strip()
            df[list(FER2013_COLUMNS)].to_csv(tmp, index=False, lineterminator="\n")
            source = f"{zip_path.name} -> {icml_name} (colunas normalizadas)"
        else:
            raise DownloadError(f"[FER-2013] {zip_path.name} não contém fer2013.tar.gz nem icml_face_data.csv "
                                f"(conteúdo: {names})")
    os.replace(tmp, out_csv)
    return source


def write_checksum(csv_path: Path, digest: str, source: str) -> Path:
    path = csv_path.parent / "CHECKSUM.txt"
    path.write_text(
        f"{digest}  {csv_path.name}\n"
        f"# algoritmo: SHA-256\n"
        f"# fonte: {source}\n"
        f"# gerado em: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n",
        encoding="utf-8")
    return path


def check_expected_hash(digest: str) -> None:
    if EXPECTED_SHA256 is None:
        print("[FER-2013] EXPECTED_SHA256 ainda não foi fixado no script; nada a comparar.")
    elif digest == EXPECTED_SHA256:
        print("[FER-2013] SHA-256 confere com o esperado (mesmo arquivo nas duas máquinas).")
    else:
        print(f"[FER-2013] ATENÇÃO: o SHA-256 ({digest}) é diferente do esperado ({EXPECTED_SHA256}).\n"
              "           O arquivo pode estar corrompido ou ter vindo de outra fonte. Avise a equipe.")


def validate(csv_path: Path) -> None:
    df, X = read_fer2013_csv(csv_path, FER2013_SIZES)
    counts = df["Usage"].value_counts()
    print(f"[FER-2013] válido: {len(df)} linhas, imagens {X.shape[1]}x{X.shape[2]}, "
          + ", ".join(f"{k}={counts[k]}" for k in FER2013_SIZES))


def prepare_fer2013(fer_dir: Path, force: bool = False) -> Path:
    csv_path = fer_dir / FER2013_CSV.name
    if csv_path.is_file() and not force:
        print(f"[FER-2013] {csv_path} já existe; validando...")
        try:
            validate(csv_path)
        except ValueError as exc:
            raise DownloadError(f"{exc}\nApague {csv_path} e rode de novo para extrair outra vez.") from exc
        digest = sha256_file(csv_path)
        if not (fer_dir / "CHECKSUM.txt").is_file():
            write_checksum(csv_path, digest, "arquivo já existente")
        print(f"[FER-2013] já existe e é válido. SHA-256: {digest}")
        check_expected_hash(digest)
        return csv_path

    fer_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(fer_dir.glob("*.zip"))
    if zips and not force:
        zip_path = zips[0]
        print(f"[FER-2013] zip já baixado ({zip_path.name}); pulando o download.")
    else:
        where = find_kaggle_credentials()
        if where is None:
            raise DownloadError(kaggle_help_message("Não encontrei o token da API do Kaggle."))
        print(f"[FER-2013] token do Kaggle encontrado em: {where}")
        zip_path = download_fer2013_zip(fer_dir)

    try:
        source = extract_fer2013(zip_path, csv_path)
    except (zipfile.BadZipFile, tarfile.TarError, EOFError) as exc:
        raise DownloadError(f"[FER-2013] {zip_path.name} está corrompido ou incompleto ({exc}). "
                            f"Apague o arquivo e rode de novo.") from exc
    print(f"[FER-2013] extraído de {source}")
    try:
        validate(csv_path)
    except ValueError as exc:
        raise DownloadError(f"[FER-2013] o arquivo extraído não passou na validação:\n{exc}") from exc
    digest = sha256_file(csv_path)
    write_checksum(csv_path, digest, source)
    print(f"[FER-2013] SHA-256: {digest}  (gravado em {fer_dir / 'CHECKSUM.txt'})")
    check_expected_hash(digest)
    return csv_path


# ------------------------------------------------------------------------- CK+
def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    """Extrai os membros do zip em ``dest``, ignorando ``__MACOSX`` e caminhos que saiam de ``dest``."""
    dest = dest.resolve()
    n = 0
    for member in zf.infolist():
        name = member.filename
        if name.startswith("__MACOSX/") or "/._" in name or name.startswith("._"):
            continue
        target = (dest / name).resolve()
        if dest not in target.parents and target != dest:
            raise DownloadError(f"[CK+] caminho suspeito no zip: {name}")
        zf.extract(member, dest)
        n += 1
    return n


def prepare_ckplus(ck_dir: Path) -> bool:
    """Cria a pasta e o LEIA-ME e, se os zips existirem, descompacta. Devolve ``True`` se o CK+ estiver pronto."""
    ck_dir.mkdir(parents=True, exist_ok=True)
    (ck_dir / "LEIA-ME.txt").write_text(CKPLUS_README, encoding="utf-8")

    if find_ckplus_dirs(ck_dir) is None:
        present = [z for z in CKPLUS_ZIPS if (ck_dir / z).is_file()]
        for name in present:
            print(f"[CK+] descompactando {name}...")
            with zipfile.ZipFile(ck_dir / name) as zf:
                print(f"[CK+]   {_safe_extract(zf, ck_dir)} arquivos")

    dirs = find_ckplus_dirs(ck_dir)
    if dirs is None:
        missing = [z for z in CKPLUS_ZIPS[:2] if not (ck_dir / z).is_file()]
        print(f"[CK+] ainda não disponível em {ck_dir} (faltam: {', '.join(missing) or 'as pastas esperadas'}).\n"
              f"[CK+] Veja {ck_dir / 'LEIA-ME.txt'}. Seguindo sem o CK+.")
        return False
    images_dir, emotion_dir = dirs
    n_subjects = sum(1 for p in images_dir.iterdir() if p.is_dir())
    n_labels = sum(1 for _ in emotion_dir.rglob("*_emotion.txt"))
    print(f"[CK+] pronto: {images_dir} ({n_subjects} pessoas) e {emotion_dir} ({n_labels} rótulos).")
    return True


# ------------------------------------------------------------------------ main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", choices=("fer2013", "ckplus"), help="preparar só uma das bases")
    parser.add_argument("--force", action="store_true", help="baixar e extrair a FER-2013 de novo")
    parser.add_argument("--fer-dir", type=Path, default=FER2013_CSV.parent)
    parser.add_argument("--ckplus-dir", type=Path, default=CKPLUS_RAW_DIR)
    args = parser.parse_args(argv)
    ensure_utf8_stdio()

    failed = False
    if args.only in (None, "fer2013"):
        try:
            prepare_fer2013(args.fer_dir, force=args.force)
        except DownloadError as exc:
            print(exc, file=sys.stderr)
            failed = True
    if args.only in (None, "ckplus"):
        try:
            prepare_ckplus(args.ckplus_dir)  # se o CK+ faltar, só avisa e não conta como erro
        except DownloadError as exc:
            print(exc, file=sys.stderr)
            failed = True
    if failed:
        return 1
    print("\nPróximo passo: python -m src.common.prepare_data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
