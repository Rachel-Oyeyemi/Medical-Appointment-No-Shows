"""Download the official dataset from Kaggle."""
from __future__ import annotations
import argparse, logging, shutil, subprocess
from pathlib import Path
from utils import configure_logging
LOGGER=logging.getLogger(__name__); HANDLE="joniarroba/noshowappointments"

def kagglehub_download(output: Path) -> Path:
    import kagglehub
    cache=Path(kagglehub.dataset_download(HANDLE)); files=list(cache.rglob("*.csv"))
    if not files: raise FileNotFoundError("No CSV in KaggleHub download")
    output.mkdir(parents=True,exist_ok=True); target=output/"appointments.csv"; shutil.copy2(files[0],target); return target

def cli_download(output: Path) -> Path:
    output.mkdir(parents=True,exist_ok=True); subprocess.run(["kaggle","datasets","download","-d",HANDLE,"-p",str(output),"--unzip"],check=True)
    files=list(output.glob("*.csv"));
    if not files: raise FileNotFoundError("No CSV produced by Kaggle CLI")
    target=output/"appointments.csv"; files[0].replace(target) if files[0]!=target else None; return target

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",default="data/raw"); p.add_argument("--source",choices=["auto","kagglehub","cli"],default="auto"); a=p.parse_args(); configure_logging(); errors=[]
    methods=[kagglehub_download,cli_download] if a.source=="auto" else [kagglehub_download if a.source=="kagglehub" else cli_download]
    for method in methods:
        try: LOGGER.info("Downloaded to %s",method(Path(a.output_dir))); return
        except Exception as exc: errors.append(f"{method.__name__}: {exc}")
    raise RuntimeError("Authenticate to Kaggle before downloading. "+" | ".join(errors))
if __name__=="__main__": main()
