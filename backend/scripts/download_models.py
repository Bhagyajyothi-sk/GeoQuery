import os
import urllib.request
import logging
from pathlib import Path

# Add backend to sys path so we can import config
import sys
_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("download_models")

def download_file(url: str, dest_path: Path):
    if not url:
        logger.warning(f"No URL provided for {dest_path.name}, skipping.")
        return
        
    if dest_path.exists():
        logger.info(f"File already exists at {dest_path}, skipping download.")
        return

    logger.info(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)
    logger.info(f"Successfully downloaded {dest_path.name}")

def main():
    logger.info("Starting model download process...")
    
    remoteclip_url = os.environ.get("REMOTECLIP_MODEL_URL")
    faiss_url = os.environ.get("FAISS_INDEX_URL")
    
    project_root = Path(__file__).resolve().parents[2]
    
    # RemoteCLIP
    clip_path = settings.remoteclip_model_path
    clip_dest = Path(clip_path) if Path(clip_path).is_absolute() else project_root / clip_path
    
    # FAISS
    faiss_path = settings.faiss_index_path
    faiss_dest = Path(faiss_path) if Path(faiss_path).is_absolute() else project_root / faiss_path
    
    if not settings.mock_search:
        if remoteclip_url:
            download_file(remoteclip_url, clip_dest)
        else:
            logger.warning("MOCK_SEARCH=false but REMOTECLIP_MODEL_URL is not set!")
            
        if faiss_url:
            download_file(faiss_url, faiss_dest)
        else:
            logger.warning("MOCK_SEARCH=false but FAISS_INDEX_URL is not set!")
    else:
        logger.info("MOCK_SEARCH=true, skipping model downloads.")

if __name__ == "__main__":
    main()
