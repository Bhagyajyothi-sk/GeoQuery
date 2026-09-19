import os
import sys
import logging
from pathlib import Path
import requests

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.core.config import settings

def setup_logger():
    logger = logging.getLogger("geoquery_downloader")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger

logger = setup_logger()

def download_file(url: str, dest_path: Path, item_name: str):
    logger.info(f"[GeoQuery] Checking {item_name}...")
    if dest_path.exists():
        logger.info(f"[GeoQuery] {item_name} ready.")
        return

    logger.info(f"[GeoQuery] Downloading {item_name}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"[GeoQuery] {item_name} ready.")
    except Exception as e:
        logger.error(f"[GeoQuery] ERROR: Failed to download {item_name} from {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        sys.exit(1)

def main():
    if settings.mock_search:
        logger.info("[GeoQuery] MOCK_SEARCH=true. Skipping model downloads.")
        return

    remoteclip_url = os.environ.get("REMOTECLIP_MODEL_URL")
    faiss_url = os.environ.get("FAISS_INDEX_URL")
    faiss_meta_url = os.environ.get("FAISS_METADATA_URL")
    
    project_root = Path(__file__).resolve().parents[2]
    clip_dest = Path(settings.remoteclip_model_path) if Path(settings.remoteclip_model_path).is_absolute() else project_root / settings.remoteclip_model_path
    faiss_dest = Path(settings.faiss_index_path) if Path(settings.faiss_index_path).is_absolute() else project_root / settings.faiss_index_path
    faiss_meta_dest = Path(settings.tile_metadata_path) if Path(settings.tile_metadata_path).is_absolute() else project_root / settings.tile_metadata_path
    
    if not clip_dest.exists() and not remoteclip_url:
        logger.error("[GeoQuery] ERROR: REMOTECLIP_MODEL_URL environment variable is missing. "
                     "Please set it to the URL of RemoteCLIP-ViT-B-32.pt")
        sys.exit(1)
        
    if not faiss_dest.exists() and not faiss_url:
        logger.error("[GeoQuery] ERROR: FAISS_INDEX_URL environment variable is missing. "
                     "Please set it to the URL of satellite.index")
        sys.exit(1)

    if not faiss_meta_dest.exists() and not faiss_meta_url:
        logger.error("[GeoQuery] ERROR: FAISS_METADATA_URL environment variable is missing. "
                     "Please set it to the URL of tile_metadata.json")
        sys.exit(1)
    
    # RemoteCLIP
    download_file(remoteclip_url, clip_dest, "RemoteCLIP model")
    
    # FAISS
    download_file(faiss_url, faiss_dest, "FAISS index")
    
    # FAISS Metadata
    download_file(faiss_meta_url, faiss_meta_dest, "FAISS metadata")

if __name__ == "__main__":
    main()
