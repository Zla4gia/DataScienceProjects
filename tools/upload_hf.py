"""Upload hf_models/ (model repository) and hf_static/ (static Space) to Hugging Face. Both are created PRIVATE.

Run `hf auth login` first (you enter the token yourself; it is stored by huggingface_hub, never in this repo).
Usage:  python tools/upload_hf.py            (uses the logged-in account's name)
        python tools/upload_hf.py --public   (make both public; only after checking the data licences)
"""
import sys
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
private = "--public" not in sys.argv
api = HfApi()
user = api.whoami()["name"]
print("logged in as", user, "| private =", private)

models = f"{user}/skill-enhancement-models"
api.create_repo(models, repo_type="model", private=private, exist_ok=True)
api.upload_folder(folder_path=str(ROOT / "hf_models"), repo_id=models, repo_type="model", commit_message="Upload saved models for the 30 projects")
print("model repository: https://huggingface.co/" + models)

space = f"{user}/skill-enhancement"
try:
    api.create_repo(space, repo_type="space", space_sdk="static", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(ROOT / "hf_static"), repo_id=space, repo_type="space", commit_message="Static landing page")
    print("static Space: https://huggingface.co/spaces/" + space)
except Exception as e:                      # a private Space may need a paid plan; the model repository is already uploaded
    print("Static Space not created:", type(e).__name__, str(e)[:300])
