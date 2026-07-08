# download_models.py
"""
Downloads pretrained model files from Google Drive into the results/ folder.
Run this once after cloning the repo, before running app.py.
"""

import os
import sys

try:
    import gdown
except ImportError:
    print("❌ 'gdown' is not installed. Run: pip install gdown")
    sys.exit(1)

# -----------------------------
# Config
# -----------------------------
RESULTS_DIR = "results"
FOLDER_URL = "https://drive.google.com/drive/folders/1CZ34B5fUouCLP3hdtdB55QsOHI13AlWA"

REQUIRED_FILES = [
    "PlantVillage_selected_rf.pkl",
    "PlantVillage_selected_svm.pkl",
    "PlantVillage_ga_mask.npz",
    "PlantVillage_features_train.npz",
]

# -----------------------------
# Download
# -----------------------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Skip download if everything's already there
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(RESULTS_DIR, f))]

    if not missing:
        print("✅ All model files already present in results/. Skipping download.")
        return

    print(f"[Download] Missing files: {missing}")
    print("[Download] Fetching from Google Drive folder...\n")

    try:
        gdown.download_folder(
            url=FOLDER_URL,
            output=RESULTS_DIR,
            quiet=False,
            use_cookies=False
        )
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("👉 Check that the Drive folder is shared as 'Anyone with the link → Viewer'.")
        sys.exit(1)

    # Verify everything landed correctly
    still_missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(RESULTS_DIR, f))]

    if still_missing:
        print(f"\n⚠️  Warning: these expected files are still missing after download: {still_missing}")
        print("   Check the Drive folder contents match REQUIRED_FILES in this script.")
    else:
        print("\n✅ All required model files downloaded successfully to results/")


if __name__ == "__main__":
    main()