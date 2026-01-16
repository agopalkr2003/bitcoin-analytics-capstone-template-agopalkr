import os
import shutil

import gdown

# Configuration
# Main folder ID from: https://drive.google.com/drive/folders/1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7
FOLDER_ID = "1gizJ_n-QCnE8qrFM-BU3J_ZpaR3HCjn7"
DATA_DIR = "data"
ALLOWED_EXTENSIONS = {".parquet", ".md", ".csv"}


def download_data():
    """Downloads GDrive folder and organizes contents preserving subfolder structure."""
    print("--- Downloading Capstone Data ---")

    # Create temporary directory for download
    tmp_dir = "tmp_capstone_data"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    try:
        # Download folder content
        print("Downloading from GDrive...")
        try:
            gdown.download_folder(
                id=FOLDER_ID, output=tmp_dir, quiet=False, remaining_ok=True
            )
        except Exception as e:
            print(f"\nWarning: Download interrupted - {type(e).__name__}")
            print("Some files may have been downloaded before the error.")
            print("This is often due to Google Drive rate limiting.")
            print("Continuing with any successfully downloaded files...\n")

        # Create destination directory
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        # Move allowed files preserving subfolder structure
        print("Organizing files...")
        files_organized = 0
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ALLOWED_EXTENSIONS:
                    src_file = os.path.join(root, file)
                    # Preserve relative path structure (e.g., Coin Metrics/file.csv)
                    rel_path = os.path.relpath(src_file, tmp_dir)
                    dst_file = os.path.join(DATA_DIR, rel_path)

                    # Create destination subdirectory if needed
                    dst_dir = os.path.dirname(dst_file)
                    if not os.path.exists(dst_dir):
                        os.makedirs(dst_dir)

                    print(f"  Keeping: {rel_path}")
                    shutil.move(src_file, dst_file)
                    files_organized += 1
                else:
                    print(f"  Skipping: {file}")

        if files_organized == 0:
            print("\nNo files were downloaded. Google Drive may be rate limiting.")
            print("Try again later or download manually from:")
            print(f"  https://drive.google.com/drive/folders/{FOLDER_ID}")
        else:
            print(f"\nSuccessfully organized {files_organized} file(s).")

    finally:
        # Clean up temporary directory
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    print("--- Finished downloading ---\n")


def main():
    download_data()
    print("Data download and organization complete.")


if __name__ == "__main__":
    main()
