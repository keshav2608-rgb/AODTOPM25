from harmony import Client
from harmony.config import Environment
import earthaccess
import os
from pathlib import Path


def download_merra_job(job_id: str, output_dir: str = "data/raw/merra2"):

    output_path = Path(output_dir)

    #
    if output_path.exists() and any(output_path.iterdir()):
        print("[INFO] MERRA data already exists. Skipping download.")
        return str(output_path)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get credentials from environment
    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if not username or not password:
        raise ValueError("Missing Earthdata credentials in environment variables.")

    # Correct way for Harmony
    harmony_client = Client(auth=(username, password), env=Environment.PROD)

    # Login for downloads
    earthaccess.login(strategy="netrc")

    print(f"[INFO] Checking job status: {job_id}")

    job = harmony_client.status(job_id)
    status = job["status"] if isinstance(job, dict) else job.status

    print(f"[INFO] Status: {status}")

    if status not in ("successful", "complete_with_errors"):
        print("[INFO] Waiting for job to complete...")
        harmony_client.wait_for_processing(job_id, show_progress=True)

    urls = list(harmony_client.result_urls(job_id))

    if not urls:
        raise ValueError("No files returned.")

    print(f"[INFO] Downloading {len(urls)} files...")

    earthaccess.download(urls, output_dir, threads=2)

    print(f"[SUCCESS] Files saved in: {output_dir}")

    return output_dir