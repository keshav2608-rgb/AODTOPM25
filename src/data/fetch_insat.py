def download_insat_data(config: dict) -> str:
    import os
    import stat
    import paramiko
    from pathlib import Path
    from dotenv import load_dotenv

    load_dotenv()

    host = config["host"]
    port = config["port"]
    remote_dir = config["remote_dir"]
    output_dir = config["output_dir"]

    username = os.getenv("MOSDAC_USERNAME")
    password = os.getenv("MOSDAC_PASSWORD")

    # (skip if already downloaded)
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        print("[INFO] INSAT data already exists. Skipping download.")
        return str(output_path)

    # ===== YOUR ORIGINAL CODE STARTS (UNCHANGED) =====

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    def download_recursive(sftp, remote_dir, local_dir):
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        for item in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{item.filename}"
            local_path = os.path.join(local_dir, item.filename)

            if stat.S_ISDIR(item.st_mode):
                download_recursive(sftp, remote_path, local_path)
            else:
                if item.filename.endswith(".h5"):
                    if os.path.exists(local_path):
                        print(f"[SKIP] {local_path}")
                        continue

                    print(f"[DOWNLOAD] {remote_path}")
                    sftp.get(remote_path, local_path)

    try:
        transport = paramiko.Transport((host, port))
        transport.connect(username=username, password=password)

        sftp = paramiko.SFTPClient.from_transport(transport)

        print("[INFO] Connected to MOSDAC SFTP")

        download_recursive(sftp, remote_dir, output_dir)

        sftp.close()
        transport.close()

        print("[SUCCESS] INSAT download completed")

        return output_dir

    except Exception as e:
        raise RuntimeError("INSAT download failed") from e

    # ===== YOUR ORIGINAL CODE ENDS =====