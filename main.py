import yaml
from src.data.fetch_merra import download_merra_job
from src.data.fetch_insat import download_insat_data   


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_config()

    # -----------------------------
    # MERRA (existing)
    # -----------------------------
    job_id = config["merra"]["job_id"]
    output_dir = config["merra"]["output_dir"]

    download_merra_job(job_id, output_dir)

    # -----------------------------
    # INSAT (NEW)
    # -----------------------------
    insat_config = config["insat"]

    download_insat_data(insat_config)


if __name__ == "__main__":
    main()