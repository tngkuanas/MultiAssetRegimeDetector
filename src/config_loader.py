import yaml
from pathlib import Path

def load_config():
    """
    Loads the configuration from the config.yaml file in the project root.
    """
    config_path = Path(__file__).parent.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            "config.yaml not found in the project root. "
            "Please ensure the file exists and has the correct configurations."
        )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    # Example of loading the config
    config = load_config()
    print("--- Configuration Loaded ---")
    print(config)
    print("\n--- Accessing a specific value ---")
    print(f"Symbols: {config['symbols']}")
    print(f"FRED API Key: {config['fred_api_key']}")

