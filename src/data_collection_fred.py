import pandas as pd
from fredapi import Fred
from config_loader import load_config

def get_fred_data(series_ids, start_date, end_date):
    """
    Fetches macroeconomic data from the FRED API.

    :param series_ids: A dictionary mapping FRED series IDs to desired column names.
    :param start_date: The start date for the data in 'YYYY-MM-DD' format.
    :param end_date: The end date for the data in 'YYYY-MM-DD' format.
    :return: A pandas DataFrame with the requested data, forward-filled.
    """
    try:
        config = load_config()
        api_key = config.get("fred_api_key")

        if not api_key or api_key == "PASTE_YOUR_NEW_FRED_API_KEY_HERE":
            print(
                "Error: FRED API Key is not set in config.yaml. Please paste your key. Skipping FRED data collection."
            )
            return None

        fred = Fred(api_key=api_key)
        
        macro_df = pd.DataFrame()
        for series_id, name in series_ids.items():
            series_data = fred.get_series(series_id, start_date, end_date)
            macro_df[name] = series_data

        macro_df.ffill(inplace=True)
        macro_df.bfill(inplace=True)
        
        print(f"Successfully fetched {len(series_ids)} FRED series.")
        return macro_df

    except Exception as e:
        print(
            "Could not fetch FRED data. "
            "Please check your API key in config.yaml and series IDs."
        )
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # --- Example Usage ---
    config = load_config()
    api_key = config.get("fred_api_key")

    if not api_key or api_key == "PASTE_YOUR_NEW_FRED_API_KEY_HERE":
        print("Warning: FRED_API_KEY not set in config.yaml. Example usage skipped.")
    else:
        series_to_fetch = config.get("fred_series", {})
        start = config.get("start_date", "2008-01-01")
        end = config.get("end_date", "2024-01-01")

        macro_data = get_fred_data(
            series_ids=series_to_fetch, start_date=start, end_date=end
        )

        if macro_data is not None:
            print("\n--- Fetched Macroeconomic Data ---")
            print(macro_data.head())
            print("\n...")
            print(macro_data.tail())
