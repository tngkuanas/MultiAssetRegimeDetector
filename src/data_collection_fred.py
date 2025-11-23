import sys
import os
import pandas as pd
from fredapi import Fred

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import FRED_API_KEY # Import API key from config.py

def get_fred_data(series_ids, start_date, end_date): # Removed api_key parameter
    """
    Fetches macroeconomic data from the FRED API.

    :param series_ids: A dictionary mapping FRED series IDs to desired column names.
                       e.g., {'DGS10': '10y_treasury_yield'}
    :param start_date: The start date for the data in 'YYYY-MM-DD' format.
    :param end_date: The end date for the data in 'YYYY-MM-DD' format.
    :return: A pandas DataFrame containing the requested macroeconomic data,
             forward-filled to handle missing values on non-business days.
    """
    try:
        if FRED_API_KEY == 'YOUR_API_KEY_HERE' or not FRED_API_KEY:
            print("FRED API Key is not set in config.py. Skipping FRED data collection.")
            return None

        fred = Fred(api_key=FRED_API_KEY) # Use imported API key
        
        # Create a DataFrame to hold all the series data
        macro_df = pd.DataFrame()

        # Fetch each series and add it to the DataFrame
        for series_id, name in series_ids.items():
            series_data = fred.get_series(series_id, start_date, end_date)
            macro_df[name] = series_data

        # FRED data often has NaNs on weekends/holidays. Forward-fill to propagate
        # the last known value to these days.
        macro_df.ffill(inplace=True)
        # Backward-fill any remaining NaNs at the beginning of the series
        macro_df.bfill(inplace=True)
        
        print(f"Successfully fetched {len(series_ids)} FRED series.")
        return macro_df

    except Exception as e:
        print(f"Could not fetch FRED data. Please check your API key in config.py and series IDs.")
        print(f"Error: {e}")
        return None

if __name__ == '__main__':
    # --- Example Usage ---
    # IMPORTANT: Ensure your FRED_API_KEY is set in config.py

    if FRED_API_KEY == 'YOUR_API_KEY_HERE' or not FRED_API_KEY:
        print("Warning: FRED_API_KEY not set in config.py. Example usage skipped.")
    else:
        series_to_fetch = {
            'DGS10': '10y_treasury_yield',
            'T10Y2Y': '10y-2y_spread',
            'CPIAUCSL': 'cpi',
            'UNRATE': 'unemployment_rate',
            'VIXCLS': 'vix'
        }
        
        start = '2008-01-01'
        end = '2024-01-01'

        macro_data = get_fred_data(series_ids=series_to_fetch, start_date=start, end_date=end)

        if macro_data is not None:
            print("\n--- Fetched Macroeconomic Data ---")
            print(macro_data.head())
            print("\n...")
            print(macro_data.tail())
