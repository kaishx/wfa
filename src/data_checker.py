import pandas as pd
import numpy as np
import os
import glob

CACHE_DIR = "market_data_cache"
max_gaps = 50  # edit this number depending on the acceptable number of gaps

def filter():
    print(f"scanning {CACHE_DIR}")

    files = glob.glob(os.path.join(CACHE_DIR, "*.pkl"))
    deleted_count = 0
    kept_count = 0

    for file_path in files:
        try:
            df = pd.read_pickle(file_path)
            if isinstance(df, pd.Series):
                df = df.to_frame()

            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
            else:
                df.index = df.index.tz_convert('US/Eastern')

            rth_mask = (df.index.time >= pd.to_datetime("09:30").time()) & \
                       (df.index.time <= pd.to_datetime("16:00").time())
            df_rth = df[rth_mask]

            time_diffs = df_rth.index.to_series().diff().dropna()
            # i will count all gaps as longer than 15 minutes (the lowest interval) and shorter than 16 hours (filtering out overnight gaps)
            gap_mask = (time_diffs > pd.Timedelta(minutes=15)) & (time_diffs < pd.Timedelta(hours=16))
            gap_count = gap_mask.sum()

            if gap_count > max_gaps:
                print(f"removing {os.path.basename(file_path)} as it has {gap_count} gaps")
                os.remove(file_path)
                deleted_count += 1
            else:
                kept_count += 1

        except Exception as e:
            print(f"error reading {os.path.basename(file_path)}: {e}")

    print(f"cleanup complete, kept: {kept_count} files, deleted: {deleted_count} files")

if __name__ == "__main__":
    filter()