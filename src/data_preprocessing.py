import pandas as pd
from datetime import date


def compute_concurrency(times: pd.Series) -> int:
    """
    Compute the number of overlapping time pairs within a specified window.
    Counts pairs of times where the difference is between -16 and 5 minutes.
    
    Args:
        times (pd.Series): Series of datetime objects.
    
    Returns:
        int: Number of overlapping time pairs.
    """
    
    times = times.dropna().sort_values()
    count = 0
    for i, t1 in enumerate(times):
        for j, t2 in enumerate(times):
            if i >= j:  # avoid double counting and self
                continue
            delta = (t2 - t1).total_seconds() / 60.0  # minutes difference
            if -16 <= delta <= 5:
                count += 1
    return count


def get_season(date: date) -> str:
    """
    Return the meteorological season for a given date.
    
    Args:
        date (date): A datetime.date object.
    
    Returns:
        str: Season name ('Winter', 'Spring', 'Summer', 'Fall').
    """

    month = date.month
    day = date.day
    if (month == 12 and day >= 21) or (month <= 3 and (month != 3 or day <= 19)):
        return 'Winter'
    elif (month == 3 and day >= 20) or (month in [4, 5]) or (month == 6 and day <= 20):
        return 'Spring'
    elif (month == 6 and day >= 21) or (month in [7, 8]) or (month == 9 and day <= 21):
        return 'Summer'
    else:
        return 'Fall'


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess flight data to compute scheduled and actual concurrency metrics,
    and annotate with season and other derived columns.
    
    Args:
        df (DataFrame): Raw flight data containing columns like
                        'dep_airport_group', 'arr_airport_group', 'std', 'atd', 'sta', 'ata', 'cancelled'.
    
    Returns:
        DataFrame: Aggregated flight data grouped by scheduled date, hour, and airport group,
                   including concurrency metrics and season.
    """
    # Drop unnecessary columns
    df_reduced = df.drop(columns=["service_type", "dep_airport", "arr_airport"], errors='ignore')

    # Remove cancelled flights
    df_cleaned = df_reduced[df_reduced["cancelled"] != 1].copy()
    df_cleaned = df_cleaned.drop(columns="cancelled", errors='ignore')

    # Departures dataframe
    df_departures = df_cleaned[["dep_airport_group", "std", "atd"]].copy()
    df_departures = df_departures.rename(columns={
        "dep_airport_group": "airport_group",
        "std": "datetime",
        "atd": "actual_datetime"
    })

    # Arrivals dataframe
    df_arrivals = df_cleaned[["arr_airport_group", "sta", "ata"]].copy()
    df_arrivals = df_arrivals.rename(columns={
        "arr_airport_group": "airport_group",
        "sta": "datetime",
        "ata": "actual_datetime"
    })

    # Combine departures and arrivals
    df_events = pd.concat([df_departures, df_arrivals], ignore_index=True)

    # Ensure datetime columns are proper datetimes
    df_events["datetime"] = pd.to_datetime(df_events["datetime"], errors="coerce")
    df_events["actual_datetime"] = pd.to_datetime(df_events["actual_datetime"], errors="coerce")

    # Extract scheduled and actual dates and hours
    df_events["sched_date"] = df_events["datetime"].dt.date
    df_events["sched_hour"] = df_events["datetime"].dt.hour
    df_events["actual_date"] = df_events["actual_datetime"].dt.date
    df_events["actual_hour"] = df_events["actual_datetime"].dt.hour

    # Group by scheduled date, hour, and airport group
    df_grouped = (
        df_events.groupby(["sched_date", "sched_hour", "airport_group"])
        .apply(lambda g: pd.Series({
            "sched_flights": len(g),
            "sched_concurrence": compute_concurrency(g["datetime"]),
            "actual_concurrence": compute_concurrency(g["actual_datetime"])
        }))
        .reset_index()
        .rename(columns = {"sched_date": "date", "sched_hour": "hour"})
        .sort_values(by = ["date", "hour", "airport_group"])
    )

    # Flag concurrency presence
    df_grouped["concurrency"] = (df_grouped["actual_concurrence"] > 0).astype(int)

    # Add season
    df_grouped["season"] = df_grouped["date"].apply(get_season)

    # Reorder columns to place season after hour
    cols = df_grouped.columns.tolist()
    cols.insert(cols.index("hour") + 1, cols.pop(cols.index("season")))
    df_grouped = df_grouped[cols]

    return df_grouped