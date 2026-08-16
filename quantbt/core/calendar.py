"""A-share trading calendar utilities."""

import pandas as pd


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    freq: str = "monthly",
) -> list[pd.Timestamp]:
    """Generate rebalance schedule from available trading dates.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Available trading days.
    freq : str
        'daily', 'weekly', 'monthly', or 'quarterly'.

    Returns
    -------
    list[pd.Timestamp]
        Rebalance dates.
    """
    if freq == "daily":
        return [pd.Timestamp(d) for d in dates]

    df = pd.DataFrame({"date": pd.DatetimeIndex(dates)})
    if freq == "weekly":
        # ISO year + ISO week: calendar-year week keys would split the
        # cross-year ISO week (e.g. 2019-12-30..2020-01-05) into two groups
        # and trigger two rebalances inside one ISO week.
        iso = df["date"].dt.isocalendar()
        df["key"] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
        rebal = df.groupby("key")["date"].first()
    elif freq == "monthly":
        rebal = df.groupby(df["date"].dt.to_period("M"))["date"].first()
    elif freq == "quarterly":
        rebal = df.groupby(df["date"].dt.to_period("Q"))["date"].first()
    else:
        raise ValueError(f"Unknown rebalance frequency: {freq}")

    return [pd.Timestamp(d) for d in rebal]