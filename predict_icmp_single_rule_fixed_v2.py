import argparse
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict, deque

def parse_kibana_timestamp(series):
    """
    Handles ISO timestamps and Kibana CSV timestamps like:
    May 21, 2026 @ 03:51:51.304
    """
    s = series.astype(str).str.strip()
    s2 = s.str.replace(" @ ", " ", regex=False)

    dt = pd.to_datetime(s2, format="%b %d, %Y %H:%M:%S.%f", errors="coerce", utc=True)

    missing = dt.isna()
    if missing.any():
        dt2 = pd.to_datetime(s2[missing], errors="coerce", utc=True)
        dt.loc[missing] = dt2

    return dt

def datetime_to_seconds(dt_series):
    """
    Robust timestamp conversion.
    Do NOT use astype('int64') here because some Windows/Python/Pandas
    combinations can behave unexpectedly with timezone-aware datetimes.
    """
    return dt_series.map(lambda x: x.timestamp()).to_numpy(dtype=float)

def add_behavior_features(input_df, feature_cols):
    work = input_df.copy()

    ts_col = "@timestamp" if "@timestamp" in work.columns else "timestamp"
    if ts_col not in work.columns:
        raise ValueError("CSV must contain @timestamp or timestamp column")

    work[ts_col] = parse_kibana_timestamp(work[ts_col])
    before = len(work)
    work = work.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    if len(work) == 0:
        raise ValueError(
            "All rows were dropped because timestamps could not be parsed. "
            "Check the @timestamp format in your CSV."
        )

    if len(work) < before:
        print(f"Warning: dropped {before - len(work)} rows with invalid timestamps.")

    if "source.ip" not in work.columns:
        work["source.ip"] = "unknown"
    if "destination.ip" not in work.columns:
        work["destination.ip"] = "unknown"

    n = len(work)
    ts_sec = datetime_to_seconds(work[ts_col])
    src_arr = work["source.ip"].astype(str).to_numpy()
    dst_arr = work["destination.ip"].astype(str).to_numpy()

    feat = {c: np.zeros(n, dtype=float) for c in feature_cols}

    src_hist, pair_hist = defaultdict(deque), defaultdict(deque)
    src_last, pair_last = {}, {}

    for i in range(n):
        t, src, dst = ts_sec[i], src_arr[i], dst_arr[i]
        pair = (src, dst)

        if src in src_last:
            prev = src_last[src]
            delta = t - ts_sec[prev]
            feat["delta_prev_src_sec"][i] = delta
            feat["same_dst_as_prev_src"][i] = 1.0 if dst_arr[prev] == dst else 0.0

            recent_idx = list(src_hist[src])[-10:]
            if len(recent_idx) >= 2:
                deltas = np.diff(ts_sec[recent_idx])
                feat["avg_delta_src_10_events"][i] = float(np.mean(deltas))
                feat["min_delta_src_10_events"][i] = float(np.min(deltas))
            else:
                feat["avg_delta_src_10_events"][i] = delta
                feat["min_delta_src_10_events"][i] = delta
        else:
            feat["delta_prev_src_sec"][i] = 9999
            feat["avg_delta_src_10_events"][i] = 9999
            feat["min_delta_src_10_events"][i] = 9999

        if pair in pair_last:
            feat["delta_prev_pair_sec"][i] = t - ts_sec[pair_last[pair]]
        else:
            feat["delta_prev_pair_sec"][i] = 9999

        for dq in (src_hist[src], pair_hist[pair]):
            while dq and ts_sec[dq[0]] < t - 60:
                dq.popleft()

        for win in [5, 10, 30, 60]:
            src_recent = [j for j in src_hist[src] if ts_sec[j] >= t - win]
            pair_recent = [j for j in pair_hist[pair] if ts_sec[j] >= t - win]

            feat[f"count_src_{win}s"][i] = len(src_recent) + 1
            feat[f"count_pair_{win}s"][i] = len(pair_recent) + 1

            eps_col = f"events_per_sec_{win}s"
            if eps_col in feat:
                feat[eps_col][i] = (len(src_recent) + 1) / win

            if win in [10, 30, 60]:
                unique_dst = {dst_arr[j] for j in src_recent}
                unique_dst.add(dst)
                feat[f"unique_dst_src_{win}s"][i] = len(unique_dst)

        src_hist[src].append(i)
        pair_hist[pair].append(i)
        src_last[src] = i
        pair_last[pair] = i

    for c, arr in feat.items():
        work[c] = arr

    return work

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    package = joblib.load(args.model)
    model = package["model"]
    feature_cols = package["feature_cols"]

    df = pd.read_csv(args.csv)
    feat = add_behavior_features(df, feature_cols)

    X = feat[feature_cols]
    pred = model.predict(X)

    out = feat.copy()
    out["predicted_label"] = pred

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = list(model.classes_)
        out["confidence"] = proba.max(axis=1)
        for i, cls in enumerate(classes):
            out[f"prob_{cls}"] = proba[:, i]

    out.to_csv(args.output, index=False)

    print(f"Rows processed: {len(out)}")
    print(f"Saved predictions to {args.output}")
    print(out["predicted_label"].value_counts())

if __name__ == "__main__":
    main()
