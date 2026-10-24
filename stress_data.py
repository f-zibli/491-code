#!/usr/bin/env python3
"""
Build the combined patient dataset from the separate *_all.csv files  (VS Code / local version).

How to run in VS Code:
    1. pip install pandas numpy            (once, in the terminal VS Code is using)
    2. edit the three paths in the CONFIG block below
    3. press the "Run Python File" button, or in the terminal:  python build_combined_dataset_vscode.py
       (optional overrides:  --data-dir  --participants  --out  --bin-min 5)

Output: one row per patient per time bin, with these groups of columns
    ids / time : patient_id, timestamp (LOCAL clock, label = END of the bin), timestamp_utc
    glucose    : glucose (mean), glucose_last (latest reading in the bin), glucose_min, glucose_max, glucose_n
    wearable   : heart_rate, activity, calories, asleep, spo2, respiratory_rate, stress
    flags      : worn, and <signal>_available for each wearable signal

How records that are not recorded at exact timestamps are combined:
  * every record is assigned to a time bin (15 minutes by default);
  * readings (glucose, heart rate, SpO2, respiratory rate, stress) -> MEAN of the readings in the bin;
  * totals (steps, calories) -> SUM; a record that has an end time and spans several bins is split between
    them in proportion to the time it spends in each (so totals are conserved);
  * sleep stages (start/end intervals) -> `asleep` = fraction of the bin spent asleep (0 to 1);
  * glucose defines the rows; every other signal is joined onto those bins and left EMPTY (NaN) if missing.
The end time is used while building but is not a column in the result.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                  # e.g. running as a notebook cell
    HERE = Path.cwd()

# ======================================================================
# CONFIG  - edit these
# ======================================================================
DATA_DIR = HERE / "AI-READI - Split Datasets"      # folder with glucose_all.csv, heart_rate_all.csv, ...
                                                   # Windows example: Path(r"C:\Users\you\Data\AI-READI - Split Datasets")
PARTICIPANTS_TSV = HERE / "participants.tsv"       # used for local time and the optional T2D filter ("" to skip)
OUT_PATH = DATA_DIR / "combined_dataset_v5.csv"

BIN_MIN = 15                                       # bin length in minutes (5 also works: your CGM is 5-minute data)
LABEL_BINS_BY = "end"                              # "end": row T summarises (T-bin, T] - nothing from the future inside it
                                                   # "start": row T summarises [T, T+bin)
CHUNK_ROWS = 1_000_000                             # rows read at a time (lower this if you run out of memory)
REQUIRED = ["glucose"]                             # patients must have these; everything else may be missing
T2D_ONLY = False                                   # True: keep only type 2 diabetes participants (needs participants.tsv)
ZERO_FILL_WHEN_WORN = False                        # True: empty steps/calories/asleep become 0 where the watch was worn.
                                                   # Only sensible if the activity file has NO zero-valued rows (see the
                                                   # "records with value 0" line printed for activity).

# How to aggregate point readings within each time bin.
# Options: "mean", "median"
POINT_AGGREGATION = "mean"

# Flag unusually high readings for quality review; do not delete them
# solely because they are unusual.
FLAG_OUTLIERS = True

# Minimum readings required to flag a bin using a robust rule
MIN_OUTLIER_N = 3


T2D_GROUPS = ["oral_medication_and_or_non_insulin_injectable_medication_controlled", "insulin_dependent"]
# VERIFY these against the clinical_site values printed by this script.
SITE_TZ = {"UW": "America/Los_Angeles", "UCSD": "America/Los_Angeles", "UAB": "America/Chicago"}

# Sleep labels that mean NOT asleep. Every other known label (light, deep, rem, asleep ...) counts as asleep.
# Canonical sleep-stage categories
SLEEP_STAGES = ["awake", "light", "rem", "deep", "unknown"]

UNKNOWN_LABELS = {
    "", "nan", "none", "unknown", "unmeasurable",
    "unspecified", "null"
}

AWAKE_LABELS = {"awake", "wake", "wakefulness", "awake_in_bed"}

SLEEP_STAGE_ALIASES = {
    "awake": "awake",
    "wake": "awake",
    "wakefulness": "awake",
    "awake_in_bed": "awake",
    "in_bed": "awake",
    "out_of_bed": "awake",
    "out of bed": "awake",
    "light": "light",
    "light_sleep": "light",
    "rem": "rem",
    "rem_sleep": "rem",
    "deep": "deep",
    "deep_sleep": "deep",
}


# kind: "mean" (readings) | "sum" (totals; `end` splits a record across the bins it covers) | "asleep" (intervals)
# valid=(low, high): values outside are dropped (0 bpm heart rate, stress codes like -1, impossible SpO2 ...)
FILES = {
    "glucose": dict(file="glucose_all.csv", patient="person_id", time="timestamp", value="glucose_mg_dL",
                    kind="mean", detail=True, where=("event_type", "EGV"), valid=(0, np.inf)),
    "heart_rate": dict(file="heart_rate_all.csv", patient="person_id", time="timestamp", value="heart_rate_bpm",
                       kind="mean", valid=(0, np.inf)),
    "activity": dict(file="physical_activity_all.csv", patient="person_id", time="start_datetime",
                     end="end_datetime", value="value (steps)", kind="sum", valid=(0, np.inf)),
    "calories": dict(file="calorie_activity_all.csv", patient="person_id", time="timestamp",
                     end=None, value="calories_value (kcal)", kind="sum", valid=(0, np.inf)),   # set end= if the file has one
    "asleep": dict(file="sleep_state_all.csv", patient="person_id", time="start_datetime",
                   end="end_datetime", value="sleep_stage_state", kind="asleep"),
    "spo2": dict(file="oxygen_saturation_all.csv", patient="person_id", time="timestamp",
                 value="oxygen_saturation_%", kind="mean", valid=(0, np.inf)),
    "respiratory_rate": dict(file="respiratory_rate_all.csv", patient="person_id", time="datetime",
                             value="respiratory_rate (breaths/min)", kind="mean", valid=(0, np.inf)),
    "stress": dict(file="stress_data_all.csv", patient="person_id", time="datetime",
                   value="stress_levels", kind="mean", valid=(0, np.inf)),
}
WEARABLE = [n for n in FILES if n != "glucose"]

EPOCH = pd.Timestamp("1970-01-01")
BINSEC = BIN_MIN * 60


# ======================================================================
# time helpers
# ======================================================================
def secs(t):
    """tz-naive UTC timestamps -> epoch seconds."""
    return ((t - EPOCH) // pd.Timedelta(seconds=1)).astype("int64")


def point_bin_start(t):
    """Start (epoch seconds) of the bin a point reading belongs to.
    'end' labelling: bins are (T-bin, T], so a reading at 10:03 -> bin 10:00-10:15 (stamped 10:15); at 10:15 -> same bin."""
    edge = 1 if LABEL_BINS_BY == "end" else 0
    return ((secs(t) - edge) // BINSEC) * BINSEC


# ======================================================================
# reading the big files in chunks
# ======================================================================
def read_chunks(cfg):
    """Yield cleaned chunks: patient_id, start (UTC, tz-naive), end (or NaT), value."""
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise SystemExit(f"Missing file: {path}\nCSV files found in {DATA_DIR}: "
                         f"{sorted(p.name for p in DATA_DIR.glob('*.csv'))}")
    header = pd.read_csv(path, nrows=0).columns
    for key in ("patient", "time", "value"):
        if cfg[key] not in header:
            raise SystemExit(f"{cfg['file']}: column {cfg[key]!r} not found. Columns are {list(header)}")
    end_col = cfg.get("end")
    if end_col and end_col not in header:
        print(f"  note: {end_col!r} not found in {cfg['file']} (columns: {list(header)}); "
              "records will use their start time only")
        end_col = None
    use = [cfg["patient"], cfg["time"], cfg["value"]] + ([end_col] if end_col else [])
    where = cfg.get("where")
    if where:
        if where[0] not in header:
            raise SystemExit(f"{cfg['file']}: filter column {where[0]!r} not found. Columns are {list(header)}")
        use.append(where[0])

    for ch in pd.read_csv(path, usecols=use, chunksize=CHUNK_ROWS, low_memory=False):
        if where:
            ch = ch[ch[where[0]] == where[1]]
        out = pd.DataFrame({
            "patient_id": ch[cfg["patient"]].astype(str),
            "start": pd.to_datetime(ch[cfg["time"]], errors="coerce", utc=True).dt.tz_localize(None),
        })
        out["end"] = (pd.to_datetime(ch[end_col], errors="coerce", utc=True).dt.tz_localize(None)
                      if end_col else pd.NaT)
        if cfg["kind"] == "asleep":
            raw = ch[cfg["value"]].fillna("").astype(str).str.strip().str.lower()
            out["value"] = raw.map(SLEEP_STAGE_ALIASES).fillna("unknown")
        else:
            out["value"] = pd.to_numeric(
                ch[cfg["value"]], errors="coerce"
            )
            lo, hi = cfg.get("valid", (0, np.inf))
            # Keep values >= 0; no upper limit by default.
            out = out[out["value"].between(lo, hi, inclusive="both")]
        yield out.dropna(subset=["start", "value"])


def spread(ch):
    """Split interval records [start, end) across the bins they cover.
    Returns (row positions, bin start seconds, overlap seconds, record duration seconds)."""
    s, e = secs(ch["start"]).to_numpy(), secs(ch["end"]).to_numpy()
    keep = (e > s) & (e - s <= 24 * 3600)                    # drop reversed / zero-length / broken multi-day records
    idx = np.flatnonzero(keep)
    s, e = s[keep], e[keep]
    if len(s) == 0:
        z = np.array([], dtype=np.int64)
        return idx, z, z, z
    b0 = (s // BINSEC) * BINSEC
    n = (e - b0 + BINSEC - 1) // BINSEC                      # bins touched by each record
    rep = np.repeat(np.arange(len(s)), n)
    offs = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    bs = b0[rep] + offs * BINSEC
    ov = np.minimum(e[rep], bs + BINSEC) - np.maximum(s[rep], bs)
    return idx[rep], bs, ov, (e - s)[rep]


def point_stats(ch, detail):
    """Per (patient, bin): sum and count of point readings (+ min, max, latest reading for glucose)."""
    t = ch.assign(bin=point_bin_start(ch["start"]), ts=secs(ch["start"])).sort_values("ts")
    gr = t.groupby(["patient_id", "bin"])
    if detail:
        return gr.agg(sum=("value", "sum"), count=("value", "count"), min=("value", "min"),
                      max=("value", "max"), last_v=("value", "last"), last_t=("ts", "last"))
    return gr.agg(sum=("value", "sum"), count=("value", "count"))


def count_sleep_overlaps(ch):
    """
    Count sleep intervals that overlap a previous interval
    for the same patient.

    Intervals are treated as [start, end).
    """
    valid = ch.dropna(subset=["start", "end"]).copy()
    valid = valid[valid["end"] > valid["start"]]

    if valid.empty:
        return 0

    valid = valid.sort_values(
        ["patient_id", "start", "end"]
    )

    previous_max_end = (
        valid.groupby("patient_id")["end"]
        .transform(lambda s: s.cummax().shift())
    )

    overlaps = valid["start"] < previous_max_end
    return int(overlaps.sum())

# ======================================================================
# aggregate one signal
# ======================================================================
def aggregate(name, cfg):
    """Return (DataFrame indexed by (patient_id, bin start), set of patients, diagnostics)."""
    kind, detail = cfg["kind"], cfg.get("detail", False)
    parts, patients = [], set()
    d = dict(
    records=0,
    intervals=0,
    longer=0,
    cross=0,
    dur=[],
    zero=0,
    labels={},
    overlaps=0,
    )
    for ch in read_chunks(cfg):
        if ch.empty:
            continue
        if kind == "asleep":
            d["overlaps"] += count_sleep_overlaps(ch)
        patients.update(ch["patient_id"].unique())
        d["records"] += len(ch)
        if kind == "asleep":
            for k, v in ch["value"].value_counts().items():
                d["labels"][k] = d["labels"].get(k, 0) + int(v)
        else:
            d["zero"] += int((ch["value"] == 0).sum())
        is_int = (ch["end"].notna() & (ch["end"] > ch["start"])).to_numpy()
        iv, pts = ch[is_int], ch[~is_int]
        if len(iv):                                          # record length diagnostics
            dur = (iv["end"] - iv["start"]).dt.total_seconds()
            d["intervals"] += len(iv)
            d["longer"] += int((dur > BINSEC).sum())
            d["cross"] += int(((secs(iv["start"]) // BINSEC) != ((secs(iv["end"]) - 1) // BINSEC)).sum())
            d["dur"].append(dur.to_numpy()[:200_000] / 60)

        if kind == "mean":
            parts.append(point_stats(ch, detail))            # (a mean-type signal with an end time uses its start)
        elif kind == "sum":
            if len(pts):
                p = point_stats(pts, False)
                parts.append(p)
            if len(iv):
                row, bs, ov, dur = spread(iv)
                if len(row):
                    w = iv["value"].to_numpy()[row] * ov / dur       # share of the record's total inside this bin
                    parts.append(pd.DataFrame({"patient_id": iv["patient_id"].to_numpy()[row], "bin": bs,
                                               "sum": w, "count": 1}).groupby(["patient_id", "bin"]).sum())
        else:  # sleep-stage intervals
            if len(iv):
                row, bs, ov, dur = spread(iv)

                if len(row):
                    stages = iv["value"].to_numpy()[row]

                    sleep_parts = pd.DataFrame({
                        "patient_id": iv["patient_id"].to_numpy()[row],
                        "bin": bs,
                    })

                    # Record the number of seconds spent in each stage.
                    for stage in SLEEP_STAGES:
                        sleep_parts[f"{stage}_seconds"] = (
                            ov * (stages == stage)
                        )

                    parts.append(
                        sleep_parts.groupby(
                            ["patient_id", "bin"]
                        ).sum()
                    )
    if not parts:
        return pd.DataFrame(columns=[name]), patients, d

    tot = pd.concat(parts)

    if kind == "asleep":
        # Combine overlapping records by patient and time bin.
        stage_cols = [f"{stage}_seconds" for stage in SLEEP_STAGES]
        stage_seconds = tot.groupby(level=[0, 1])[stage_cols].sum()

        out = pd.DataFrame(index=stage_seconds.index)

        # Fraction of each bin assigned to each sleep stage.
        for stage in SLEEP_STAGES:
            out[f"sleep_{stage}_fraction"] = (
                stage_seconds[f"{stage}_seconds"] / BINSEC
            ).clip(upper=1.0)

        # Preserve the original asleep feature for compatibility.
        out["asleep"] = (
            out["sleep_light_fraction"]
            + out["sleep_rem_fraction"]
            + out["sleep_deep_fraction"]
        ).clip(upper=1.0)

    else:
        grp = tot.groupby(level=[0, 1])
        base = grp[["sum", "count"]].sum()

        if kind == "mean":
            out = pd.DataFrame({
                name: base["sum"] / base["count"]
            })

            if detail:
                out[f"{name}_last"] = (
                    tot.sort_values("last_t")
                    .groupby(level=[0, 1])["last_v"].last()
                )
                out[f"{name}_min"] = grp["min"].min()
                out[f"{name}_max"] = grp["max"].max()
                out[f"{name}_n"] = base["count"].astype("int32")

        else:  # sum
            out = pd.DataFrame({name: base["sum"]})

    return out, patients, d


def report(name, cfg, d, out):
    n = max(d["records"], 1)
    line = f"  {d['records']:,} records after cleaning"
    if d["intervals"]:
        med = float(np.median(np.concatenate(d["dur"]))) if d["dur"] else float("nan")
        line += (f"; {d['intervals']:,} have an end time (median length {med:.1f} min, "
                 f"{100 * d['longer'] / d['intervals']:.1f}% longer than a bin, "
                 f"{100 * d['cross'] / d['intervals']:.1f}% cross a bin boundary)")
    print(line)
    if cfg["kind"] == "sum":
        print(f"  records with value 0: {100 * d['zero'] / n:.1f}%   "
              "(if above 0%, an empty bin means 'not recorded', so leave it NaN)")
    if cfg["kind"] == "asleep" and d["labels"]:
        shown = {k: v for k, v in sorted(d["labels"].items(), key=lambda kv: -kv[1])[:10]}
        print(f"  sleep labels found: {shown}")
        print(f"  counted as AWAKE: {sorted(k for k in d['labels'] if k in AWAKE_LABELS) or 'none'}; "
              f"counted as ASLEEP: {sorted(k for k in d['labels'] if k not in AWAKE_LABELS)}")
        if not any(k in AWAKE_LABELS for k in d["labels"]):
            print("  note: no 'awake' label present, so everything inside a sleep record counts as asleep")
    if cfg["kind"] == "asleep":
        print(
            f"  sleep intervals overlapping a previous interval: "
            f"{d['overlaps']:,}"
        )
        if d["overlaps"] > 0:
            print(
                "  WARNING: inspect overlapping sleep intervals before "
                "using stage fractions; overlapping records can double-count time."
            )


# ======================================================================
# main
# ======================================================================
def main():
    global DATA_DIR, PARTICIPANTS_TSV, OUT_PATH, BIN_MIN, BINSEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--participants", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bin-min", type=int, default=None)
    a, _ = ap.parse_known_args()                    # tolerant of extra arguments added by notebook runners
    if a.data_dir:
        DATA_DIR = Path(a.data_dir)
        OUT_PATH = DATA_DIR / "combined_dataset_v4.csv"
    if a.participants is not None:
        PARTICIPANTS_TSV = Path(a.participants) if a.participants else ""
    if a.out:
        OUT_PATH = Path(a.out)
    if a.bin_min:
        BIN_MIN = a.bin_min
    BINSEC = BIN_MIN * 60

    if not DATA_DIR.is_dir():
        sys.exit(f"Data folder not found: {DATA_DIR}\nEdit DATA_DIR at the top of the script.")
    part = None
    if PARTICIPANTS_TSV and Path(PARTICIPANTS_TSV).exists():
        part = pd.read_csv(PARTICIPANTS_TSV, sep="\t", dtype={"person_id": str}).drop_duplicates("person_id")
    if T2D_ONLY and (part is None or "study_group" not in part):
        sys.exit("T2D_ONLY needs participants.tsv with a study_group column.")

    # ---- 1. aggregate every signal ----
    series, patients = {}, {}
    for name, cfg in FILES.items():
        print(f"Processing {name} ...")
        series[name], patients[name], diag = aggregate(name, cfg)
        print(f"  {len(patients[name])} patients, {len(series[name]):,} bins")
        report(name, cfg, diag, series[name])

    # ---- 2. which patients to keep ----
    keep = set.intersection(*[patients[n] for n in REQUIRED])
    if T2D_ONLY:
        t2d = set(part.loc[part["study_group"].isin(T2D_GROUPS), "person_id"])
        print(f"\nT2D filter: {len(keep & t2d)} of {len(keep)} patients with {REQUIRED} are type 2 diabetes participants")
        keep &= t2d
    print(f"\nPatients retained: {len(keep)}")
    for n in FILES:
        print(f"  {n}: {len(patients[n] & keep)} retained patients have it")

    # ---- 3. merge on the glucose timeline ----
    merged = series["glucose"]
    for n in WEARABLE:
        merged = merged.join(series[n], how="left")
    merged = merged.reset_index()
    merged = merged[merged["patient_id"].isin(keep)].copy()

    for n in WEARABLE:                                  # the signals that exist for a bin, BEFORE any filling
        merged[f"{n}_available"] = merged[n].notna().astype("int8")
    sensors = [n for n in ("heart_rate", "spo2", "respiratory_rate", "stress", "activity", "calories") if n in merged]
    merged["worn"] = merged[sensors].notna().any(axis=1).astype("int8")      # any wearable data in this bin
    if ZERO_FILL_WHEN_WORN:
        worn = merged["worn"] == 1
        for n in ("activity", "calories"):
            merged.loc[worn & merged[n].isna(), n] = 0.0
        merged.loc[worn & merged["patient_id"].isin(patients["asleep"]) & merged["asleep"].isna(), "asleep"] = 0.0

    label_shift = pd.Timedelta(minutes=BIN_MIN) if LABEL_BINS_BY == "end" else pd.Timedelta(0)
    merged["timestamp_utc"] = EPOCH + pd.to_timedelta(merged.pop("bin"), unit="s") + label_shift
    merged["timestamp"] = merged["timestamp_utc"].copy()

    # ---- 4. UTC -> local clock time per study site ----
    if part is not None and "clinical_site" in part:
        merged["site"] = merged["patient_id"].map(part.set_index("person_id")["clinical_site"])
        print("\nPatients by clinical_site:",
              merged.drop_duplicates("patient_id")["site"].value_counts(dropna=False).to_dict())
        for s_name, tz in SITE_TZ.items():
            m = merged["site"] == s_name
            if m.any():
                merged.loc[m, "timestamp"] = (merged.loc[m, "timestamp_utc"].dt.tz_localize("UTC")
                                              .dt.tz_convert(tz).dt.tz_localize(None))
        unmapped = merged.loc[~merged["site"].isin(SITE_TZ), "patient_id"].nunique()
        if unmapped:
            print(f"WARNING: {unmapped} patients have a site that is not in SITE_TZ - their timestamps stay in UTC. "
                  "Fix SITE_TZ to match the site names printed above.")
        merged = merged.drop(columns="site")
    else:
        print("\nWARNING: participants.tsv (with clinical_site) not found - timestamps stay in UTC, so any "
              "time-of-day feature will be in UTC rather than local time.")

    # ---- 5. sort, validate, save ----
    glucose_cols = ["glucose"] + [c for c in merged.columns if c.startswith("glucose_")]
    sleep_cols = [
    f"sleep_{stage}_fraction"
    for stage in SLEEP_STAGES
    ]

    cols = (
        ["patient_id", "timestamp"]
        + glucose_cols
        + [n for n in WEARABLE if n != "asleep"]
        + sleep_cols
        + ["asleep", "worn"]
        + [f"{n}_available" for n in WEARABLE]
        + ["timestamp_utc"]
    )

    merged = merged[cols].sort_values(["patient_id", "timestamp_utc"]).reset_index(drop=True)
    dup = int(merged.duplicated(["patient_id", "timestamp_utc"]).sum())
    if dup:
        raise SystemExit(f"Found {dup} duplicate patient/bin rows - something is wrong with the binning.")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {OUT_PATH}: {len(merged):,} rows, {merged['patient_id'].nunique()} patients, "
          f"{BIN_MIN}-minute bins labelled by their {LABEL_BINS_BY}")
    print(merged.head().to_string())
    print("\nShare of rows that are EMPTY, per column:")
    print(merged.drop(columns=[c for c in merged if c.endswith("_available")]).isna().mean().round(3).to_string())
    print("\nTypical values (check they look physical):")
    num = ["glucose", "glucose_last", "heart_rate", "activity", "calories", "asleep", "spo2", "respiratory_rate", "stress"]
    print(merged[num].describe().loc[["mean", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()