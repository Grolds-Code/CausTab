"""
CausTab — NHANES Data Download Script
Downloads Demographics and Blood Pressure data for 4 survey cycles.
Each cycle will serve as one "environment" in our causal invariant learning framework.
Uses only urllib — no external packages needed.
"""

import urllib.request
import os
import pandas as pd

# ── What is a survey cycle? ───────────────────────────────────────────────────
# NHANES collects data in 2-year blocks. Each block is one "environment" for us.
# 2011-12, 2013-14, 2015-16, 2017-18 = 4 environments = 4 points of shift.
# ─────────────────────────────────────────────────────────────────────────────

CYCLES = {
    "2011-12": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2011/DataFiles/",
    "2013-14": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles/",
    "2015-16": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2015/DataFiles/",
    "2017-18": "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/",
}

# ── What files do we need? ────────────────────────────────────────────────────
# DEMO = Demographics (age, sex, race, income, education)
# BPQ  = Blood Pressure Questionnaire (hypertension diagnosis — our outcome Y)
# BPX  = Blood Pressure Examination (actual measured BP readings — our features)
# BMX  = Body Measures (BMI, weight, height — known causal risk factors)
# ─────────────────────────────────────────────────────────────────────────────

FILES = {
    "2011-12": ["DEMO_G.XPT", "BPQ_G.XPT", "BPX_G.XPT", "BMX_G.XPT"],
    "2013-14": ["DEMO_H.XPT", "BPQ_H.XPT", "BPX_H.XPT", "BMX_H.XPT"],
    "2015-16": ["DEMO_I.XPT", "BPQ_I.XPT", "BPX_I.XPT", "BMX_I.XPT"],
    "2017-18": ["DEMO_J.XPT", "BPQ_J.XPT", "BPX_J.XPT", "BMX_J.XPT"],
}

# ── Where to save ─────────────────────────────────────────────────────────────
# Script lives in data/ so we save right here
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file(url, save_path):
    """
    Download one file from a URL and save it locally.
    Skips download if file already exists — safe to re-run.
    """
    if os.path.exists(save_path):
        print(f"  [SKIP] Already exists: {os.path.basename(save_path)}")
        return True

    print(f"  [DOWNLOAD] {os.path.basename(save_path)} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, save_path)
        print("done.")
        return True
    except Exception as e:
        print(f"FAILED — {e}")
        return False


def load_xpt(filepath):
    """
    XPT is SAS transport format — the format NHANES uses.
    Pandas reads it natively. No extra package needed.

    Plain English: XPT is just a spreadsheet in a slightly
    old-fashioned format. pd.read_sas opens it like any CSV.
    """
    return pd.read_sas(filepath, format='xport', encoding='utf-8')


def download_all():
    print("\n== CausTab: NHANES Download ==\n")

    all_cycles = {}

    for cycle, base_url in CYCLES.items():
        print(f"Cycle {cycle}:")
        cycle_frames = {}

        for filename in FILES[cycle]:
            url = base_url + filename
            save_path = os.path.join(SAVE_DIR, filename)

            success = download_file(url, save_path)

            if success:
                df = load_xpt(save_path)
                # Strip the letter suffix from filename to get table name
                # e.g. DEMO_G -> DEMO, BPQ_H -> BPQ
                table_name = filename.split("_")[0]
                cycle_frames[table_name] = df
                print(f"  [LOADED]  {table_name}: {df.shape[0]:,} rows, "
                      f"{df.shape[1]} columns")

        all_cycles[cycle] = cycle_frames
        print()

    return all_cycles


def merge_cycle(frames, cycle_label):
    """
    Merge the 4 tables for one cycle into a single flat dataframe.
    SEQN is the unique participant ID — the key that links all tables.

    Plain English: Each table has one row per person, identified by
    their ID number (SEQN). We join them like Excel VLOOKUP on that ID.
    """
    demo = frames["DEMO"][["SEQN", "RIDAGEYR", "RIAGENDR",
                            "RIDRETH3", "INDFMPIR", "DMDEDUC2"]]
    # RIDAGEYR = age in years
    # RIAGENDR = gender (1=Male, 2=Female)
    # RIDRETH3 = race/ethnicity
    # INDFMPIR = income to poverty ratio (proxy for socioeconomic status)
    # DMDEDUC2 = education level

    bpq = frames["BPQ"][["SEQN", "BPQ020"]]
    # BPQ020 = "Ever told you had high blood pressure?" 1=Yes, 2=No
    # This is our outcome variable Y

    bpx = frames["BPX"][["SEQN", "BPXSY1", "BPXDI1", "BPXSY2", "BPXDI2"]]
    # BPXSY1/2 = Systolic blood pressure readings 1 and 2
    # BPXDI1/2 = Diastolic blood pressure readings 1 and 2
    # These are measured features — known causal drivers of hypertension

    bmx = frames["BMX"][["SEQN", "BMXBMI", "BMXWAIST"]]
    # BMXBMI   = Body Mass Index — known causal risk factor
    # BMXWAIST = Waist circumference — known causal risk factor

    merged = (demo
              .merge(bpq,  on="SEQN", how="inner")
              .merge(bpx,  on="SEQN", how="inner")
              .merge(bmx,  on="SEQN", how="inner"))

    # Create clean binary outcome: 1 = has hypertension, 0 = does not
    # BPQ020: 1=Yes, 2=No, 7=Refused, 9=Don't know — keep only 1 and 2
    merged = merged[merged["BPQ020"].isin([1.0, 2.0])].copy()
    merged["hypertension"] = (merged["BPQ020"] == 1.0).astype(int)
    merged.drop(columns=["BPQ020"], inplace=True)

    # Tag each row with its environment (cycle)
    merged["environment"] = cycle_label

    return merged


def build_dataset(all_cycles):
    """
    Combine all 4 cycles into one master dataset.
    Each row knows which environment it came from.
    This environment label is the core of our invariant learning setup.
    """
    print("== Building master dataset ==\n")
    cycle_dfs = []

    for cycle, frames in all_cycles.items():
        df = merge_cycle(frames, cycle_label=cycle)
        cycle_dfs.append(df)
        print(f"  Cycle {cycle}: {len(df):,} participants after merging")

    master = pd.concat(cycle_dfs, ignore_index=True)

    # Drop rows with any missing values in our key columns
    before = len(master)
    master.dropna(inplace=True)
    after = len(master)
    print(f"\n  Dropped {before - after:,} rows with missing values")
    print(f"  Final dataset: {after:,} participants across 4 environments")
    print(f"  Hypertension prevalence: "
          f"{master['hypertension'].mean()*100:.1f}%")

    # Save to CSV — clean and ready for modelling
    out_path = os.path.join(SAVE_DIR, "nhanes_master.csv")
    master.to_csv(out_path, index=False)
    print(f"\n  Saved to: {out_path}")

    return master


if __name__ == "__main__":
    all_cycles = download_all()
    master_df = build_dataset(all_cycles)

    print("\n== Preview ==")
    print(master_df.head())
    print("\nColumns:", list(master_df.columns))
    print("Environments:", master_df["environment"].value_counts().to_dict())