"""Three-year dataset (24 Aug 2023 - 23 Aug 2026) for the multi-year evaluation.

Older two years: CPD 'Crimes - 2001 to Present' (Socrata ijzp-q8t2), downloaded 25 Oct 2026.
Newest year: the one-year extract used in the main study (downloaded 31 Aug 2026).
The two files are harmonised to the same schema, concatenated, and cleaned with the same rules.
"""
import pandas as pd

import prep

OLD = prep.ROOT / "machinelearningcode/Coding/Dataset/Crimes_2001_to_present_subset_20230824_20250823.csv"
KEEP = ["CASE#", "DATE_OF_OCCURRENCE", "BLOCK", "IUCR", "PRIMARY_DESCRIPTION", "SECONDARY_DESCRIPTION",
        "LOCATION_DESCRIPTION", "ARREST", "DOMESTIC", "BEAT", "WARD", "LATITUDE", "LONGITUDE"]
N_YEARS = 3
N_PERIODS = 39          # 39 x 28 days = 1,092 of the 1,096 days
TRAIN = list(range(3, 27))      # 24 target periods (about 22 months)
VAL = list(range(27, 33))       # 6 target periods
TEST = list(range(33, 39))      # 6 target periods (about 6 months, to 22 Aug 2026)
ROLL = TEST


def read_old():
    d = pd.read_csv(OLD, dtype={"iucr": str, "beat": str, "ward": str, "fbi_code": str})
    d = d.rename(columns={"case_number": "CASE#", "date": "DATE_OF_OCCURRENCE", "block": "BLOCK", "iucr": "IUCR",
                          "primary_type": "PRIMARY_DESCRIPTION", "description": "SECONDARY_DESCRIPTION",
                          "location_description": "LOCATION_DESCRIPTION", "arrest": "ARREST",
                          "domestic": "DOMESTIC", "beat": "BEAT", "ward": "WARD",
                          "latitude": "LATITUDE", "longitude": "LONGITUDE"})
    d["DATE_OF_OCCURRENCE"] = pd.to_datetime(d.DATE_OF_OCCURRENCE)
    d["ARREST"] = d.ARREST.astype(str).str.lower().map({"true": "Y", "false": "N"})
    d["DOMESTIC"] = d.DOMESTIC.astype(str).str.lower().map({"true": "Y", "false": "N"})
    return d[KEEP]


def read_new():
    d = prep.read_raw()
    d["WARD"] = d.WARD.astype("string")
    d["BEAT"] = d.BEAT.astype("string")
    return d[KEEP]


def combined():
    old, new = read_old(), read_new()
    assert old.DATE_OF_OCCURRENCE.max() < new.DATE_OF_OCCURRENCE.min()
    return pd.concat([old, new], ignore_index=True), len(old), len(new)


def configure():
    """Set module constants used by prep for the three-year run."""
    prep.N_PERIODS = N_PERIODS
    prep.RARE_MIN = 1000 * N_YEARS


def loader():
    configure()
    df, n_old, n_new = combined()
    return prep.clean(df)


def derive_p1_classes(df_clean_panel, train=TRAIN, min_share=0.05):
    share, keep = prep.p1_class_rule(df_clean_panel, train, min_share)
    return share, keep


def splice_check():
    """Monthly counts around the join between the two files (diagnostic for the report)."""
    configure()
    df, log = prep.clean(combined()[0])
    per = df.groupby("PERIOD").agg(n=("VICTIM", "size"), victim_share=("VICTIM", "mean"),
                                   arrest=("ARREST_F", "mean"), domestic=("DOMESTIC_F", "mean"))
    return df, log, per


if __name__ == "__main__":
    df, log, per = splice_check()
    for k, v in log:
        print(f"{k:35s}{v:>9,}")
    print(per.round(3).to_string())
    print("date range", df.DATE.min(), df.DATE.max())
    print("last period incl. leftover days:", (df.PERIOD >= N_PERIODS).sum())
