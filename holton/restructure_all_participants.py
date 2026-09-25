"""
Restructure all_participants.csv into a simplified trial-level schema.

Input : <data_dir>/all_participants.csv (data_dir from settings.py)
Output: <data_dir>/all_participants_restructured.csv

Target columns:
    participant, block_id, trial_in_block,
    net_type, net_amount_before, net_size,
    offer_crab, offer_octopus, offer_fish,
    choice, choice_offer, switched,
    net_filled, points_scored

Column-mapping notes:
- magnitudes_1 = fish, magnitudes_2 = octopus, magnitudes_3 = crab
  (verified against preprocess_data.m which defines
   magnitudes = [fish_val, octopus_val, crab_val]).
- `net_type` is the item being collected in the net, reported as the
  item chosen on that trial (crab/octopus/fish). On the first trial of
  a block and on switch trials this is the newly-started net; on
  non-switch trials this equals the item already in the net.
- `net_amount_before` = net_contents (amount in net before the decision).
- `choice_offer` = value of the offer for the chosen item on that trial.
- `net_filled` = True only on the trial where the net crosses the goal
  threshold (net_amount_before < net_size <= net_amount_after, where
  net_amount_after = net_contents_end). The net is reset and a new block
  begins immediately after such a trial, so each block contains exactly
  one fill event (on its final trial).
- `points_scored` = cumulative count of nets filled per participant
  (running sum of net_filled).
"""

from pathlib import Path
import numpy as np
import pandas as pd

from settings import data_dir

ITEM_NAMES = {1: "fish", 2: "octopus", 3: "crab"}


def restructure(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # Identifiers / indexing
    out["participant"] = df["participant"]
    out["block_id"] = df["block"]
    out["trial_in_block"] = df["trial_number"]

    # Choice as item name
    choice_int = df["choice"].astype(int)
    out["choice"] = choice_int.map(ITEM_NAMES)

    # net_type: item being collected in the net on this trial.
    # Per instructions, use the chosen item (this is the net type at
    # end-of-trial; identical to the pre-existing net on non-switch trials
    # and identical to the new net on switch / first-of-block trials).
    out["net_type"] = out["choice"]

    # Net amounts and size
    out["net_amount_before"] = df["net_contents"]
    net_amount_after = df["net_contents_end"]
    out["net_size"] = df["goal_size"]

    # Offers per item
    out["offer_fish"] = df["magnitudes_1"]
    out["offer_octopus"] = df["magnitudes_2"]
    out["offer_crab"] = df["magnitudes_3"]

    # Offer of the chosen item
    magnitudes = df[["magnitudes_1", "magnitudes_2", "magnitudes_3"]].to_numpy()
    chosen_idx = choice_int.to_numpy() - 1  # 0-based index
    out["choice_offer"] = magnitudes[np.arange(len(df)), chosen_idx]

    # Switched (bool)
    out["switched"] = df["switch_trial"].astype(bool)

    # Net filled (bool): crossing event this trial.
    out["net_filled"] = (
        (out["net_amount_before"] < out["net_size"])
        & (net_amount_after >= out["net_size"])
    )

    # Points scored: cumulative count of filled nets per participant.
    out["points_scored"] = (
        out.groupby("participant")["net_filled"].cumsum().astype(int)
    )

    # Column order
    cols = [
        "participant",
        "block_id",
        "trial_in_block",
        "net_type",
        "net_amount_before",
        "net_size",
        "offer_crab",
        "offer_octopus",
        "offer_fish",
        "choice",
        "choice_offer",
        "switched",
        "net_filled",
        "points_scored",
    ]
    return out[cols]


def main() -> None:
    in_path = Path(data_dir) / "all_participants.csv"
    out_path = Path(data_dir) / "all_participants_restructured.csv"

    df = pd.read_csv(in_path)
    restructured = restructure(df)
    restructured.to_csv(out_path, index=False)
    print(f"Wrote {len(restructured)} rows to {out_path}")


if __name__ == "__main__":
    main()
