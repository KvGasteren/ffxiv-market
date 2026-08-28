"""
Universalis market data collector for FFXIV.

Builds three things:
  1. Static reference data (items, recipes) from the ffxiv-datamining CSVs  -> ref/
  2. A snapshot of current listings (the ask side)                          -> listings/date=YYYY-MM-DD/
  3. Recent sale history (the transaction side)                             -> sales/date=YYYY-MM-DD/

Run the reference build once. Run the snapshot on a schedule (e.g. every 4h)
so that you accumulate a panel over time -- that is what makes the
"buy now, sell later" analysis possible.

Usage:
    python universalis_collect.py ref
    python universalis_collect.py snapshot
    python universalis_collect.py snapshot --world Light --limit 500

Deps: requests, pandas, pyarrow
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

WORLD = "Light"          # world name, DC name ("Light", "Chaos"), or region ("Europe")
DATA_DIR = Path("data")
BATCH_SIZE = 100         # max item ids per Universalis request
SLEEP = 0.15             # seconds between requests; be a good citizen
LISTINGS_PER_ITEM = 10   # how deep into the order book to record
HISTORY_ENTRIES = 200    # max sale records per item per run
HISTORY_HOURS = 6        # only pull sales from the last N hours (see note below)
COMPRESSION = "zstd"

# HISTORY_HOURS deliberately overlaps the 4h schedule. Sales are immutable, so
# an overlap costs you a few duplicate rows (dedupe on item_id/sold_at/world/
# price/quantity at read time) whereas a gap loses sales permanently.

API = "https://universalis.app/api/v2"
DATAMINING = "https://raw.githubusercontent.com/xivapi/ffxiv-datamining/master/csv/en"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "personal-market-analysis/0.1"})


def get(url: str, **params):
    """GET with simple retry/backoff. Universalis is free; don't hammer it."""
    for attempt in range(10):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 504:  # Gateway timeout - more aggressive backoff
                if attempt < 9:
                    wait_time = min(2 ** (attempt + 1), 120)  # Cap at 2 minutes
                    print(f"  retry {attempt + 1} after 504 (waiting {wait_time}s)", file=sys.stderr)
                    time.sleep(wait_time)
                    continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == 9:
                raise
            print(f"  retry {attempt + 1} after {e}", file=sys.stderr)
            time.sleep(2 ** attempt)

def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# --------------------------------------------------------------------------
# 1. Reference data
# --------------------------------------------------------------------------

def read_datamining_csv(name: str) -> pd.DataFrame:
    """
    Single header row; the key column is literally named '#'.
    (These are XIVData Oxidizer exports -- note that older tutorials describe
    a 3-row SaintCoinach header with Level{Item}-style names. That's gone.)
    """
    raw = SESSION.get(f"{DATAMINING}/{name}.csv", timeout=120)
    raw.raise_for_status()
    df = pd.read_csv(io.StringIO(raw.content.decode("utf-8")), low_memory=False)
    return df.rename(columns={"#": "key"})


def build_reference():
    out = DATA_DIR / "ref"
    out.mkdir(parents=True, exist_ok=True)

    print("fetching marketable item ids...")
    marketable = get(f"{API}/marketable")
    pd.DataFrame({"item_id": marketable}).to_parquet(out / "marketable.parquet")
    print(f"  {len(marketable)} marketable items")

    print("fetching Item.csv...")
    items = read_datamining_csv("Item")
    keep = {
        "key": "item_id",
        "Name": "name",
        "LevelItem": "ilvl",
        "LevelEquip": "equip_level",
        "StackSize": "stack_size",
        "CanBeHq": "can_be_hq",
        "IsUntradable": "untradable",
        "PriceLow": "vendor_sell",     # what a vendor pays you
        "PriceMid": "vendor_buy",      # what a vendor charges you
        "Rarity": "rarity",
        "ItemSearchCategory": "search_category",
        "ItemUICategory": "ui_category",
        "ItemSortCategory": "sort_category",
    }
    items = items[list(keep)].rename(columns=keep)
    items = items[items["name"].notna() & (items["name"] != "")]
    items.to_parquet(out / "items.parquet", index=False)
    print(f"  {len(items)} named items")

    print("fetching Recipe.csv...")
    recipe = read_datamining_csv("Recipe")
    cols = ["key", "CraftType", "RecipeLevelTable", "ItemResult", "AmountResult"]
    names = ["recipe_id", "craft_type", "rlvl", "result_item_id", "result_amount",
             "ingredient_item_id", "ingredient_amount"]
    parts = []
    for i in range(8):   # a recipe has at most 8 ingredient slots
        p = recipe[cols + [f"Ingredient[{i}]", f"AmountIngredient[{i}]"]].copy()
        p.columns = names
        parts.append(p)
    ri = pd.concat(parts, ignore_index=True)
    ri = ri[(ri.result_item_id > 0)
            & (ri.ingredient_item_id > 0)
            & (ri.ingredient_amount > 0)]
    ri = ri.sort_values(["recipe_id", "ingredient_item_id"]).reset_index(drop=True)
    ri.to_parquet(out / "recipe_ingredients.parquet", index=False)
    print(f"  {len(ri)} rows covering {ri.result_item_id.nunique()} craftable items")

    print("fetching CraftType.csv...")
    ct = read_datamining_csv("CraftType")[["key", "Name"]]
    ct.columns = ["craft_type", "job"]
    ct.to_parquet(out / "craft_types.parquet", index=False)


# --------------------------------------------------------------------------
# 2 + 3. Market snapshot
# --------------------------------------------------------------------------

def flatten_listings(payload: dict, captured: str) -> tuple[list[dict], list[dict]]:
    """
    Returns (summary_rows, depth_rows).

    The summary is one row per item and is what the time series is built on.
    The depth table repeats nothing from the summary, so storing it is optional
    -- see the --depth flag.
    """
    summary, depth = [], []
    for item_id, item in (payload.get("items") or {}).items():
        iid = int(item_id)
        summary.append({
            "item_id": iid,
            "captured_at": captured,
            "last_upload": item.get("lastUploadTime"),
            "n_listings": item.get("listingsCount"),
            "n_sales_recent": item.get("recentHistoryCount"),
            "units_for_sale": item.get("unitsForSale"),
            "sale_velocity": item.get("regularSaleVelocity"),
            "sale_velocity_nq": item.get("nqSaleVelocity"),
            "sale_velocity_hq": item.get("hqSaleVelocity"),
            "avg_price": item.get("averagePrice"),
            "avg_price_nq": item.get("averagePriceNQ"),
            "avg_price_hq": item.get("averagePriceHQ"),
            "min_price": item.get("minPrice"),
            "min_price_nq": item.get("minPriceNQ"),
            "min_price_hq": item.get("minPriceHQ"),
            "world": item.get("worldName"),
        })
        for rank, l in enumerate(item.get("listings") or []):
            depth.append({
                "item_id": iid,
                "captured_at": captured,
                "rank": rank,
                "price": l.get("pricePerUnit"),
                "quantity": l.get("quantity"),
                "hq": l.get("hq"),
                "world": l.get("worldName") or item.get("worldName"),
                "total": l.get("total"),
                "tax": l.get("tax"),
                "listed_at": l.get("lastReviewTime"),
            })
    return summary, depth


def flatten_sales(payload: dict, captured: str) -> list[dict]:
    rows = []
    for item_id, item in (payload.get("items") or {}).items():
        for e in item.get("entries") or []:
            rows.append({
                "item_id": int(item_id),
                "captured_at": captured,
                "sold_at": e.get("timestamp"),
                "price": e.get("pricePerUnit"),
                "quantity": e.get("quantity"),
                "hq": e.get("hq"),
                "world": e.get("worldName") or item.get("worldName"),
                "on_mannequin": e.get("onMannequin"),
            })
    return rows


def write(rows: list[dict], name: str, world: str, day: str, stamp: str,
          sort_by: list[str]):
    if not rows:
        print(f"  no {name} rows, skipping")
        return
    df = pd.DataFrame(rows).sort_values(sort_by)   # sorting shrinks parquet a lot
    out = DATA_DIR / name / f"world={world}" / f"date={day}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{stamp}.parquet"
    df.to_parquet(path, index=False, compression=COMPRESSION)
    kb = path.stat().st_size / 1024
    print(f"  {name:<8} {len(df):>7} rows  {kb:7.0f} KB  -> {path}")


def snapshot(world: str, limit: int | None, depth: int):
    ref = DATA_DIR / "ref" / "marketable.parquet"
    if not ref.exists():
        sys.exit("run `python universalis_collect.py ref` first")
    ids = pd.read_parquet(ref)["item_id"].tolist()
    if limit:
        ids = ids[:limit]

    now = datetime.now(timezone.utc)
    captured = now.isoformat()
    day = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")

    summary_rows, depth_rows, sale_rows = [], [], []
    batches = list(chunks(ids, BATCH_SIZE))
    for i, batch in enumerate(batches, 1):
        joined = ",".join(str(x) for x in batch)

        cur = get(f"{API}/{world}/{joined}", listings=depth or 1, entries=0)
        s, d = flatten_listings(cur, captured)
        summary_rows += s
        if depth:
            depth_rows += d
        time.sleep(SLEEP)

        hist = get(f"{API}/history/{world}/{joined}",
                   entriesToReturn=HISTORY_ENTRIES,
                   entriesWithin=HISTORY_HOURS * 3600)
        sale_rows += flatten_sales(hist, captured)
        time.sleep(SLEEP)

        if i % 20 == 0 or i == len(batches):
            print(f"  batch {i}/{len(batches)}", flush=True)

    write(summary_rows, "summary", world, day, stamp, ["item_id"])
    write(depth_rows, "depth", world, day, stamp, ["item_id", "rank"])
    write(sale_rows, "sales", world, day, stamp, ["item_id", "sold_at"])


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["ref", "snapshot"])
    p.add_argument("--world", default=WORLD)
    p.add_argument("--limit", type=int, default=None,
                   help="only the first N marketable items (for testing)")
    p.add_argument("--depth", type=int, default=LISTINGS_PER_ITEM,
                   help="order-book rows to keep per item; 0 = summary only, "
                        "which is ~10x smaller and enough for the time series")
    a = p.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if a.command == "ref":
        build_reference()
    else:
        snapshot(a.world, a.limit, a.depth)


if __name__ == "__main__":
    main()
