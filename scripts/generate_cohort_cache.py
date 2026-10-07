"""Generate Demographic Cohort Distributions and 2D Density Grid Cache.

Extracts empirical demographic cohort statistics, quantile distributions,
and screen-time vs sleep-duration joint density grid from data/train.csv
without modifying raw datasets (REG-3, ADR-0003). Saves serialized JSON
artifacts to data/cohort_summary.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Standard demographic categories and bins
AGE_BRACKET_BINS = [17.5, 21.5, 25.5, 30.5, 35.5]
AGE_BRACKET_LABELS = ["18-21", "22-25", "26-30", "31-35"]

SCREEN_GRID_BINS = ["0-4", "4-6", "6-8", "8-10", "10-12", "12+"]
SLEEP_GRID_BINS = ["<5", "5-6", "6-7", "7-8", "8+"]

SCREEN_GRID_EDGES = [0.0, 4.0, 6.0, 8.0, 10.0, 12.0, np.inf]
SLEEP_GRID_EDGES = [0.0, 5.0, 6.0, 7.0, 8.0, np.inf]

SCREEN_HIST_EDGES = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
SLEEP_HIST_EDGES = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]


def get_percentiles(clean_values: np.ndarray, percentiles: list[int | float]) -> dict[str, float]:
    """Compute dictionary of percentiles with safe fallbacks."""
    if len(clean_values) == 0:
        return {f"p{int(p)}": 0.0 for p in percentiles}
    vals = np.percentile(clean_values, percentiles)
    return {f"p{int(p)}": float(round(vals[i], 4)) for i, p in enumerate(percentiles)}


def compute_histogram_bins(series: pd.Series, bin_edges: list[float]) -> list[dict[str, Any]]:
    """Compute histogram bin counts and percentages for a numeric series."""
    clean = series.dropna().values
    total = len(clean)
    counts, edges = np.histogram(clean, bins=bin_edges)
    bins_data: list[dict[str, Any]] = []
    for i in range(len(counts)):
        cnt = int(counts[i])
        pct = round(float((cnt / total) * 100.0), 2) if total > 0 else 0.0
        bins_data.append({
            "bin_start": float(round(edges[i], 2)),
            "bin_end": float(round(edges[i + 1], 2)),
            "count": cnt,
            "pct": pct,
        })
    return bins_data


def compute_cohort_metrics(sub_df: pd.DataFrame, cohort_name: str) -> dict[str, Any]:
    """Compute comprehensive behavioral metrics and percentiles for a cohort."""
    count = int(len(sub_df))
    if count == 0:
        empty_pcts = {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
        return {
            "cohort_name": cohort_name,
            "count": 0,
            "sample_count": 0,
            "sample_size": 0,
            "addiction_prevalence": 0.0,
            "mean_screen_time": 0.0,
            "mean_sleep_hours": 0.0,
            "mean_app_opens": 0.0,
            "screen_time": {
                "mean": 0.0,
                "std": 0.0,
                "percentiles": empty_pcts,
            },
            "sleep_hours": {
                "mean": 0.0,
                "std": 0.0,
                "percentiles": empty_pcts,
            },
            "screen_time_percentiles": empty_pcts,
            "sleep_percentiles": empty_pcts,
            "screen_time_bins": [],
            "sleep_bins": [],
            "social_media": {"mean": 0.0},
            "gaming": {"mean": 0.0},
            "work_study": {"mean": 0.0},
            "app_opens": {"mean": 0.0, "p50": 0.0},
            "notifications": {"mean": 0.0, "p50": 0.0},
        }

    target_series = sub_df["addicted_label"].dropna()
    prevalence = float(target_series.mean()) if len(target_series) > 0 else 0.0

    screen_series = sub_df["daily_screen_time_hours"].dropna()
    sleep_series = sub_df["sleep_hours"].dropna()
    social_series = sub_df["social_media_hours"].dropna()
    gaming_series = sub_df["gaming_hours"].dropna()
    work_series = sub_df["work_study_hours"].dropna()
    opens_series = sub_df["app_opens_per_day"].dropna()
    notifs_series = sub_df["notifications_per_day"].dropna()

    screen_mean = float(screen_series.mean()) if len(screen_series) > 0 else 0.0
    screen_std = float(screen_series.std()) if len(screen_series) > 1 else 0.0
    sleep_mean = float(sleep_series.mean()) if len(sleep_series) > 0 else 0.0
    sleep_std = float(sleep_series.std()) if len(sleep_series) > 1 else 0.0

    screen_pcts = get_percentiles(screen_series.values, [10, 25, 50, 75, 90])
    sleep_pcts = get_percentiles(sleep_series.values, [10, 25, 50, 75, 90])

    social_mean = float(social_series.mean()) if len(social_series) > 0 else 0.0
    gaming_mean = float(gaming_series.mean()) if len(gaming_series) > 0 else 0.0
    work_mean = float(work_series.mean()) if len(work_series) > 0 else 0.0

    opens_mean = float(opens_series.mean()) if len(opens_series) > 0 else 0.0
    opens_p50 = float(np.percentile(opens_series.values, 50)) if len(opens_series) > 0 else 0.0

    notifs_mean = float(notifs_series.mean()) if len(notifs_series) > 0 else 0.0
    notifs_p50 = float(np.percentile(notifs_series.values, 50)) if len(notifs_series) > 0 else 0.0

    screen_hist = compute_histogram_bins(screen_series, SCREEN_HIST_EDGES)
    sleep_hist = compute_histogram_bins(sleep_series, SLEEP_HIST_EDGES)

    return {
        "cohort_name": cohort_name,
        "count": count,
        "sample_count": count,
        "sample_size": count,
        "addiction_prevalence": round(prevalence, 4),
        "mean_screen_time": round(screen_mean, 2),
        "mean_sleep_hours": round(sleep_mean, 2),
        "mean_app_opens": round(opens_mean, 2),
        "screen_time": {
            "mean": round(screen_mean, 4),
            "std": round(screen_std, 4),
            "percentiles": screen_pcts,
        },
        "sleep_hours": {
            "mean": round(sleep_mean, 4),
            "std": round(sleep_std, 4),
            "percentiles": sleep_pcts,
        },
        "screen_time_percentiles": screen_pcts,
        "sleep_percentiles": sleep_pcts,
        "screen_time_bins": screen_hist,
        "sleep_bins": sleep_hist,
        "social_media": {"mean": round(social_mean, 4)},
        "gaming": {"mean": round(gaming_mean, 4)},
        "work_study": {"mean": round(work_mean, 4)},
        "app_opens": {"mean": round(opens_mean, 4), "p50": round(opens_p50, 4)},
        "notifications": {"mean": round(notifs_mean, 4), "p50": round(notifs_p50, 4)},
    }


def compute_screen_sleep_grid(df: pd.DataFrame) -> dict[str, Any]:
    """Compute 2D Screen-Time vs Sleep-Duration Joint Density & Prevalence Grid (AC-2.3)."""
    valid = df.dropna(subset=["daily_screen_time_hours", "sleep_hours", "addicted_label"]).copy()
    total_valid = len(valid)

    valid["screen_bin"] = pd.cut(
        valid["daily_screen_time_hours"],
        bins=SCREEN_GRID_EDGES,
        right=False,
        labels=SCREEN_GRID_BINS,
    )
    valid["sleep_bin"] = pd.cut(
        valid["sleep_hours"],
        bins=SLEEP_GRID_EDGES,
        right=False,
        labels=SLEEP_GRID_BINS,
    )

    grid_cells: list[dict[str, Any]] = []
    density_matrix: list[list[float]] = []
    density_matrix_pct: list[list[float]] = []
    addiction_rate_matrix: list[list[float]] = []
    addiction_rate_matrix_pct: list[list[float]] = []
    sample_count_matrix: list[list[int]] = []

    for s_idx, s_bin in enumerate(SCREEN_GRID_BINS):
        density_row: list[float] = []
        density_pct_row: list[float] = []
        addiction_row: list[float] = []
        addiction_pct_row: list[float] = []
        count_row: list[int] = []

        for sl_idx, sl_bin in enumerate(SLEEP_GRID_BINS):
            sub = valid[(valid["screen_bin"] == s_bin) & (valid["sleep_bin"] == sl_bin)]
            sample_count = int(len(sub))
            addicted_count = int(sub["addicted_label"].sum())

            density = float(sample_count / total_valid) if total_valid > 0 else 0.0
            density_pct = float(density * 100.0)
            addiction_rate = float(addicted_count / sample_count) if sample_count > 0 else 0.0
            addiction_rate_pct = float(addiction_rate * 100.0)

            density_row.append(round(density, 6))
            density_pct_row.append(round(density_pct, 4))
            addiction_row.append(round(addiction_rate, 4))
            addiction_pct_row.append(round(addiction_rate_pct, 3))
            count_row.append(sample_count)

            grid_cells.append({
                "screen_bin": s_bin,
                "sleep_bin": sl_bin,
                "screen_idx": s_idx,
                "sleep_idx": sl_idx,
                "sample_count": sample_count,
                "cell_density_pct": round(density_pct, 3),
                "addiction_rate_pct": round(addiction_rate_pct, 3),
                "density": round(density, 6),
                "addiction_rate": round(addiction_rate, 4),
            })

        density_matrix.append(density_row)
        density_matrix_pct.append(density_pct_row)
        addiction_rate_matrix.append(addiction_row)
        addiction_rate_matrix_pct.append(addiction_pct_row)
        sample_count_matrix.append(count_row)

    return {
        "screen_bins": SCREEN_GRID_BINS,
        "sleep_bins": SLEEP_GRID_BINS,
        "screen_bin_edges": [0.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        "sleep_bin_edges": [5.0, 6.0, 7.0, 8.0],
        "density_matrix": density_matrix,
        "density_matrix_pct": density_matrix_pct,
        "addiction_rate_matrix": addiction_rate_matrix,
        "addiction_rate_matrix_pct": addiction_rate_matrix_pct,
        "sample_count_matrix": sample_count_matrix,
        "grid_cells": grid_cells,
        "total_valid_samples": total_valid,
    }


def compute_population_overall(df: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate population metrics and full 0-100 percentile curves."""
    pop = compute_cohort_metrics(df, cohort_name="Overall Population")

    screen_clean = df["daily_screen_time_hours"].dropna().values
    sleep_clean = df["sleep_hours"].dropna().values

    quantiles_range = np.linspace(0, 100, 101)
    screen_quantiles = [float(round(q, 4)) for q in np.percentile(screen_clean, quantiles_range)]
    sleep_quantiles = [float(round(q, 4)) for q in np.percentile(sleep_clean, quantiles_range)]

    pop["screen_time_quantiles_100"] = screen_quantiles
    pop["sleep_hours_quantiles_100"] = sleep_quantiles

    return pop


def compute_quantile_benchmarks(df: pd.DataFrame) -> dict[str, Any]:
    """Pre-aggregate quantile distributions for sub-millisecond benchmark overlay lookups."""
    screen_clean = df["daily_screen_time_hours"].dropna().values
    sleep_clean = df["sleep_hours"].dropna().values

    quantiles_range = np.linspace(0, 100, 101)
    screen_quantiles = [float(round(q, 4)) for q in np.percentile(screen_clean, quantiles_range)]
    sleep_quantiles = [float(round(q, 4)) for q in np.percentile(sleep_clean, quantiles_range)]

    return {
        "population_mean_screen": float(round(screen_clean.mean(), 4)),
        "population_mean_sleep": float(round(sleep_clean.mean(), 4)),
        "screen_time": {
            "mean": float(round(screen_clean.mean(), 4)),
            "std": float(round(screen_clean.std(), 4)),
            "quantiles_100": screen_quantiles,
            "percentiles": get_percentiles(screen_clean, [10, 25, 50, 75, 90]),
        },
        "sleep_hours": {
            "mean": float(round(sleep_clean.mean(), 4)),
            "std": float(round(sleep_clean.std(), 4)),
            "quantiles_100": sleep_quantiles,
            "percentiles": get_percentiles(sleep_clean, [10, 25, 50, 75, 90]),
        },
    }


def compute_all_dimensions(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    """Compute demographic cohort breakdowns overall and conditioned on stress and gender."""
    work_df = df.copy()
    work_df["age_bracket"] = pd.cut(
        work_df["age"],
        bins=AGE_BRACKET_BINS,
        labels=AGE_BRACKET_LABELS,
    )

    dimensions_config = {
        "age_bracket": AGE_BRACKET_LABELS,
        "gender": ["Female", "Male", "Other"],
        "stress_level": ["Low", "Medium", "High"],
        "academic_work_impact": ["Yes", "No"],
    }

    dimensions_summary: dict[str, Any] = {}
    cohorts_by_dimension: dict[str, list[dict[str, Any]]] = {}

    stress_categories = ["Low", "Medium", "High"]
    gender_categories = ["Female", "Male", "Other"]

    for dim_name, cat_labels in dimensions_config.items():
        all_cohorts: list[dict[str, Any]] = []
        for cat in cat_labels:
            sub = work_df[work_df[dim_name] == cat]
            cohort_data = compute_cohort_metrics(sub, cohort_name=str(cat))
            all_cohorts.append(cohort_data)

        by_stress: dict[str, list[dict[str, Any]]] = {}
        if dim_name != "stress_level":
            for stress_val in stress_categories:
                stress_cohorts: list[dict[str, Any]] = []
                for cat in cat_labels:
                    sub = work_df[(work_df[dim_name] == cat) & (work_df["stress_level"] == stress_val)]
                    cohort_data = compute_cohort_metrics(sub, cohort_name=f"{cat}")
                    stress_cohorts.append(cohort_data)
                by_stress[stress_val] = stress_cohorts

        by_gender: dict[str, list[dict[str, Any]]] = {}
        if dim_name != "gender":
            for gender_val in gender_categories:
                gender_cohorts: list[dict[str, Any]] = []
                for cat in cat_labels:
                    sub = work_df[(work_df[dim_name] == cat) & (work_df["gender"] == gender_val)]
                    cohort_data = compute_cohort_metrics(sub, cohort_name=f"{cat}")
                    gender_cohorts.append(cohort_data)
                by_gender[gender_val] = gender_cohorts

        dimensions_summary[dim_name] = {
            "all": all_cohorts,
            "by_stress": by_stress,
            "by_gender": by_gender,
        }
        cohorts_by_dimension[dim_name] = all_cohorts

    # Include alias 'age_group' for interchangeable nomenclature
    dimensions_summary["age_group"] = dimensions_summary["age_bracket"]
    cohorts_by_dimension["age_group"] = cohorts_by_dimension["age_bracket"]

    return dimensions_summary, cohorts_by_dimension


def generate_cohort_summary(train_path: Path, output_path: Path) -> dict[str, Any]:
    """Read train.csv in read-only mode (REG-3), compute cohort summary, and write JSON."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {train_path}")

    # Read-only load of data/train.csv without modifying raw data (REG-3)
    train_df = pd.read_csv(train_path)

    dimensions_summary, cohorts_by_dim = compute_all_dimensions(train_df)
    pop_overall = compute_population_overall(train_df)
    screen_sleep_grid = compute_screen_sleep_grid(train_df)
    quantile_benchmarks = compute_quantile_benchmarks(train_df)

    cache: dict[str, Any] = {
        "version": "1.0.0",
        "total_population_records": int(len(train_df)),
        "population_overall": pop_overall,
        "dimensions": dimensions_summary,
        "cohorts_by_dimension": cohorts_by_dim,
        "screen_sleep_matrix": screen_sleep_grid,
        "quantile_benchmarks": quantile_benchmarks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    return cache


def validate_cohort_summary(data_or_path: dict[str, Any] | Path | str) -> bool:
    """Validate schema integrity, non-zero counts, and distribution validity."""
    if isinstance(data_or_path, (str, Path)):
        path = Path(data_or_path)
        if not path.exists():
            raise FileNotFoundError(f"Cohort cache not found at {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = data_or_path

    # 1. Root structure
    required_root_keys = [
        "version",
        "total_population_records",
        "population_overall",
        "dimensions",
        "cohorts_by_dimension",
        "screen_sleep_matrix",
        "quantile_benchmarks",
    ]
    for k in required_root_keys:
        assert k in data, f"Schema validation error: Missing root key '{k}'"

    total_records = data["total_population_records"]
    assert total_records == 691369, f"Expected 691,369 records, found {total_records}"

    # 2. Population overall
    pop = data["population_overall"]
    assert pop["sample_count"] == 691369, f"Expected 691,369 in population_overall sample_count, got {pop['sample_count']}"
    assert 0.0 <= pop["addiction_prevalence"] <= 1.0, "Invalid addiction prevalence in population_overall"
    assert 7.55 <= pop["screen_time"]["mean"] <= 7.75, f"Screen time mean out of bounds: {pop['screen_time']['mean']}"
    assert 6.70 <= pop["sleep_hours"]["mean"] <= 6.90, f"Sleep hours mean out of bounds: {pop['sleep_hours']['mean']}"
    assert len(pop["screen_time_quantiles_100"]) == 101, "Expected 101 percentiles for screen_time"
    assert len(pop["sleep_hours_quantiles_100"]) == 101, "Expected 101 percentiles for sleep_hours"

    for q_arr, name in [
        (pop["screen_time_quantiles_100"], "screen_time"),
        (pop["sleep_hours_quantiles_100"], "sleep_hours"),
    ]:
        for i in range(len(q_arr) - 1):
            assert q_arr[i] <= q_arr[i + 1], f"Quantiles for {name} must be non-decreasing at index {i}"

    # 3. Demographic dimensions
    dims = data["dimensions"]
    expected_dims = {
        "age_bracket": ["18-21", "22-25", "26-30", "31-35"],
        "gender": ["Female", "Male", "Other"],
        "stress_level": ["Low", "Medium", "High"],
        "academic_work_impact": ["Yes", "No"],
    }

    for dim_name, exp_labels in expected_dims.items():
        assert dim_name in dims, f"Dimension '{dim_name}' missing from dimensions"
        dim_entry = dims[dim_name]
        assert "all" in dim_entry, f"'all' cohorts missing for dimension '{dim_name}'"
        cohorts = dim_entry["all"]
        assert len(cohorts) == len(exp_labels), f"Expected {len(exp_labels)} cohorts for '{dim_name}', got {len(cohorts)}"

        for c_idx, c in enumerate(cohorts):
            assert c["cohort_name"] == exp_labels[c_idx], f"Cohort name mismatch: {c['cohort_name']} != {exp_labels[c_idx]}"
            assert c["sample_count"] > 0, f"Cohort '{c['cohort_name']}' has 0 samples"
            assert 0.0 <= c["addiction_prevalence"] <= 1.0, f"Invalid prevalence in '{c['cohort_name']}'"
            assert c["screen_time"]["mean"] > 0, f"Non-positive screen mean in '{c['cohort_name']}'"
            assert c["sleep_hours"]["mean"] > 0, f"Non-positive sleep mean in '{c['cohort_name']}'"
            assert c["screen_time"]["std"] > 0, f"Non-positive screen std in '{c['cohort_name']}'"
            assert c["sleep_hours"]["std"] > 0, f"Non-positive sleep std in '{c['cohort_name']}'"

            # Percentile monotonic check
            for p_dict in [c["screen_time"]["percentiles"], c["sleep_hours"]["percentiles"]]:
                assert p_dict["p10"] <= p_dict["p25"] <= p_dict["p50"] <= p_dict["p75"] <= p_dict["p90"]

            # App opens and notifications
            assert c["app_opens"]["mean"] > 0 and c["app_opens"]["p50"] > 0
            assert c["notifications"]["mean"] > 0 and c["notifications"]["p50"] > 0

        # Stress conditioned sub-cohorts
        if "by_stress" in dim_entry and dim_entry["by_stress"]:
            for s_val in ["Low", "Medium", "High"]:
                assert s_val in dim_entry["by_stress"], f"Conditioned stress '{s_val}' missing for '{dim_name}'"
                sub_cohorts = dim_entry["by_stress"][s_val]
                for sc in sub_cohorts:
                    assert sc["sample_count"] > 0, f"Conditioned cohort '{sc['cohort_name']}' has 0 samples"
                    assert 0.0 <= sc["addiction_prevalence"] <= 1.0

    # 4. Screen-sleep joint density grid
    grid = data["screen_sleep_matrix"]
    assert len(grid["screen_bins"]) == 6, f"Expected 6 screen bins, got {len(grid['screen_bins'])}"
    assert len(grid["sleep_bins"]) == 5, f"Expected 5 sleep bins, got {len(grid['sleep_bins'])}"
    assert len(grid["grid_cells"]) == 30, f"Expected 30 grid cells, got {len(grid['grid_cells'])}"
    assert len(grid["density_matrix"]) == 6
    assert all(len(r) == 5 for r in grid["density_matrix"])
    assert len(grid["addiction_rate_matrix"]) == 6
    assert all(len(r) == 5 for r in grid["addiction_rate_matrix"])

    total_density_pct = 0.0
    total_grid_samples = 0
    for cell in grid["grid_cells"]:
        assert cell["sample_count"] > 0, f"Cell {cell['screen_bin']} x {cell['sleep_bin']} has 0 samples"
        assert 0.0 <= cell["cell_density_pct"] <= 100.0, f"Invalid cell_density_pct in cell: {cell}"
        assert 0.0 <= cell["addiction_rate_pct"] <= 100.0, f"Invalid addiction_rate_pct in cell: {cell}"
        total_density_pct += cell["cell_density_pct"]
        total_grid_samples += cell["sample_count"]

    assert abs(total_density_pct - 100.0) < 0.5, f"Total density pct {total_density_pct:.2f}% deviates from 100%"
    assert total_grid_samples == grid["total_valid_samples"], "Grid samples count mismatch"

    # 5. Quantile benchmarks
    q_bench = data["quantile_benchmarks"]
    assert 7.55 <= q_bench["population_mean_screen"] <= 7.75, "Quantile bench screen mean mismatch"
    assert 6.70 <= q_bench["population_mean_sleep"] <= 6.90, "Quantile bench sleep mean mismatch"
    assert len(q_bench["screen_time"]["quantiles_100"]) == 101
    assert len(q_bench["sleep_hours"]["quantiles_100"]) == 101

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-aggregate demographic cohort distributions and 2D density grid cache."
    )
    parser.add_argument("--train-path", type=str, default="data/train.csv", help="Path to raw train.csv")
    parser.add_argument("--output", type=str, default="data/cohort_summary.json", help="Path to output cohort_summary.json")
    parser.add_argument("--validate", action="store_true", help="Run self-verification on output JSON cache")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing JSON cache without regenerating")
    args = parser.parse_args()

    train_path = Path(args.train_path).resolve()
    output_path = Path(args.output).resolve()

    if not args.validate_only:
        print(f"Generating cohort cache from {train_path}...")
        cache = generate_cohort_summary(train_path, output_path)
        print(f"Generated cohort cache successfully: {output_path} ({cache['total_population_records']:,} rows).")
    else:
        print(f"Validating existing cohort cache at {output_path}...")

    if args.validate:
        print(f"Validating cohort cache integrity at {output_path}...")
        is_valid = validate_cohort_summary(output_path)
        if is_valid:
            print("Cohort cache validation PASSED: Schema integrity, non-zero sample counts, and percentage ranges verified.")
            sys.exit(0)
        else:
            print("Cohort cache validation FAILED.")
            sys.exit(1)


if __name__ == "__main__":
    main()
