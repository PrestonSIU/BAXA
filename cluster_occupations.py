#!/usr/bin/env python3
"""Cluster 1970 Census occupations by historical DOT task requirements.

The model is intentionally small and dependency-light. It standardizes the five
Autor-Levy-Murnane task measures, fits deterministic k-means to occupations (not
person-year rows), and attaches the resulting groups to the NLSY79 panel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SOURCE_FEATURES = ("ehf", "finger", "dcp", "sts", "math")
OUTPUT_FEATURES = {
    "ehf": "TASK_NONROUTINE_MANUAL",
    "finger": "TASK_ROUTINE_MANUAL",
    "dcp": "TASK_NONROUTINE_COGNITIVE_INTERPERSONAL",
    "sts": "TASK_ROUTINE_COGNITIVE",
    "math": "TASK_NONROUTINE_COGNITIVE_ANALYTICAL",
}
DISPLAY_FEATURES = {
    "ehf": "nonroutine manual/physical",
    "finger": "routine manual",
    "dcp": "interpersonal",
    "sts": "routine cognitive",
    "math": "analytical",
}


@dataclass(frozen=True)
class KMeansResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    iterations: int


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _squared_distances(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _initialize_kmeans_pp(
    x: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    centers = [x[int(rng.integers(0, len(x)))]]
    selected = {tuple(centers[0])}
    while len(centers) < n_clusters:
        distance = np.min(_squared_distances(x, np.asarray(centers)), axis=1)
        total = float(distance.sum())
        if total <= 0:
            candidates = [row for row in x if tuple(row) not in selected]
            if not candidates:
                raise ValueError("Fewer distinct rows than requested clusters.")
            chosen = candidates[0]
        else:
            chosen = x[int(rng.choice(len(x), p=distance / total))]
        centers.append(chosen)
        selected.add(tuple(chosen))
    return np.asarray(centers, dtype=float)


def _fit_once(
    x: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    max_iter: int,
    tolerance: float,
) -> KMeansResult:
    centers = _initialize_kmeans_pp(x, n_clusters, rng)
    labels = np.zeros(len(x), dtype=int)

    for iteration in range(1, max_iter + 1):
        distances = _squared_distances(x, centers)
        labels = np.argmin(distances, axis=1)
        new_centers = centers.copy()
        nearest_distance = distances[np.arange(len(x)), labels]

        for cluster in range(n_clusters):
            members = x[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
            else:
                # Re-seed an empty cluster at the worst represented occupation.
                new_centers[cluster] = x[int(np.argmax(nearest_distance))]

        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers
        if shift <= tolerance:
            break

    distances = _squared_distances(x, centers)
    labels = np.argmin(distances, axis=1)
    inertia = float(distances[np.arange(len(x)), labels].sum())
    return KMeansResult(labels, centers, inertia, iteration)


def fit_kmeans(
    x: np.ndarray,
    n_clusters: int,
    seed: int,
    n_init: int = 50,
    max_iter: int = 300,
    tolerance: float = 1e-7,
) -> KMeansResult:
    if x.ndim != 2 or len(x) < n_clusters:
        raise ValueError("The feature matrix must have at least one row per cluster.")
    if not 2 <= n_clusters <= 20:
        raise ValueError("n_clusters must be between 2 and 20.")

    seed_sequence = np.random.SeedSequence(seed)
    best: KMeansResult | None = None
    for child_seed in seed_sequence.spawn(n_init):
        result = _fit_once(
            x,
            n_clusters,
            np.random.default_rng(child_seed),
            max_iter,
            tolerance,
        )
        if best is None or result.inertia < best.inertia - 1e-10:
            best = result
    assert best is not None
    return best


def silhouette_score(x: np.ndarray, labels: np.ndarray) -> float:
    """Return the ordinary unweighted mean silhouette across occupations."""
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(x):
        return float("nan")
    distances = np.sqrt(np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=2))
    scores = np.zeros(len(x), dtype=float)
    for row in range(len(x)):
        same = labels == labels[row]
        same[row] = False
        a = float(distances[row, same].mean()) if same.any() else 0.0
        b = min(float(distances[row, labels == other].mean()) for other in unique if other != labels[row])
        scores[row] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def cluster_count_diagnostics(
    tasks: pd.DataFrame, seed: int, minimum: int = 2, maximum: int = 8
) -> pd.DataFrame:
    raw_x = tasks[list(SOURCE_FEATURES)].to_numpy(dtype=float)
    x = (raw_x - raw_x.mean(axis=0)) / raw_x.std(axis=0, ddof=0)
    rows = []
    for n_clusters in range(minimum, maximum + 1):
        result = fit_kmeans(x, n_clusters=n_clusters, seed=seed)
        rows.append(
            {
                "N_CLUSTERS": n_clusters,
                "INERTIA_STANDARDIZED_SPACE": result.inertia,
                "SILHOUETTE_STANDARDIZED_SPACE": silhouette_score(x, result.labels),
            }
        )
    return pd.DataFrame(rows)


def _base_cluster_name(center: np.ndarray) -> str:
    z = dict(zip(SOURCE_FEATURES, center))
    if z["dcp"] > 0.5 and z["math"] > 0.5:
        return "Analytical and interpersonal"
    if z["ehf"] > 0.5:
        return "Physical and nonroutine manual"
    if z["finger"] > 0.5 or z["sts"] > 0.5:
        return "Routine cognitive and manual"
    if max(z.values()) < 0:
        return "Lower measured task intensity"
    ordered = sorted(SOURCE_FEATURES, key=lambda feature: z[feature], reverse=True)
    return f"{DISPLAY_FEATURES[ordered[0]].title()} and {DISPLAY_FEATURES[ordered[1]]}"


def canonicalize_clusters(result: KMeansResult) -> tuple[KMeansResult, list[str]]:
    """Give clusters stable, human-readable IDs independent of initialization."""
    base_names = [_base_cluster_name(center) for center in result.centers]
    preferred_order = {
        "Physical and nonroutine manual": 0,
        "Routine cognitive and manual": 1,
        "Analytical and interpersonal": 2,
        "Lower measured task intensity": 3,
    }
    order = sorted(
        range(len(base_names)),
        key=lambda index: (
            preferred_order.get(base_names[index], 10),
            base_names[index],
            tuple(np.round(result.centers[index], 12)),
        ),
    )
    old_to_new = {old: new for new, old in enumerate(order)}
    labels = np.array([old_to_new[int(label)] for label in result.labels], dtype=int)
    centers = result.centers[order]

    seen: dict[str, int] = {}
    names: list[str] = []
    for old in order:
        name = base_names[old]
        seen[name] = seen.get(name, 0) + 1
        suffix = f" {seen[name]}" if seen[name] > 1 else ""
        names.append(f"{name}{suffix}")
    return KMeansResult(labels, centers, result.inertia, result.iterations), names


def modal_text(values: Iterable[object]) -> object:
    cleaned = pd.Series(list(values), dtype="string").dropna()
    cleaned = cleaned[cleaned.str.strip().ne("")]
    if cleaned.empty:
        return pd.NA
    counts = cleaned.value_counts()
    return sorted(counts[counts == counts.max()].index.tolist())[0]


def load_task_measures(path: Path) -> pd.DataFrame:
    tasks = pd.read_csv(path)
    required = {"occ70", *SOURCE_FEATURES}
    missing = sorted(required.difference(tasks.columns))
    if missing:
        raise ValueError(f"Task file is missing columns: {', '.join(missing)}")
    tasks = tasks[["occ70", *SOURCE_FEATURES]].copy()
    tasks["occ70"] = pd.to_numeric(tasks["occ70"], errors="raise").astype("int64")
    for feature in SOURCE_FEATURES:
        tasks[feature] = pd.to_numeric(tasks[feature], errors="raise")
    if tasks["occ70"].duplicated().any():
        duplicates = tasks.loc[tasks["occ70"].duplicated(), "occ70"].tolist()
        raise ValueError(f"Task file contains duplicate occupation codes: {duplicates[:10]}")
    if tasks[list(SOURCE_FEATURES)].isna().any().any():
        raise ValueError("Task features may not be missing.")
    return tasks.sort_values("occ70").reset_index(drop=True)


def load_panel(path: Path, occupation_column: str, title_column: str) -> pd.DataFrame:
    required = ["CASEID", "YEAR", "SAMPWEIGHT", occupation_column, title_column]
    panel = pd.read_csv(path, usecols=required, low_memory=False)
    panel[occupation_column] = pd.to_numeric(panel[occupation_column], errors="coerce").astype("Int64")
    panel["SAMPWEIGHT"] = pd.to_numeric(panel["SAMPWEIGHT"], errors="coerce")
    panel["YEAR"] = pd.to_numeric(panel["YEAR"], errors="coerce").astype("Int64")
    return panel


def occupation_statistics(
    panel: pd.DataFrame, occupation_column: str, title_column: str
) -> pd.DataFrame:
    observed = panel.dropna(subset=[occupation_column]).copy()
    grouped = observed.groupby(occupation_column, sort=True, observed=True)
    stats = grouped.agg(
        OCCUPATION_TITLE=(title_column, modal_text),
        NLSY_PERSON_YEARS=("CASEID", "size"),
        NLSY_UNIQUE_PEOPLE=("CASEID", "nunique"),
        FIRST_YEAR=("YEAR", "min"),
        LAST_YEAR=("YEAR", "max"),
        NLSY_WEIGHTED_PERSON_YEARS=("SAMPWEIGHT", "sum"),
    ).reset_index()
    return stats.rename(columns={occupation_column: "occ70"})


def build_outputs(
    tasks: pd.DataFrame,
    panel: pd.DataFrame,
    occupation_column: str,
    title_column: str,
    n_clusters: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw_x = tasks[list(SOURCE_FEATURES)].to_numpy(dtype=float)
    means = raw_x.mean(axis=0)
    standard_deviations = raw_x.std(axis=0, ddof=0)
    if np.any(standard_deviations <= 0):
        raise ValueError("Every task feature must have positive variation.")
    standardized_x = (raw_x - means) / standard_deviations

    result = fit_kmeans(standardized_x, n_clusters=n_clusters, seed=seed)
    result, cluster_names = canonicalize_clusters(result)
    cluster_ids = result.labels + 1
    distances = np.sqrt(
        np.sum((standardized_x - result.centers[result.labels]) ** 2, axis=1)
    )

    occupations = tasks.copy()
    occupations["TASK_CLUSTER_ID"] = cluster_ids
    occupations["TASK_CLUSTER_LABEL"] = [cluster_names[label] for label in result.labels]
    occupations["DISTANCE_TO_CENTROID"] = distances
    for index, feature in enumerate(SOURCE_FEATURES):
        occupations[f"Z_{OUTPUT_FEATURES[feature]}"] = standardized_x[:, index]

    stats = occupation_statistics(panel, occupation_column, title_column)
    occupations = occupations.merge(stats, on="occ70", how="left", validate="one_to_one")
    occupations["NLSY_PERSON_YEARS"] = occupations["NLSY_PERSON_YEARS"].fillna(0).astype("int64")
    occupations["NLSY_UNIQUE_PEOPLE"] = occupations["NLSY_UNIQUE_PEOPLE"].fillna(0).astype("int64")
    occupations["NLSY_WEIGHTED_PERSON_YEARS"] = occupations["NLSY_WEIGHTED_PERSON_YEARS"].fillna(0.0)
    occupations["FIRST_YEAR"] = occupations["FIRST_YEAR"].astype("Int64")
    occupations["LAST_YEAR"] = occupations["LAST_YEAR"].astype("Int64")
    occupations = occupations.rename(columns={"occ70": "OCC1970"})
    occupations = occupations.rename(columns=OUTPUT_FEATURES)

    lookup_columns = [
        "OCC1970",
        "TASK_CLUSTER_ID",
        "TASK_CLUSTER_LABEL",
        "DISTANCE_TO_CENTROID",
        *OUTPUT_FEATURES.values(),
    ]
    person_year = panel.merge(
        occupations[lookup_columns],
        left_on=occupation_column,
        right_on="OCC1970",
        how="left",
        validate="many_to_one",
    )
    person_year["TASK_CLUSTER_ID"] = person_year["TASK_CLUSTER_ID"].astype("Int64")
    person_year = person_year.drop(columns=["OCC1970"])

    matched = person_year["TASK_CLUSTER_ID"].notna()
    matched_weight = person_year.loc[matched, "SAMPWEIGHT"].sum()
    summary_rows: list[dict[str, object]] = []
    raw_centers = result.centers * standard_deviations + means
    for cluster_index, cluster_name in enumerate(cluster_names):
        cluster_id = cluster_index + 1
        occupation_mask = occupations["TASK_CLUSTER_ID"].eq(cluster_id)
        panel_mask = person_year["TASK_CLUSTER_ID"].eq(cluster_id)
        row: dict[str, object] = {
            "TASK_CLUSTER_ID": cluster_id,
            "TASK_CLUSTER_LABEL": cluster_name,
            "DOT_OCCUPATIONS": int(occupation_mask.sum()),
            "NLSY_OBSERVED_OCCUPATIONS": int(
                (occupation_mask & occupations["NLSY_PERSON_YEARS"].gt(0)).sum()
            ),
            "NLSY_PERSON_YEARS": int(panel_mask.sum()),
            "NLSY_PERSON_YEAR_SHARE": float(panel_mask.sum() / matched.sum()) if matched.any() else float("nan"),
            "NLSY_WEIGHTED_PERSON_YEAR_SHARE": float(
                person_year.loc[panel_mask, "SAMPWEIGHT"].sum() / matched_weight
            )
            if matched_weight > 0
            else float("nan"),
        }
        for feature_index, feature in enumerate(SOURCE_FEATURES):
            row[f"MEAN_{OUTPUT_FEATURES[feature]}"] = float(raw_centers[cluster_index, feature_index])
            row[f"Z_{OUTPUT_FEATURES[feature]}"] = float(result.centers[cluster_index, feature_index])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    metadata: dict[str, object] = {
        "algorithm": "k-means++ with 50 deterministic initializations",
        "fit_unit": "1970 Census occupation; each of 411 occupations has equal weight",
        "n_clusters": n_clusters,
        "random_seed": seed,
        "iterations_for_best_fit": result.iterations,
        "inertia_standardized_space": result.inertia,
        "silhouette_standardized_space": silhouette_score(standardized_x, result.labels),
        "task_occupations": int(len(tasks)),
        "panel_rows": int(len(panel)),
        "panel_rows_with_1970_occupation": int(panel[occupation_column].notna().sum()),
        "panel_rows_matched_to_task_data": int(matched.sum()),
        "matched_share_of_all_panel_rows": float(matched.mean()),
        "matched_share_of_rows_with_1970_occupation": float(
            matched.sum() / panel[occupation_column].notna().sum()
        )
        if panel[occupation_column].notna().any()
        else float("nan"),
        "feature_means": dict(zip(SOURCE_FEATURES, means.tolist())),
        "feature_standard_deviations": dict(zip(SOURCE_FEATURES, standard_deviations.tolist())),
        "cluster_labels_are_descriptive": True,
    }
    return occupations, person_year, summary, metadata


def unmatched_occupation_summary(
    person_year: pd.DataFrame, occupation_column: str, title_column: str
) -> pd.DataFrame:
    unmatched = person_year.loc[
        person_year[occupation_column].notna() & person_year["TASK_CLUSTER_ID"].isna()
    ].copy()
    if unmatched.empty:
        return pd.DataFrame(
            columns=[
                occupation_column,
                title_column,
                "NLSY_PERSON_YEARS",
                "NLSY_UNIQUE_PEOPLE",
                "FIRST_YEAR",
                "LAST_YEAR",
            ]
        )
    return (
        unmatched.groupby([occupation_column, title_column], dropna=False, sort=True)
        .agg(
            NLSY_PERSON_YEARS=("CASEID", "size"),
            NLSY_UNIQUE_PEOPLE=("CASEID", "nunique"),
            FIRST_YEAR=("YEAR", "min"),
            LAST_YEAR=("YEAR", "max"),
        )
        .reset_index()
        .sort_values(["NLSY_PERSON_YEARS", occupation_column], ascending=[False, True])
    )


def write_report(summary: pd.DataFrame, metadata: dict[str, object], output_path: Path) -> None:
    rows = [
        "# NLSY79 task-based occupation clustering",
        "",
        (
            f"The model fits {metadata['n_clusters']} k-means clusters to "
            f"{metadata['task_occupations']} 1970 Census occupations using five standardized "
            "1977 DOT task measures. Occupations receive equal weight during fitting."
        ),
        "",
        "## Fit and coverage",
        "",
        f"- Silhouette score: {metadata['silhouette_standardized_space']:.3f}",
        f"- Standardized-space inertia: {metadata['inertia_standardized_space']:.3f}",
        f"- NLSY79 person-year rows: {metadata['panel_rows']:,}",
        f"- Rows matched to a historical task profile: {metadata['panel_rows_matched_to_task_data']:,} "
        f"({metadata['matched_share_of_all_panel_rows']:.1%})",
        f"- Match rate among rows with a 1970 occupation: "
        f"{metadata['matched_share_of_rows_with_1970_occupation']:.1%}",
        f"- Highest silhouette in the 2–8 cluster diagnostic: "
        f"k={metadata['best_silhouette_k']} (the default k=4 favors economic interpretability)",
        "",
        "## Cluster summary",
        "",
        "| ID | Description | DOT occupations | NLSY occupations | Person-years | Matched share |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        rows.append(
            f"| {row.TASK_CLUSTER_ID} | {row.TASK_CLUSTER_LABEL} | {row.DOT_OCCUPATIONS:,} | "
            f"{row.NLSY_OBSERVED_OCCUPATIONS:,} | {row.NLSY_PERSON_YEARS:,} | "
            f"{row.NLSY_PERSON_YEAR_SHARE:.1%} |"
        )
    rows.extend(
        [
            "",
            "Cluster names summarize centroid patterns and are not occupational status or skill rankings. "
            "Rows without a 1970-basis occupation remain unmatched rather than being imputed from a newer code system.",
        ]
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        type=Path,
        default=project_root / "NLSY79_Focused_Subset_v6.csv",
        help="NLSY79 person-year CSV.",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "dot77_1970_task_measures.csv",
        help="Occupation-level 1977 DOT task measures.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Directory for model outputs.",
    )
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1979)
    parser.add_argument("--occupation-column", default="OCC1970_WEEKLY")
    parser.add_argument("--title-column", default="TITLE1970_WEEKLY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_task_measures(args.tasks)
    panel = load_panel(args.panel, args.occupation_column, args.title_column)
    occupations, person_year, summary, metadata = build_outputs(
        tasks,
        panel,
        args.occupation_column,
        args.title_column,
        args.n_clusters,
        args.seed,
    )
    metadata["panel_file"] = str(args.panel.resolve())
    metadata["panel_sha256"] = file_sha256(args.panel)
    metadata["task_file"] = str(args.tasks.resolve())
    metadata["task_file_sha256"] = file_sha256(args.tasks)
    diagnostics = cluster_count_diagnostics(tasks, args.seed)
    metadata["diagnostic_k_range"] = [2, 8]
    metadata["best_silhouette_k"] = int(
        diagnostics.loc[
            diagnostics["SILHOUETTE_STANDARDIZED_SPACE"].idxmax(), "N_CLUSTERS"
        ]
    )
    unmatched = unmatched_occupation_summary(
        person_year, args.occupation_column, args.title_column
    )
    metadata["unmatched_rows_with_1970_occupation"] = int(
        unmatched["NLSY_PERSON_YEARS"].sum()
    )
    metadata["unmatched_1970_occupation_codes"] = int(
        unmatched[args.occupation_column].nunique()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupations.to_csv(args.output_dir / "occupation_clusters.csv", index=False, float_format="%.8g")
    person_year.to_csv(args.output_dir / "person_year_task_clusters.csv", index=False, float_format="%.8g")
    summary.to_csv(args.output_dir / "cluster_summary.csv", index=False, float_format="%.8g")
    diagnostics.to_csv(args.output_dir / "k_diagnostics.csv", index=False, float_format="%.8g")
    unmatched.to_csv(
        args.output_dir / "unmatched_occupation_codes.csv", index=False, float_format="%.8g"
    )
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(summary, metadata, args.output_dir / "model_report.md")

    print(
        f"Created {args.n_clusters} task clusters; matched "
        f"{metadata['panel_rows_matched_to_task_data']:,} of {metadata['panel_rows']:,} person-year rows."
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
