#!/usr/bin/env python3
"""Cluster NLSY79 occupations across coding eras through a common OCC2010 bridge.

The model uses published O*NET task composites for Census 2000 occupations,
aggregates them to the harmonized IPUMS OCC2010 classification, and maps both
1970- and 2000-basis NLSY79 occupation codes to probability-weighted OCC2010
task profiles. K-means is fitted once to the common OCC2010 profiles.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_occupations import (
    KMeansResult,
    file_sha256,
    fit_kmeans,
    silhouette_score,
)


# Ordered to match the economic interpretation used by the original DOT model.
ONET_FEATURES = (
    "nr_man_phys",
    "r_man",
    "nr_cog_pers",
    "r_cog",
    "nr_cog_anal",
)
OUTPUT_FEATURES = (
    "TASK_NONROUTINE_MANUAL_PHYSICAL",
    "TASK_ROUTINE_MANUAL",
    "TASK_NONROUTINE_COGNITIVE_INTERPERSONAL",
    "TASK_ROUTINE_COGNITIVE",
    "TASK_NONROUTINE_COGNITIVE_ANALYTICAL",
)
def load_onet_tasks(path: Path) -> pd.DataFrame:
    tasks = pd.read_csv(path)
    required = {"occ2000", *ONET_FEATURES}
    missing = sorted(required.difference(tasks.columns))
    if missing:
        raise ValueError(f"O*NET task file is missing columns: {', '.join(missing)}")
    tasks = tasks[["occ2000", *ONET_FEATURES]].copy()
    tasks["occ2000"] = pd.to_numeric(tasks["occ2000"], errors="raise").astype("int64")
    for feature in ONET_FEATURES:
        tasks[feature] = pd.to_numeric(tasks[feature], errors="coerce")
    if tasks["occ2000"].duplicated().any():
        raise ValueError("O*NET task file contains duplicate Census 2000 codes.")
    return tasks.sort_values("occ2000").reset_index(drop=True)


def load_ipums_bridge(path: Path) -> pd.DataFrame:
    bridge = pd.read_csv(
        path,
        usecols=["YEAR", "PERWT", "OCC", "OCC2010"],
        dtype={"YEAR": "int16", "PERWT": "float64", "OCC": "int32", "OCC2010": "int32"},
    )
    bridge = bridge.loc[
        bridge["YEAR"].isin([1970, 2000])
        & bridge["PERWT"].gt(0)
        & bridge["OCC"].gt(0)
        & bridge["OCC2010"].gt(0)
        & bridge["OCC2010"].lt(9990)
    ].copy()
    return bridge


def bridge_paths(bridge: pd.DataFrame, year: int) -> pd.DataFrame:
    paths = (
        bridge.loc[bridge["YEAR"].eq(year)]
        .groupby(["OCC", "OCC2010"], as_index=False)["PERWT"]
        .sum()
        .rename(columns={"OCC": "SOURCE_OCC_3DIGIT", "PERWT": "PATH_WEIGHT"})
    )
    totals = paths.groupby("SOURCE_OCC_3DIGIT")["PATH_WEIGHT"].transform("sum")
    paths["PATH_PROBABILITY"] = paths["PATH_WEIGHT"] / totals
    paths["SOURCE_SCHEME"] = str(year)
    return paths[
        [
            "SOURCE_SCHEME",
            "SOURCE_OCC_3DIGIT",
            "OCC2010",
            "PATH_WEIGHT",
            "PATH_PROBABILITY",
        ]
    ]


def build_common_profiles(
    bridge: pd.DataFrame,
    onet_tasks: pd.DataFrame,
    minimum_source_coverage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths_2000 = bridge_paths(bridge, 2000)
    joined = paths_2000.merge(
        onet_tasks,
        left_on="SOURCE_OCC_3DIGIT",
        right_on="occ2000",
        how="left",
        validate="many_to_one",
    )
    joined["HAS_TASK_PROFILE"] = joined[list(ONET_FEATURES)].notna().all(axis=1)
    total_weight = joined.groupby("OCC2010")["PATH_WEIGHT"].sum()
    profiled_weight = (
        joined.loc[joined["HAS_TASK_PROFILE"]]
        .groupby("OCC2010")["PATH_WEIGHT"]
        .sum()
    )

    valid = joined.loc[joined["HAS_TASK_PROFILE"]].copy()
    for feature in ONET_FEATURES:
        valid[f"_{feature}_weighted"] = valid[feature] * valid["PATH_WEIGHT"]
    aggregations: dict[str, tuple[str, str]] = {
        "PROFILED_SOURCE_WEIGHT": ("PATH_WEIGHT", "sum"),
        "CENSUS2000_SOURCE_CODES": ("SOURCE_OCC_3DIGIT", "nunique"),
    }
    aggregations.update(
        {feature: (f"_{feature}_weighted", "sum") for feature in ONET_FEATURES}
    )
    profiles = valid.groupby("OCC2010", as_index=False).agg(**aggregations)
    for feature in ONET_FEATURES:
        profiles[feature] = profiles[feature] / profiles["PROFILED_SOURCE_WEIGHT"]
    profiles["TOTAL_SOURCE_WEIGHT"] = profiles["OCC2010"].map(total_weight)
    profiles["TASK_SOURCE_COVERAGE"] = (
        profiles["OCC2010"].map(profiled_weight) / profiles["TOTAL_SOURCE_WEIGHT"]
    )
    profiles["TASK_PROFILE_ELIGIBLE"] = profiles["TASK_SOURCE_COVERAGE"].ge(
        minimum_source_coverage
    )
    return profiles.sort_values("OCC2010").reset_index(drop=True), paths_2000


def _cluster_name(center: np.ndarray) -> str:
    physical, routine_manual, interpersonal, routine_cognitive, analytical = center
    if analytical > 0.5 and interpersonal > 0.5:
        return "Analytical and interpersonal"
    if physical > 0.5 and routine_manual > 0.5:
        return "Physical and routine manual"
    if physical > 0.5:
        return "Physical and nonroutine manual"
    if routine_cognitive > 0.35:
        return "Routine cognitive"
    order = np.argsort(center)[::-1]
    labels = ["physical", "routine manual", "interpersonal", "routine cognitive", "analytical"]
    return f"{labels[int(order[0])].title()} and {labels[int(order[1])]}"


def canonicalize_clusters(result: KMeansResult) -> tuple[KMeansResult, list[str]]:
    base_names = [_cluster_name(center) for center in result.centers]
    preferred = {
        "Physical and nonroutine manual": 0,
        "Physical and routine manual": 1,
        "Routine cognitive": 2,
        "Analytical and interpersonal": 3,
    }
    order = sorted(
        range(len(base_names)),
        key=lambda index: (
            preferred.get(base_names[index], 10),
            base_names[index],
            tuple(np.round(result.centers[index], 12)),
        ),
    )
    old_to_new = {old: new for new, old in enumerate(order)}
    labels = np.array([old_to_new[int(value)] for value in result.labels], dtype=int)
    centers = result.centers[order]
    seen: dict[str, int] = {}
    names = []
    for old in order:
        name = base_names[old]
        seen[name] = seen.get(name, 0) + 1
        suffix = f" {seen[name]}" if seen[name] > 1 else ""
        names.append(f"{name}{suffix}")
    return KMeansResult(labels, centers, result.inertia, result.iterations), names


def fit_common_clusters(
    profiles: pd.DataFrame, n_clusters: int, seed: int
) -> tuple[pd.DataFrame, KMeansResult, list[str], np.ndarray, np.ndarray]:
    eligible = profiles.loc[profiles["TASK_PROFILE_ELIGIBLE"]].copy()
    raw_x = eligible[list(ONET_FEATURES)].to_numpy(dtype=float)
    means = raw_x.mean(axis=0)
    standard_deviations = raw_x.std(axis=0, ddof=0)
    standardized = (raw_x - means) / standard_deviations
    result = fit_kmeans(standardized, n_clusters=n_clusters, seed=seed)
    result, names = canonicalize_clusters(result)
    eligible["TASK_CLUSTER_ID"] = result.labels + 1
    eligible["TASK_CLUSTER_LABEL"] = [names[label] for label in result.labels]
    eligible["DISTANCE_TO_CENTROID"] = np.sqrt(
        np.sum((standardized - result.centers[result.labels]) ** 2, axis=1)
    )
    for index, output_name in enumerate(OUTPUT_FEATURES):
        eligible[output_name] = raw_x[:, index]
        eligible[f"Z_{output_name}"] = standardized[:, index]
    return eligible, result, names, means, standard_deviations


def build_source_profiles(
    paths: pd.DataFrame,
    common_profiles: pd.DataFrame,
    result: KMeansResult,
    cluster_names: list[str],
    means: np.ndarray,
    standard_deviations: np.ndarray,
    minimum_profile_coverage: float,
) -> pd.DataFrame:
    task_columns = ["OCC2010", *ONET_FEATURES]
    joined = paths.merge(
        common_profiles[task_columns], on="OCC2010", how="left", validate="many_to_one"
    )
    joined["HAS_TASK_PROFILE"] = joined[list(ONET_FEATURES)].notna().all(axis=1)
    joined["_covered_probability"] = (
        joined["PATH_PROBABILITY"] * joined["HAS_TASK_PROFILE"]
    )
    for feature in ONET_FEATURES:
        joined[f"_{feature}_expected"] = joined[feature] * joined["PATH_PROBABILITY"]

    rows = []
    for (scheme, source_code), group in joined.groupby(
        ["SOURCE_SCHEME", "SOURCE_OCC_3DIGIT"], sort=True
    ):
        modal = group.sort_values(
            ["PATH_PROBABILITY", "OCC2010"], ascending=[False, True]
        ).iloc[0]
        coverage = float(group["_covered_probability"].sum())
        probabilities = group["PATH_PROBABILITY"].to_numpy(dtype=float)
        positive = probabilities[probabilities > 0]
        row: dict[str, object] = {
            "SOURCE_SCHEME": scheme,
            "SOURCE_OCC_3DIGIT": int(source_code),
            "OCC2010_MODAL": int(modal["OCC2010"]),
            "OCC2010_MODAL_SHARE": float(modal["PATH_PROBABILITY"]),
            "OCC2010_TARGET_COUNT": int(group["OCC2010"].nunique()),
            "PROFILED_OCC2010_COUNT": int(
                group.loc[group["HAS_TASK_PROFILE"], "OCC2010"].nunique()
            ),
            "TASK_PROFILE_COVERAGE": coverage,
            "CROSSWALK_ENTROPY": float(-np.sum(positive * np.log(positive))),
        }
        for feature in ONET_FEATURES:
            numerator = float(group[f"_{feature}_expected"].sum())
            row[feature] = numerator / coverage if coverage > 0 else np.nan
        rows.append(row)
    source_profiles = pd.DataFrame(rows)

    eligible = source_profiles["TASK_PROFILE_COVERAGE"].ge(minimum_profile_coverage)
    raw_x = source_profiles.loc[eligible, list(ONET_FEATURES)].to_numpy(dtype=float)
    standardized = (raw_x - means) / standard_deviations
    distances = np.sum((standardized[:, None, :] - result.centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(distances, axis=1)
    source_profiles.loc[eligible, "TASK_CLUSTER_ID"] = labels + 1
    source_profiles.loc[eligible, "TASK_CLUSTER_LABEL"] = [
        cluster_names[label] for label in labels
    ]
    source_profiles.loc[eligible, "DISTANCE_TO_CENTROID"] = np.sqrt(
        distances[np.arange(len(labels)), labels]
    )
    for index, output_name in enumerate(OUTPUT_FEATURES):
        source_profiles.loc[eligible, output_name] = raw_x[:, index]
        source_profiles.loc[eligible, f"Z_{output_name}"] = standardized[:, index]
    source_profiles["TASK_CLUSTER_ID"] = source_profiles["TASK_CLUSTER_ID"].astype("Int64")
    return source_profiles.sort_values(
        ["SOURCE_SCHEME", "SOURCE_OCC_3DIGIT"]
    ).reset_index(drop=True)


def load_panel(path: Path) -> pd.DataFrame:
    columns = [
        "CASEID",
        "YEAR",
        "SAMPWEIGHT",
        "OCC_BASIS",
        "OCC_CODE",
        "OCCUPATION_TITLE",
        "SOC_MAJOR_GROUP",
    ]
    panel = pd.read_csv(
        path,
        usecols=columns,
        dtype={"OCC_BASIS": "string", "OCCUPATION_TITLE": "string", "SOC_MAJOR_GROUP": "string"},
        low_memory=False,
    )
    panel["SOURCE_SCHEME"] = pd.Series(pd.NA, index=panel.index, dtype="string")
    panel.loc[panel["OCC_BASIS"].eq("1970"), "SOURCE_SCHEME"] = "1970"
    panel.loc[
        panel["OCC_BASIS"].str.startswith("2000", na=False), "SOURCE_SCHEME"
    ] = "2000"
    source_code = pd.to_numeric(panel["OCC_CODE"], errors="coerce")
    four_digit = panel["OCC_BASIS"].eq("2000_4digit")
    source_code.loc[four_digit] = np.floor(source_code.loc[four_digit] / 10)
    panel["SOURCE_OCC_3DIGIT"] = source_code.astype("Int64")
    return panel


def attach_profiles(panel: pd.DataFrame, source_profiles: pd.DataFrame) -> pd.DataFrame:
    output = panel.merge(
        source_profiles,
        on=["SOURCE_SCHEME", "SOURCE_OCC_3DIGIT"],
        how="left",
        validate="many_to_one",
    )
    output["TASK_CLUSTER_ID"] = output["TASK_CLUSTER_ID"].astype("Int64")
    output["TASK_PROFILE_METHOD"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output.loc[
        output["TASK_CLUSTER_ID"].notna(), "TASK_PROFILE_METHOD"
    ] = "IPUMS_OCC2010_BRIDGE_TO_ONET_CENSUS2000"
    return output


def cluster_summary(
    common_profiles: pd.DataFrame,
    source_profiles: pd.DataFrame,
    person_year: pd.DataFrame,
    cluster_names: list[str],
    result: KMeansResult,
    means: np.ndarray,
    standard_deviations: np.ndarray,
) -> pd.DataFrame:
    matched = person_year["TASK_CLUSTER_ID"].notna()
    matched_weight = person_year.loc[matched, "SAMPWEIGHT"].sum()
    raw_centers = result.centers * standard_deviations + means
    rows = []
    for index, name in enumerate(cluster_names):
        cluster_id = index + 1
        common_mask = common_profiles["TASK_CLUSTER_ID"].eq(cluster_id)
        source_mask = source_profiles["TASK_CLUSTER_ID"].eq(cluster_id)
        panel_mask = person_year["TASK_CLUSTER_ID"].eq(cluster_id)
        row: dict[str, object] = {
            "TASK_CLUSTER_ID": cluster_id,
            "TASK_CLUSTER_LABEL": name,
            "COMMON_OCC2010_PROFILES": int(common_mask.sum()),
            "SOURCE_OCCUPATION_CODES": int(source_mask.sum()),
            "NLSY_PERSON_YEARS": int(panel_mask.sum()),
            "NLSY_PERSON_YEAR_SHARE": float(panel_mask.sum() / matched.sum()),
            "NLSY_WEIGHTED_PERSON_YEAR_SHARE": float(
                person_year.loc[panel_mask, "SAMPWEIGHT"].sum() / matched_weight
            ),
        }
        for feature_index, output_name in enumerate(OUTPUT_FEATURES):
            row[f"CENTROID_{output_name}"] = float(raw_centers[index, feature_index])
            row[f"Z_CENTROID_{output_name}"] = float(result.centers[index, feature_index])
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(person_year: pd.DataFrame) -> pd.DataFrame:
    rows = []
    basis_values = ["1970", "2000_3digit", "2000_4digit", "Missing"]
    for basis in basis_values:
        mask = person_year["OCC_BASIS"].isna() if basis == "Missing" else person_year["OCC_BASIS"].eq(basis)
        subset = person_year.loc[mask]
        with_occupation = subset["OCC_CODE"].notna()
        matched = subset["TASK_CLUSTER_ID"].notna()
        rows.append(
            {
                "OCC_BASIS": basis,
                "PANEL_ROWS": int(len(subset)),
                "ROWS_WITH_OCCUPATION": int(with_occupation.sum()),
                "ROWS_WITH_TASK_CLUSTER": int(matched.sum()),
                "MATCH_SHARE_OF_BASIS_ROWS": float(matched.mean()) if len(subset) else np.nan,
                "MATCH_SHARE_OF_ROWS_WITH_OCCUPATION": float(matched.sum() / with_occupation.sum())
                if with_occupation.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cluster_diagnostics(x: np.ndarray, seed: int) -> pd.DataFrame:
    rows = []
    for n_clusters in range(2, 9):
        result = fit_kmeans(x, n_clusters=n_clusters, seed=seed)
        rows.append(
            {
                "N_CLUSTERS": n_clusters,
                "INERTIA_STANDARDIZED_SPACE": result.inertia,
                "SILHOUETTE_STANDARDIZED_SPACE": silhouette_score(x, result.labels),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    metadata: dict[str, object],
    summary: pd.DataFrame,
    coverage: pd.DataFrame,
    diagnostics: pd.DataFrame,
    person_year: pd.DataFrame,
) -> None:
    rows = [
        "# Harmonized NLSY79 task-based occupation clustering",
        "",
        (
            "This model maps both 1970- and 2000-basis NLSY79 occupations to a common "
            "OCC2010 classification, attaches employment-weighted O*NET task profiles, "
            "and fits one occupation-level k-means model."
        ),
        "",
        "## Fit and coverage",
        "",
        f"- Common OCC2010 task profiles: {metadata['common_occ2010_profiles']:,}",
        f"- Silhouette score for k={metadata['n_clusters']}: {metadata['silhouette_standardized_space']:.3f}",
        f"- Highest silhouette for k=2–8: k={metadata['best_silhouette_k']}",
        f"- Matched person-years: {metadata['matched_person_years']:,} of {metadata['panel_rows']:,} "
        f"({metadata['matched_share_of_all_rows']:.1%})",
        f"- Match rate among rows with an occupation: {metadata['matched_share_of_rows_with_occupation']:.1%}",
        "",
        "| Occupation basis | Rows | With occupation | With task cluster | Match among occupied |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        share = "n.a." if pd.isna(row.MATCH_SHARE_OF_ROWS_WITH_OCCUPATION) else f"{row.MATCH_SHARE_OF_ROWS_WITH_OCCUPATION:.1%}"
        rows.append(
            f"| {row.OCC_BASIS} | {row.PANEL_ROWS:,} | {row.ROWS_WITH_OCCUPATION:,} | "
            f"{row.ROWS_WITH_TASK_CLUSTER:,} | {share} |"
        )
    rows.extend(
        [
            "",
            "## Cluster summary",
            "",
            "| ID | Description | OCC2010 profiles | Person-years | Matched share |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        rows.append(
            f"| {row.TASK_CLUSTER_ID} | {row.TASK_CLUSTER_LABEL} | "
            f"{row.COMMON_OCC2010_PROFILES:,} | {row.NLSY_PERSON_YEARS:,} | "
            f"{row.NLSY_PERSON_YEAR_SHARE:.1%} |"
        )
    rows.extend(
        [
            "",
            "## What the four groups mean",
            "",
            "The labels summarize each cluster's centroid. They are task profiles, not status "
            "rankings, and individual occupations can be close to more than one centroid.",
            "",
            "- **Physical and nonroutine manual:** above-average physical, spatial, vehicle, "
            "tool-handling, and dexterity demands, with below-average cognitive task scores.",
            "- **Physical and routine manual:** the strongest repetitive-motion, machine-control, "
            "equipment-paced, and physical task profile.",
            "- **Routine cognitive:** the highest repetitive, exact, and structured cognitive "
            "task score, with otherwise near-average or below-average task scores.",
            "- **Analytical and interpersonal:** the strongest data analysis, interpretation, "
            "creative thinking, relationship, directing, motivating, and coaching profile.",
            "",
            "### Standardized cluster centers",
            "",
            "Positive values are above the mean across the 448 common OCC2010 profiles; negative "
            "values are below it.",
            "",
            "| ID | Cluster | Nonroutine physical | Routine manual | Interpersonal | Routine cognitive | Analytical |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.itertuples(index=False):
        values = [
            getattr(row, f"Z_CENTROID_{feature}") for feature in OUTPUT_FEATURES
        ]
        rows.append(
            f"| {row.TASK_CLUSTER_ID} | {row.TASK_CLUSTER_LABEL} | "
            + " | ".join(f"{value:+.2f}" for value in values)
            + " |"
        )

    observed_titles = (
        person_year.dropna(subset=["TASK_CLUSTER_ID", "OCCUPATION_TITLE"])
        .groupby(
            ["TASK_CLUSTER_ID", "TASK_CLUSTER_LABEL", "OCCUPATION_TITLE"],
            as_index=False,
        )
        .size()
        .sort_values(
            ["TASK_CLUSTER_ID", "size", "OCCUPATION_TITLE"],
            ascending=[True, False, True],
        )
        .groupby("TASK_CLUSTER_ID", as_index=False)
        .head(5)
    )
    rows.extend(
        [
            "",
            "### Most common observed occupation titles",
            "",
            "These are the five most frequent NLSY79 titles within each matched cluster. They are "
            "examples from this sample, not definitions of the cluster.",
            "",
            "| ID | Cluster | Common observed titles |",
            "|---:|---|---|",
        ]
    )
    for cluster_id, group in observed_titles.groupby("TASK_CLUSTER_ID", sort=True):
        label = str(group["TASK_CLUSTER_LABEL"].iloc[0])
        examples = "; ".join(
            f"{title} ({count:,})"
            for title, count in zip(group["OCCUPATION_TITLE"], group["size"])
        )
        rows.append(f"| {int(cluster_id)} | {label} | {examples} |")

    diagnostic_rows = diagnostics.set_index("N_CLUSTERS")
    selected_k = int(metadata["n_clusters"])
    selected_silhouette = float(
        diagnostic_rows.loc[selected_k, "SILHOUETTE_STANDARDIZED_SPACE"]
    )
    best_k = int(metadata["best_silhouette_k"])
    best_silhouette = float(
        diagnostic_rows.loc[best_k, "SILHOUETTE_STANDARDIZED_SPACE"]
    )
    rows.extend(
        [
            "",
            "## Presentation guidance and limitations",
            "",
            f"The four-cluster solution is an interpretation choice: its silhouette score is "
            f"{selected_silhouette:.3f}, while k={best_k} has the highest k=2–8 silhouette "
            f"({best_silhouette:.3f}). Present the four groups as a useful economic summary, not "
            "as uniquely determined natural categories.",
            "",
            "The O*NET task taxonomy is held fixed across all survey years. This prevents the "
            "2002 coding change from mechanically changing task clusters, but it applies a "
            "post-2000 task description retrospectively to earlier occupations. The native "
            "occupation titles are retained using separate 1970, Census 2000 three-digit, and "
            "Census 2000 four-digit title namespaces. Military and unresolved occupation codes "
            "remain unmatched.",
        ]
    )
    output_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel", type=Path, default=project_root / "NLSY79_Integrated_EHR_v6.csv"
    )
    parser.add_argument("--ipums", type=Path, default=project_root / "usa_00002.csv")
    parser.add_argument(
        "--onet-tasks", type=Path, default=directory / "data" / "onet_task_occ2000.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=directory / "results_harmonized"
    )
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1979)
    parser.add_argument("--minimum-profile-coverage", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onet_tasks = load_onet_tasks(args.onet_tasks)
    bridge = load_ipums_bridge(args.ipums)
    all_common_profiles, paths_2000 = build_common_profiles(
        bridge, onet_tasks, args.minimum_profile_coverage
    )
    common_profiles, result, names, means, standard_deviations = fit_common_clusters(
        all_common_profiles, args.n_clusters, args.seed
    )

    paths = pd.concat([bridge_paths(bridge, 1970), paths_2000], ignore_index=True)
    source_profiles = build_source_profiles(
        paths,
        common_profiles,
        result,
        names,
        means,
        standard_deviations,
        args.minimum_profile_coverage,
    )
    panel = load_panel(args.panel)
    person_year = attach_profiles(panel, source_profiles)
    summary = cluster_summary(
        common_profiles,
        source_profiles,
        person_year,
        names,
        result,
        means,
        standard_deviations,
    )
    coverage = coverage_summary(person_year)
    standardized_common = (
        common_profiles[list(ONET_FEATURES)].to_numpy(dtype=float) - means
    ) / standard_deviations
    diagnostics = cluster_diagnostics(standardized_common, args.seed)

    unmatched = (
        person_year.loc[
            person_year["OCC_CODE"].notna() & person_year["TASK_CLUSTER_ID"].isna()
        ]
        .groupby(
            ["OCC_BASIS", "OCC_CODE", "OCCUPATION_TITLE", "SOC_MAJOR_GROUP"],
            dropna=False,
            sort=True,
        )
        .agg(
            NLSY_PERSON_YEARS=("CASEID", "size"),
            NLSY_UNIQUE_PEOPLE=("CASEID", "nunique"),
            FIRST_YEAR=("YEAR", "min"),
            LAST_YEAR=("YEAR", "max"),
        )
        .reset_index()
        .sort_values(["NLSY_PERSON_YEARS", "OCC_BASIS", "OCC_CODE"], ascending=[False, True, True])
    )

    rows_with_occupation = int(panel["OCC_CODE"].notna().sum())
    matched = int(person_year["TASK_CLUSTER_ID"].notna().sum())
    metadata: dict[str, object] = {
        "algorithm": "k-means++ with 50 deterministic initializations",
        "fit_unit": "common OCC2010 task profile; equal occupation weights",
        "task_taxonomy": "Acemoglu-Autor O*NET composites aggregated from Census 2000 occupations",
        "crosswalk_method": "IPUMS OCC2010 paths; probability-weighted expected task profiles",
        "n_clusters": args.n_clusters,
        "random_seed": args.seed,
        "minimum_profile_coverage": args.minimum_profile_coverage,
        "common_occ2010_profiles": int(len(common_profiles)),
        "source_occupation_profiles": int(source_profiles["TASK_CLUSTER_ID"].notna().sum()),
        "panel_rows": int(len(panel)),
        "rows_with_occupation": rows_with_occupation,
        "matched_person_years": matched,
        "matched_share_of_all_rows": float(matched / len(panel)),
        "matched_share_of_rows_with_occupation": float(matched / rows_with_occupation),
        "inertia_standardized_space": result.inertia,
        "silhouette_standardized_space": silhouette_score(
            standardized_common, result.labels
        ),
        "best_silhouette_k": int(
            diagnostics.loc[
                diagnostics["SILHOUETTE_STANDARDIZED_SPACE"].idxmax(), "N_CLUSTERS"
            ]
        ),
        "feature_means": dict(zip(OUTPUT_FEATURES, means.tolist())),
        "feature_standard_deviations": dict(
            zip(OUTPUT_FEATURES, standard_deviations.tolist())
        ),
        "panel_sha256": file_sha256(args.panel),
        "ipums_bridge_sha256": file_sha256(args.ipums),
        "onet_task_file_sha256": file_sha256(args.onet_tasks),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    common_profiles.to_csv(
        args.output_dir / "common_occ2010_task_profiles.csv", index=False, float_format="%.8g"
    )
    paths.to_csv(
        args.output_dir / "source_to_occ2010_probabilities.csv", index=False, float_format="%.8g"
    )
    source_profiles.to_csv(
        args.output_dir / "source_occupation_task_profiles.csv", index=False, float_format="%.8g"
    )
    person_year.to_csv(
        args.output_dir / "person_year_harmonized_task_clusters.csv", index=False, float_format="%.8g"
    )
    summary.to_csv(
        args.output_dir / "cluster_summary.csv", index=False, float_format="%.8g"
    )
    coverage.to_csv(
        args.output_dir / "coverage_by_occupation_basis.csv", index=False, float_format="%.8g"
    )
    diagnostics.to_csv(
        args.output_dir / "k_diagnostics.csv", index=False, float_format="%.8g"
    )
    unmatched.to_csv(
        args.output_dir / "unmatched_occupation_codes.csv", index=False, float_format="%.8g"
    )
    (args.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        args.output_dir / "model_report.md",
        metadata,
        summary,
        coverage,
        diagnostics,
        person_year,
    )

    print(
        f"Created {args.n_clusters} harmonized task clusters; matched {matched:,} of "
        f"{len(panel):,} person-year rows ({matched / len(panel):.1%})."
    )
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
