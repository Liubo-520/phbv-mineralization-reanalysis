"""Precision-preserving reconstruction of the PHBV workbook.

Rows are paired by workbook row and measurement-column position. This avoids
the global stack-and-concatenate assumption used by the public preprocessing
notebook and retains source coordinates for every observation.
"""

from __future__ import annotations

import re
import hashlib
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
NA_TOKENS = {"", "na", "n/a", "n.a.", "ν/α", "nan", "none", "-"}

MECHANISM_MAP = {
    "enzymemediated_microbial_biodegradation": "enzyme_mediated_microbial_biodegradation",
    "hydrolysisassisted_microbial_assimilation": "hydrolysis_assisted_microbial_assimilation",
    "surface_erosiondriven_biodegradation": "surface_erosion_driven_biodegradation",
    "microbial_biodegradation": "microbial_biodegradation",
}
MORPHOLOGY_MAP = {
    "films": "films",
    "film": "films",
    "circular_blend_films": "films",
    "square_films": "films",
    "sheets": "sheets",
    "rectangular_strips_flat_sheets": "sheets",
    "particle": "particles",
    "continuous_phbv_matrix_with_dispersed_filler_particles": "matrix_composites",
    "matrix": "matrix_composites",
    "filaments_pellets": "pellets",
}
MICROBE_MAP = {
    "bacteria_and_fungi": "bacteria_dominated",
    "bacteria": "bacteria_dominated",
    "the_phas_degraders_were_identified_from_genus_cupriavidus_acidovorax_variovora_streptomyces_and_ideonella_and_two_starch_degraders_were_identified_from_genus_moraxella_and_alcaligenes": "bacteria_dominated",
    "marine_bacteria_and_fungi": "bacteria_dominated",
    "thermophilic_bacteria_fungi": "bacteria_dominated",
    "thermophilic_bacteria_and_fungi": "bacteria_dominated",
    "fungi": "fungi_dominated",
    "fungi_yeasts_and_bacteria": "bacteria_fungi_mixed",
    "bacteria_fungi_actinomycetes": "bacteria_fungi_mixed",
    "mixed_microbial_community": "diverse_environmental",
    "bacteria_and_possibly_fungi_or_algae": "diverse_environmental",
}
ENVIRONMENT_MAP = {
    "soil": "soil",
    "marine": "marine_aquatic",
    "freshwater": "marine_aquatic",
    "industrial_compost": "compost_organic_fertilizer",
    "compost": "compost_organic_fertilizer",
    "thermophilic_compost": "compost_organic_fertilizer",
    "liquid_mineral_medium": "laboratory_mineral_media",
}
ADDITIVE_MAP = {
    "filler": "filler",
    "plasticizer": "plasticizer",
    "not_applicable": "not_applicable",
    "nucleating_agent": "functional_additive",
    "crosslinker": "functional_additive",
}

# Primary-source endpoint audit: studies 5 and 6 report mass-loss/soil-burial
# responses rather than CO2-derived mineralization. See references/source_endpoint_audit.csv.
ENDPOINT_AUDITED_MINERALIZATION_STUDIES = {1, 2, 3, 4, 7, 15, 16, 20, 22, 23, 24}


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = text.strip().lower().replace("%", "").replace("(", "").replace(")", "")
    text = text.replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().lower().replace("%", "").replace("(", "").replace(")", "")
    text = text.replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return re.sub(r"_+", "_", text).strip("_")


def parse_number(value: Any) -> tuple[float, str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan, "missing"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), "numeric"
    text = str(value).strip()
    if text.casefold() in NA_TOKENS:
        return np.nan, "explicit_na"
    match = NUMBER_RE.search(text)
    if not match:
        return np.nan, "unparsed"
    status = "central_value" if ("±" in text or "+/-" in text or "(" in text) else "numeric_text"
    return float(match.group()), status


def _read_sheet(path: Path, name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=name)


def pair_wide_measurements(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = _read_sheet(path, "Time_point_features")
    response = _read_sheet(path, "Biodegradation_features")
    if len(time) != len(response) or time.shape[1] != response.shape[1]:
        raise ValueError("Time and response worksheets have incompatible shapes")

    observations: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for row_idx in range(len(time)):
        t_key = tuple(time.iloc[row_idx, :2])
        y_key = tuple(response.iloc[row_idx, :2])
        if t_key != y_key:
            raise ValueError(f"Study/instance mismatch at Excel row {row_idx + 2}: {t_key} != {y_key}")
        study_id, instance = map(int, t_key)
        time_name = str(time.iat[row_idx, 2]).strip()
        response_name = str(response.iat[row_idx, 2]).strip()
        curve_id = f"S{study_id:02d}_I{instance:03d}"
        for column_idx in range(3, time.shape[1]):
            time_raw = time.iat[row_idx, column_idx]
            response_raw = response.iat[row_idx, column_idx]
            if pd.isna(time_raw) and pd.isna(response_raw):
                continue
            time_value, time_status = parse_number(time_raw)
            response_value, response_status = parse_number(response_raw)
            paired = pd.notna(time_value) and pd.notna(response_value)
            entry = {
                "study_id": study_id,
                "instance": instance,
                "curve_id": curve_id,
                "source_excel_row": row_idx + 2,
                "source_excel_column": column_idx + 1,
                "point_index": column_idx - 2,
                "sample_name_time_raw": time_name,
                "sample_name_response_raw": response_name,
                "sample_name_match": time_name.casefold() == response_name.casefold(),
                "biodegradation_time_raw": time_raw,
                "biodegradation_percentage_raw": response_raw,
                "time_parse_status": time_status,
                "response_parse_status": response_status,
                "pair_status": "paired" if paired else "not_modelable",
            }
            audit.append(entry)
            if paired:
                observations.append(
                    entry
                    | {
                        "biodegradation_time_days": time_value,
                        "biodegradation_percentage": response_value,
                    }
                )
    paired = pd.DataFrame(observations)
    pairing_audit = pd.DataFrame(audit)
    if paired.duplicated(["curve_id", "point_index"]).any():
        raise ValueError("Duplicate curve/point coordinates detected")
    return paired, pairing_audit


def build_curve_metadata(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    material = _read_sheet(path, "Materials_features")
    environment = _read_sheet(path, "Environmental_features")
    material.columns = [normalize_name(c) for c in material.columns]
    environment.columns = [normalize_name(c) for c in environment.columns]

    duplicates = material[material.duplicated(["study_id", "instance"], keep=False)].copy()
    material = material.drop_duplicates(["study_id", "instance"], keep="first")
    if environment.duplicated(["study_id", "instance"]).any():
        raise ValueError("Environmental metadata are not unique by study/instance")

    m_name = material.pop("sample_name").rename("sample_name_material_raw")
    e_name = environment.pop("sample_name").rename("sample_name_environment_raw")
    material.insert(2, "sample_name_material_raw", m_name)
    environment.insert(2, "sample_name_environment_raw", e_name)
    metadata = material.merge(environment, on=["study_id", "instance"], how="left", suffixes=("_material", "_environment"))
    metadata["study_id"] = metadata["study_id"].astype(int)
    metadata["instance"] = metadata["instance"].astype(int)
    metadata["curve_id"] = metadata.apply(lambda r: f"S{r.study_id:02d}_I{r.instance:03d}", axis=1)

    for col in metadata.select_dtypes(include="object"):
        if not col.startswith("sample_name_") and col != "curve_id":
            metadata[col] = metadata[col].map(normalize_value)

    for col in ["additive1_percentage_wt", "additive2_percentage_wt", "additive3_percentage_wt"]:
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce").fillna(0.0)

    metadata["degradation_mechanism"] = metadata["degradation_mechanism"].map(MECHANISM_MAP).fillna("uncategorized")
    metadata["sample_shape_morphology"] = metadata["sample_shape_morphology"].map(MORPHOLOGY_MAP).fillna("uncategorized")
    metadata["pha_degrading_microbes"] = metadata["pha_degrading_microbes"].map(MICROBE_MAP).fillna("uncategorized")
    metadata["degradation_environment"] = metadata["degradation_environment"].map(ENVIRONMENT_MAP).fillna("other")
    for col in ["additive_type_1", "additive_type_2", "additive_type_3"]:
        pct = col.replace("type_", "").replace("additive", "additive") + "_percentage_wt"
        index = col.split("_")[-1]
        pct = f"additive{index}_percentage_wt"
        metadata.loc[metadata[pct].eq(0), col] = "not_applicable"
        if col == "additive_type_1":
            metadata[col] = metadata[col].replace("_filler", "filler").map(ADDITIVE_MAP).fillna("uncategorized")
        else:
            metadata[col] = metadata[col].fillna("not_applicable")

    metadata["t_biodeg"] = pd.to_numeric(metadata["t_biodeg"], errors="coerce")
    selected = [
        "study_id", "instance", "curve_id", "sample_name_material_raw", "sample_name_environment_raw",
        "adjusted_hb_ratio_formulation_mol", "adjusted_hv_ratio_formulation_mol",
        "biodegradation_condition", "degradation_mechanism", "additives", "t_biodeg",
        "degradation_environment", "additive_type_1", "additive1_percentage_wt",
        "additive_type_2", "additive2_percentage_wt", "additive_type_3",
        "additive3_percentage_wt", "sample_shape_morphology", "pha_degrading_microbes",
    ]
    return metadata, duplicates


def _version_summary(name: str, data: pd.DataFrame, rule: str) -> dict[str, Any]:
    return {
        "dataset": name,
        "rows": len(data),
        "studies": data["study_id"].nunique(),
        "curves": data["curve_id"].nunique(),
        "min_points_per_curve": data.groupby("curve_id").size().min(),
        "median_points_per_curve": data.groupby("curve_id").size().median(),
        "max_points_per_curve": data.groupby("curve_id").size().max(),
        "fractional_time_rows": float(data["biodegradation_time_days"].mod(1).ne(0).mean()),
        "fractional_response_rows": float(data["biodegradation_percentage"].mod(1).ne(0).mean()),
        "rows_gt100": int(data["biodegradation_percentage"].gt(100).sum()),
        "rule": rule,
    }


def reconstruct(workbook: Path, published_csv: Path, processed_dir: Path, audit_dir: Path) -> dict[str, pd.DataFrame]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    paired, pairing_audit = pair_wide_measurements(workbook)
    metadata, material_duplicates = build_curve_metadata(workbook)

    analysis_full = paired.merge(metadata, on=["study_id", "instance", "curve_id"], how="left", validate="many_to_one")
    if analysis_full["sample_name_material_raw"].isna().any():
        raise ValueError("At least one parsed curve has no material metadata")
    feature_columns = [
        "adjusted_hb_ratio_formulation_mol", "adjusted_hv_ratio_formulation_mol",
        "biodegradation_condition", "degradation_mechanism", "additives", "t_biodeg",
        "degradation_environment", "additive_type_1", "additive1_percentage_wt",
        "additive_type_2", "additive2_percentage_wt", "additive_type_3",
        "additive3_percentage_wt", "sample_shape_morphology", "pha_degrading_microbes",
    ]
    curve_rows = analysis_full.drop_duplicates("curve_id")
    signatures = curve_rows[["study_id"] + feature_columns].fillna("<missing>").astype(str).agg("|".join, axis=1)
    family_map = {
        curve_id: "F" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]
        for curve_id, signature in zip(curve_rows["curve_id"], signatures)
    }
    analysis_full["curve_family_id"] = analysis_full["curve_id"].map(family_map)
    trace_front = [
        "biodegradation_time_days", "biodegradation_percentage", "study_id", "instance", "curve_id",
        "curve_family_id", "point_index", "source_excel_row", "source_excel_column", "sample_name_time_raw",
        "sample_name_response_raw", "sample_name_material_raw", "sample_name_environment_raw",
    ]
    analysis = analysis_full[trace_front + feature_columns].copy()

    raw_front = trace_front + [
        "biodegradation_time_raw", "biodegradation_percentage_raw",
        "time_parse_status", "response_parse_status", "sample_name_match", "pair_status",
    ]
    d_raw = analysis_full[raw_front + [c for c in analysis_full.columns if c not in raw_front]].copy()
    d1 = analysis.loc[analysis["biodegradation_percentage"].le(100)].copy()
    d2 = analysis.copy()
    d2["biodegradation_percentage"] = d2["biodegradation_percentage"].clip(upper=100)
    d3 = d1.loc[d1["study_id"].isin(ENDPOINT_AUDITED_MINERALIZATION_STUDIES)].copy()

    published = pd.read_csv(published_csv)
    published["curve_id"] = published.apply(lambda r: f"S{int(r.study_id):02d}_I{int(r.instance):03d}", axis=1)
    d0_path = processed_dir / "D0_published_cleaned.csv"
    shutil.copyfile(published_csv, d0_path)
    d_raw.to_csv(processed_dir / "D_raw_unbounded.csv", index=False)
    d1.to_csv(processed_dir / "D1_precision_preserved_exclude_gt100.csv", index=False)
    d2.to_csv(processed_dir / "D2_precision_preserved_cap_gt100.csv", index=False)
    d3.to_csv(processed_dir / "D3_endpoint_audited_mineralization.csv", index=False)
    metadata.to_csv(processed_dir / "curve_metadata.csv", index=False)

    pairing_audit.to_csv(audit_dir / "time_percentage_pairing_audit.csv", index=False)
    material_duplicates.to_csv(audit_dir / "material_duplicate_keys.csv", index=False)

    comparison = pd.DataFrame([
        _version_summary("D0_published_cleaned", published, "Public notebook: truncate decimals, then exclude truncated B% > 100"),
        _version_summary("D1_precision_preserved_exclude_gt100", d1, "Preserve precision and exclude original B% > 100"),
        _version_summary("D2_precision_preserved_cap_gt100", d2, "Preserve precision and cap original B% > 100 at 100"),
        _version_summary("D3_endpoint_audited_mineralization", d3, "D1 restricted to primary-source-confirmed CO2/respirometric mineralization studies"),
        _version_summary("D_raw_unbounded", d_raw, "Preserve all paired numeric observations"),
    ])
    comparison.to_csv(audit_dir / "dataset_version_comparison.csv", index=False)

    public_like = d_raw.copy()
    public_like["public_time"] = np.trunc(public_like["biodegradation_time_days"])
    public_like["public_response"] = np.trunc(public_like["biodegradation_percentage"])
    public_like = public_like.loc[public_like["public_response"].le(100)]
    examples = public_like.loc[
        public_like["biodegradation_time_days"].ne(public_like["public_time"])
        | public_like["biodegradation_percentage"].ne(public_like["public_response"]),
        [
            "study_id", "instance", "curve_id", "point_index", "biodegradation_time_days",
            "public_time", "biodegradation_percentage", "public_response",
        ],
    ].head(100)
    examples.to_csv(audit_dir / "value_parsing_examples.csv", index=False)

    dictionary = pd.DataFrame({
        "column": d1.columns,
        "dtype": [str(d1[c].dtype) for c in d1.columns],
        "missing_count_D1": [int(d1[c].isna().sum()) for c in d1.columns],
        "role": [
            "target" if c == "biodegradation_percentage" else
            "time_feature" if c == "biodegradation_time_days" else
            "group_or_trace" if c in {"study_id", "instance", "curve_id"} or c.startswith("source_") or c.startswith("sample_name_") or c == "point_index" else
            "candidate_feature"
            for c in d1.columns
        ],
    })
    with pd.ExcelWriter(processed_dir / "data_dictionary.xlsx", engine="openpyxl") as writer:
        dictionary.to_excel(writer, sheet_name="data_dictionary", index=False)
        comparison.to_excel(writer, sheet_name="dataset_versions", index=False)
        metadata.to_excel(writer, sheet_name="curve_metadata", index=False)

    return {"D0": published, "D1": d1, "D2": d2, "D3": d3, "D_raw": d_raw, "comparison": comparison}
