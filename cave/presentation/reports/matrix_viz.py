from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from cave.presentation.reports.specs import MatrixRunRecord
from cave.demonstrations.subjects.dashboard import classical_mds


def save_population_plot(
    records: list[MatrixRunRecord],
    distance_jsons: tuple[Path, ...],
    output: str | Path,
) -> Path:
    output = Path(output)
    if not distance_jsons:
        raise ValueError("at least one distance json is required")
    figure, axes = plt.subplots(
        1,
        len(distance_jsons),
        figsize=(6.2 * len(distance_jsons), 5.4),
        squeeze=False,
    )
    condition_ids = sorted(
        {
            record.factor_id("condition", record.variant_id) or record.variant_id
            for record in records
        }
    )
    start_condition_ids = sorted(
        {
            record.factor_id("start_condition", record.subject_id) or record.subject_id
            for record in records
        }
    )
    color_map = {
        condition_id: plt.get_cmap("tab10")(index % 10)
        for index, condition_id in enumerate(condition_ids)
    }
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_map = {
        start_id: marker_cycle[index % len(marker_cycle)]
        for index, start_id in enumerate(start_condition_ids)
    }

    for axis, distance_path in zip(axes[0], distance_jsons):
        payload = json.loads(distance_path.read_text(encoding="utf-8"))
        distances = np.array(payload["distances"], dtype=float)
        coords = classical_mds(distances)
        for index, record in enumerate(records):
            axis.scatter(
                coords[index, 0],
                coords[index, 1],
                color=color_map[
                    record.factor_id("condition", record.variant_id) or record.variant_id
                ],
                marker=marker_map[
                    record.factor_id("start_condition", record.subject_id) or record.subject_id
                ],
                s=58,
                alpha=0.86,
                edgecolor="#111827",
                linewidth=0.35,
            )
        axis.axhline(0.0, color="#d1d5db", linewidth=0.8)
        axis.axvline(0.0, color="#d1d5db", linewidth=0.8)
        axis.set_title(_title_for_embedding(payload["embedding"]), loc="left", fontweight="bold")
        axis.set_xlabel("MDS 1")
        axis.set_ylabel("MDS 2")

    variant_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=condition_id,
            markerfacecolor=color,
            markeredgecolor="#111827",
            markersize=8,
        )
        for condition_id, color in color_map.items()
    ]
    subject_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="#111827",
            label=start_id,
            linestyle="None",
            markersize=8,
        )
        for start_id, marker in marker_map.items()
    ]
    figure.legend(
        handles=variant_handles + subject_handles,
        loc="lower center",
        ncol=min(6, len(variant_handles) + len(subject_handles)),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(output, dpi=150)
    plt.close(figure)
    return output


def write_cluster_summary(
    cluster_jsons: tuple[Path, ...],
    output: str | Path,
) -> Path:
    output = Path(output)
    summaries = []
    for cluster_path in cluster_jsons:
        payload = json.loads(cluster_path.read_text(encoding="utf-8"))
        clusters = payload["clusters"]
        largest = max(clusters, key=lambda item: item["size"]) if clusters else None
        summaries.append(
            {
                "embedding": payload["embedding"],
                "threshold": payload["threshold"],
                "cluster_count": len(clusters),
                "largest_cluster_size": 0 if largest is None else largest["size"],
                "largest_cluster_variant_counts": (
                    {}
                    if largest is None
                    else dict(Counter(largest["variant_ids"]))
                ),
                "largest_cluster_subject_counts": (
                    {}
                    if largest is None
                    else dict(Counter(largest["subject_ids"]))
                ),
                "largest_cluster_sequence_counts": (
                    {}
                    if largest is None
                    else dict(Counter(largest["sequence_ids"]))
                ),
                "largest_cluster_condition_counts": (
                    {}
                    if largest is None
                    else dict(Counter(largest.get("factor_ids", {}).get("condition", [])))
                ),
                "largest_cluster_start_condition_counts": (
                    {}
                    if largest is None
                    else dict(
                        Counter(largest.get("factor_ids", {}).get("start_condition", []))
                    )
                ),
                "largest_cluster_treatment_counts": (
                    {}
                    if largest is None
                    else dict(Counter(largest.get("factor_ids", {}).get("treatment", [])))
                ),
            }
        )
    output.write_text(
        json.dumps({"embeddings": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _title_for_embedding(name: str) -> str:
    return name.replace("_", " ").title()
