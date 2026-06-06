from __future__ import annotations

from cave.observation.compression import summarize_episode_compression
from cave.pressure.checks.compression_clamp import (
    build_compression_clamp_episode,
    check_compression_clamp,
)
from cave.pressure.checks.primitive_compression import (
    build_primitive_compression_episode,
    check_primitive_compression,
)


def test_primitive_compression_pressure_distinguishes_paid_work() -> None:
    result = check_primitive_compression()

    assert result["ok"]
    metrics = result["metrics"]
    assert metrics["ratio-5-active"]["compression_ratio"] == 5.0
    assert metrics["ratio-1-active"]["compression_ratio"] == 1.0
    assert metrics["ratio-5-active"]["update_work"] > 0.0
    assert metrics["ratio-5-container"]["ownership_subject_fraction"] == 0.0
    assert (
        metrics["ratio-5-active"]["paid_compression_proxy"]
        > metrics["ratio-5-container"]["paid_compression_proxy"]
    )
    assert (
        metrics["ratio-5-random"]["mean_distortion"]
        > metrics["ratio-5-active"]["mean_distortion"]
    )


def test_compression_summary_reads_episode_metadata() -> None:
    episode = build_primitive_compression_episode("ratio-5-active")
    summary = summarize_episode_compression(episode)

    assert summary["pressure"]["compression_ratio"] == 5.0
    assert summary["work"]["ownership_subject_fraction"] == 1.0
    assert summary["effect"]["loss_to_update_coupling"] > 0.0
    assert summary["summary"]["paid_compression_proxy"] > 0.0

def test_compression_clamp_distinguishes_adaptive_governance() -> None:
    result = check_compression_clamp()

    assert result["ok"]
    metrics = result["metrics"]
    assert metrics["active"]["mean_selectivity"] > metrics["random-compressor"]["mean_selectivity"]
    assert (
        metrics["active"]["lag_loss_to_update_coupling"]
        > metrics["shuffled-loss"]["lag_loss_to_update_coupling"]
    )
    assert metrics["active"]["mean_action_success"] > metrics["no-update"]["mean_action_success"]
    assert (
        metrics["active"]["adaptive_governance_proxy"]
        > metrics["oracle-rails"]["adaptive_governance_proxy"]
    )


def test_compression_clamp_episode_records_capacity_and_selection() -> None:
    episode = build_compression_clamp_episode("active")
    summary = summarize_episode_compression(episode)

    assert summary["pressure"]["mean_compression_ratio"] > 1.0
    assert summary["work"]["ownership_subject_fraction"] == 1.0
    selected_counts = [
        len(obs.metadata["compression_clamp"]["selected_features"])
        for obs in episode.observations
    ]
    capacities = [
        obs.metadata["compression_clamp"]["capacity"]
        for obs in episode.observations
    ]
    assert selected_counts == capacities
