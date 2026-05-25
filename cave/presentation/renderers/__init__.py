__all__ = [
    "LayoutSpec",
    "MatplotlibRenderer",
    "FlattenedTopologyState",
    "TopologySurface",
    "TopologyTrajectory",
    "default_episode_set_embeddings",
    "episode_set_distance_payload",
    "flatten_topology_state",
    "save_episode_set_dashboard",
    "save_episode_set_distances_json",
    "save_topology_atlas",
    "save_topology_atlas_metrics",
    "save_topology_population_animation",
    "save_topology_population_dashboard",
    "save_topology_scatter_migration",
    "topology_atlas_metrics_payload",
    "topology_atlas_results",
    "save_topology_state_surface",
    "topology_trajectories",
    "topology_state_surface",
]


def __getattr__(name: str):
    if name in {"LayoutSpec", "MatplotlibRenderer"}:
        from cave.presentation.renderers.matplotlib_renderer import LayoutSpec, MatplotlibRenderer

        return {
            "LayoutSpec": LayoutSpec,
            "MatplotlibRenderer": MatplotlibRenderer,
        }[name]
    if name in {
        "default_episode_set_embeddings",
        "episode_set_distance_payload",
        "save_episode_set_dashboard",
        "save_episode_set_distances_json",
    }:
        from cave.presentation.renderers.episode_set_dashboard import (
            default_episode_set_embeddings,
            episode_set_distance_payload,
            save_episode_set_dashboard,
            save_episode_set_distances_json,
        )

        return {
            "default_episode_set_embeddings": default_episode_set_embeddings,
            "episode_set_distance_payload": episode_set_distance_payload,
            "save_episode_set_dashboard": save_episode_set_dashboard,
            "save_episode_set_distances_json": save_episode_set_distances_json,
        }[name]
    if name in {
        "save_topology_atlas",
        "save_topology_atlas_metrics",
        "topology_atlas_metrics_payload",
        "topology_atlas_results",
    }:
        from cave.presentation.renderers.topology_atlas_renderer import (
            save_topology_atlas,
            save_topology_atlas_metrics,
            topology_atlas_metrics_payload,
            topology_atlas_results,
        )

        return {
            "save_topology_atlas": save_topology_atlas,
            "save_topology_atlas_metrics": save_topology_atlas_metrics,
            "topology_atlas_metrics_payload": topology_atlas_metrics_payload,
            "topology_atlas_results": topology_atlas_results,
        }[name]
    if name in {
        "TopologySurface",
        "FlattenedTopologyState",
        "flatten_topology_state",
        "save_topology_state_surface",
        "topology_state_surface",
    }:
        from cave.presentation.renderers.topology_surface_renderer import (
            FlattenedTopologyState,
            TopologySurface,
            flatten_topology_state,
            save_topology_state_surface,
            topology_state_surface,
        )

        return {
            "TopologySurface": TopologySurface,
            "FlattenedTopologyState": FlattenedTopologyState,
            "flatten_topology_state": flatten_topology_state,
            "save_topology_state_surface": save_topology_state_surface,
            "topology_state_surface": topology_state_surface,
        }[name]
    if name in {
        "TopologyTrajectory",
        "save_topology_population_animation",
        "save_topology_population_dashboard",
        "save_topology_scatter_migration",
        "topology_trajectories",
    }:
        from cave.presentation.renderers.topology_population_renderer import (
            TopologyTrajectory,
            save_topology_population_animation,
            save_topology_population_dashboard,
            save_topology_scatter_migration,
            topology_trajectories,
        )

        return {
            "TopologyTrajectory": TopologyTrajectory,
            "save_topology_population_animation": save_topology_population_animation,
            "save_topology_population_dashboard": save_topology_population_dashboard,
            "save_topology_scatter_migration": save_topology_scatter_migration,
            "topology_trajectories": topology_trajectories,
        }[name]
    raise AttributeError(name)
