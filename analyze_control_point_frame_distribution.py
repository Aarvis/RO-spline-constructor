"""Analyze reached-state spline frame coverage using unique knot pairs.

For a requested value N, every unique-knot pair (i, i + N + 1) is analyzed;
there are therefore exactly N unique knots strictly between the endpoints.

This program maps each unique knot value through the stored
``frame_to_u``/``frame_indices`` arrays so the knot locations can be expressed
in dataset-frame units. It also reconstructs the saved spline at each frame and
reports absolute-error distributions for position, velocity, acceleration, and
jerk over the full dataset, per episode, per dimension, and per
episode-dimension pair.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import BSpline
from tqdm.auto import tqdm


DEFAULT_CONFIG = Path(__file__).with_name("config_control_point_frame_distribution.yaml")
SIGNAL_ORDERS: dict[str, int] = {
    "position": 0,
    "velocity": 1,
    "acceleration": 2,
    "jerk": 3,
}


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    source_dataset_root: Path
    episodes_subdir: str
    arrays_subdir: str
    max_episodes: int | None
    spline_npz_name: str
    state_array_name: str
    timestamps_array_name: str
    fallback_fps: float
    use_timestamps_if_available: bool
    allowed_target_modes: tuple[str, ...]
    intermediate_counts: tuple[int, ...]
    rounded_frame_method: str
    percentiles: tuple[float, ...]
    absolute_error_percentiles: tuple[float, ...]
    histogram_metric: str
    output_dir: Path
    overwrite: bool
    episode_ratios_csv: str
    ratio_distribution_csv: str
    pair_details_parquet: str
    pair_distribution_csv: str
    pair_histogram_csv: str
    absolute_error_distribution_csv: str
    signal_error_distribution_csv: str
    signal_error_per_episode_csv: str
    signal_error_per_dimension_csv: str
    signal_error_per_episode_dimension_csv: str
    summary_json: str
    resolved_config_yaml: str
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    config_path = args.config.expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    input_cfg = raw.get("input", {})
    analysis = raw.get("analysis", {})
    output = raw.get("output", {})

    dataset_root = Path(args.dataset_root or raw["dataset_root"]).expanduser().resolve()
    source_dataset_root = Path(raw.get("source_dataset_root", dataset_root)).expanduser().resolve()
    max_episodes = args.max_episodes if args.max_episodes is not None else raw.get("max_episodes")
    counts = tuple(sorted({int(value) for value in analysis["intermediate_control_point_counts"]}))
    if not counts or counts[0] < 0:
        raise ValueError("intermediate_control_point_counts must contain non-negative integers")

    rounded_method = str(analysis.get("rounded_frame_method", "nearest"))
    if rounded_method not in {"nearest", "floor", "ceil"}:
        raise ValueError("rounded_frame_method must be nearest, floor, or ceil")

    run_name = str(output.get("run_name", "reached_state_knot_frame_distribution"))
    output_dir = dataset_root / "metadata" / "reached_state_episode_spline_constructor" / run_name
    return Config(
        dataset_root=dataset_root,
        source_dataset_root=source_dataset_root,
        episodes_subdir=str(raw.get("episodes_subdir", "episodes")),
        arrays_subdir=str(raw.get("arrays_subdir", "arrays")),
        max_episodes=int(max_episodes) if max_episodes is not None else None,
        spline_npz_name=str(input_cfg.get("spline_npz_name", "global_reached_state_spline.npz")),
        state_array_name=str(input_cfg.get("state_array_name", "state_65d.npy")),
        timestamps_array_name=str(input_cfg.get("timestamps_array_name", "timestamps.npy")),
        fallback_fps=float(input_cfg.get("fallback_fps", 30.0)),
        use_timestamps_if_available=bool(input_cfg.get("use_timestamps_if_available", True)),
        allowed_target_modes=tuple(str(x) for x in input_cfg.get("allowed_target_modes", ["global_reached_state"])),
        intermediate_counts=counts,
        rounded_frame_method=rounded_method,
        percentiles=tuple(float(x) for x in analysis.get("percentiles", [0, 1, 2, 5, 10, 15, 25, 50, 75, 85, 95, 99, 99.9, 100])),
        absolute_error_percentiles=tuple(
            float(x)
            for x in analysis.get(
                "absolute_error_percentiles",
                [0, 1, 2, 5, 10, 15, 25, 50, 75, 85, 95, 99, 99.9, 99.99, 99.999, 100],
            )
        ),
        histogram_metric=str(analysis.get("histogram_metric", "frames_inclusive")),
        output_dir=output_dir,
        overwrite=bool(args.overwrite or output.get("overwrite", False)),
        episode_ratios_csv=str(output.get("episode_ratios_csv", "episode_knot_frame_ratios.csv")),
        ratio_distribution_csv=str(output.get("ratio_distribution_csv", "knot_frame_ratio_distribution.csv")),
        pair_details_parquet=str(output.get("pair_details_parquet", "knot_pair_frame_counts.parquet")),
        pair_distribution_csv=str(output.get("pair_distribution_csv", "knot_pair_frame_distribution.csv")),
        pair_histogram_csv=str(output.get("pair_histogram_csv", "knot_pair_frame_histogram.csv")),
        absolute_error_distribution_csv=str(output.get("absolute_error_distribution_csv", "absolute_error_distribution.csv")),
        signal_error_distribution_csv=str(output.get("signal_error_distribution_csv", "signal_absolute_error_distribution.csv")),
        signal_error_per_episode_csv=str(output.get("signal_error_per_episode_csv", "signal_absolute_error_per_episode.csv")),
        signal_error_per_dimension_csv=str(output.get("signal_error_per_dimension_csv", "signal_absolute_error_per_dimension.csv")),
        signal_error_per_episode_dimension_csv=str(
            output.get("signal_error_per_episode_dimension_csv", "signal_absolute_error_per_episode_dimension.csv")
        ),
        summary_json=str(output.get("summary_json", "summary.json")),
        resolved_config_yaml=str(output.get("resolved_config_yaml", "resolved_config.yaml")),
        raw=raw,
    )


def discover_episode_splines(cfg: Config) -> list[tuple[str, Path]]:
    episodes_root = cfg.dataset_root / cfg.episodes_subdir
    if not episodes_root.is_dir():
        raise FileNotFoundError(f"Episode directory does not exist: {episodes_root}")
    episode_dirs = sorted(path for path in episodes_root.iterdir() if path.is_dir())
    if cfg.max_episodes is not None:
        episode_dirs = episode_dirs[: cfg.max_episodes]
    missing = [episode.name for episode in episode_dirs if not (episode / cfg.arrays_subdir / cfg.spline_npz_name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} episode(s) lack the requested reached-state spline "
            f"{cfg.spline_npz_name!r}; first missing: {preview}"
        )
    return [(episode.name, episode / cfg.arrays_subdir / cfg.spline_npz_name) for episode in episode_dirs]


def scalar(array: np.ndarray, name: str) -> int:
    flat = np.asarray(array).reshape(-1)
    if flat.size != 1:
        raise ValueError(f"{name} must contain exactly one value; got shape {array.shape}")
    return int(flat[0])


def scalar_float(array: np.ndarray, name: str) -> float:
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    if flat.size != 1:
        raise ValueError(f"{name} must contain exactly one value; got shape {array.shape}")
    return float(flat[0])


def round_frame_positions(values: np.ndarray, method: str) -> np.ndarray:
    if method == "nearest":
        return np.floor(values + 0.5).astype(np.int64)
    if method == "floor":
        return np.floor(values).astype(np.int64)
    return np.ceil(values).astype(np.int64)


def infer_dt_seconds(source_arrays_dir: Path, num_frames: int, cfg: Config) -> tuple[float, float, str]:
    timestamps_path = source_arrays_dir / cfg.timestamps_array_name
    if cfg.use_timestamps_if_available and timestamps_path.is_file():
        timestamps = np.load(timestamps_path)
        if timestamps.ndim == 1 and timestamps.shape[0] == num_frames:
            diffs = np.diff(timestamps.astype(np.float64))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
            if diffs.size > 0:
                dt_seconds = float(np.median(diffs))
                return dt_seconds, 1.0 / dt_seconds, "timestamps"

    fps = float(cfg.fallback_fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fallback_fps must be positive; got {fps!r}")
    return 1.0 / fps, fps, "fallback_fps"


def load_knot_frames(
    episode_uid: str, path: Path, cfg: Config
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    with np.load(path, allow_pickle=False) as data:
        required = {"global_knots", "global_degree", "frame_indices", "frame_to_u"}
        absent = sorted(required.difference(data.files))
        if absent:
            raise KeyError(f"{episode_uid}: {path.name} lacks keys {absent}")

        mode = str(np.asarray(data["spline_target_mode"]).reshape(-1)[0]) if "spline_target_mode" in data else ""
        if mode not in cfg.allowed_target_modes:
            raise ValueError(
                f"{episode_uid}: spline_target_mode={mode!r} is not an allowed reached-state mode "
                f"{cfg.allowed_target_modes}"
            )

        knots = np.asarray(data["global_knots"], dtype=np.float64)
        degree = scalar(data["global_degree"], "global_degree")
        frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        frame_to_u = np.asarray(data["frame_to_u"], dtype=np.float64)
        num_frames = scalar(data["num_original_frames"], "num_original_frames") if "num_original_frames" in data else len(frame_indices)

    unique_knots = np.unique(knots.astype(np.float64))
    if len(frame_indices) != num_frames or len(frame_to_u) != num_frames:
        raise ValueError(
            f"{episode_uid}: frame mapping length mismatch: frames={num_frames}, "
            f"frame_indices={len(frame_indices)}, frame_to_u={len(frame_to_u)}"
        )
    if num_frames < 2 or np.any(np.diff(frame_to_u) < 0):
        raise ValueError(f"{episode_uid}: frame_to_u must be monotonic and contain at least two frames")
    if unique_knots.size < 2:
        raise ValueError(f"{episode_uid}: unique knot count must be at least 2")

    continuous_frames = np.interp(unique_knots, frame_to_u, frame_indices.astype(np.float64))
    rounded_frames = round_frame_positions(continuous_frames, cfg.rounded_frame_method)
    rounded_frames = np.clip(rounded_frames, int(frame_indices.min()), int(frame_indices.max()))
    return unique_knots, continuous_frames, rounded_frames, num_frames, degree, mode


def compute_derivatives(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_order = 2 if signal.shape[0] >= 3 else 1
    velocity = np.gradient(signal, dt, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=edge_order)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=edge_order)
    return (
        velocity.astype(np.float64, copy=False),
        acceleration.astype(np.float64, copy=False),
        jerk.astype(np.float64, copy=False),
    )


def derivative_or_zero(spline: BSpline, order: int, u: np.ndarray, duration: float) -> np.ndarray:
    if order > int(spline.k):
        return np.zeros((len(u), spline.c.shape[1]), dtype=np.float64)
    return spline.derivative(order)(u).astype(np.float64, copy=False) / (duration**order)


def feature_name(dim_index: int) -> str:
    return f"dim_{dim_index:02d}"


def load_signal_absolute_errors(episode_uid: str, spline_path: Path, cfg: Config) -> dict[str, np.ndarray]:
    source_arrays_dir = cfg.source_dataset_root / cfg.episodes_subdir / episode_uid / cfg.arrays_subdir
    source_state_path = source_arrays_dir / cfg.state_array_name
    if not source_state_path.is_file():
        raise FileNotFoundError(
            f"{episode_uid}: source state array not found for absolute error analysis: {source_state_path}"
        )

    target_state = np.load(source_state_path).astype(np.float64, copy=False)
    with np.load(spline_path, allow_pickle=False) as data:
        required = {"global_knots", "global_coefficients", "global_degree", "frame_to_u"}
        absent = sorted(required.difference(data.files))
        if absent:
            raise KeyError(f"{episode_uid}: {spline_path.name} lacks keys {absent}")
        knots = np.asarray(data["global_knots"], dtype=np.float64)
        coefficients = np.asarray(data["global_coefficients"], dtype=np.float64)
        degree = scalar(data["global_degree"], "global_degree")
        frame_to_u = np.asarray(data["frame_to_u"], dtype=np.float64)
        dt_seconds = scalar_float(data["dt_seconds"], "dt_seconds") if "dt_seconds" in data else math.nan
        duration_seconds = scalar_float(data["duration_seconds"], "duration_seconds") if "duration_seconds" in data else math.nan

    if target_state.ndim != 2:
        raise ValueError(f"{episode_uid}: expected 2D state array, got {target_state.shape}")
    if target_state.shape[0] != len(frame_to_u):
        raise ValueError(
            f"{episode_uid}: target state frame count {target_state.shape[0]} does not match frame_to_u {len(frame_to_u)}"
        )
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        dt_seconds, _, _ = infer_dt_seconds(source_arrays_dir, int(target_state.shape[0]), cfg)
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        duration_seconds = dt_seconds * max(len(frame_to_u) - 1, 1)
    if not math.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError(f"{episode_uid}: invalid dt_seconds={dt_seconds!r} in {spline_path.name}")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError(f"{episode_uid}: invalid duration_seconds={duration_seconds!r} in {spline_path.name}")

    spline = BSpline(knots, coefficients, degree)
    predicted_position = spline(frame_to_u).astype(np.float64, copy=False)
    if predicted_position.shape != target_state.shape:
        raise ValueError(
            f"{episode_uid}: predicted state shape {predicted_position.shape} does not match target state {target_state.shape}"
        )

    target_velocity, target_acceleration, target_jerk = compute_derivatives(target_state, dt_seconds)
    signal_targets = {
        "position": target_state,
        "velocity": target_velocity,
        "acceleration": target_acceleration,
        "jerk": target_jerk,
    }
    signal_predictions = {
        "position": predicted_position,
        "velocity": derivative_or_zero(spline, 1, frame_to_u, duration_seconds),
        "acceleration": derivative_or_zero(spline, 2, frame_to_u, duration_seconds),
        "jerk": derivative_or_zero(spline, 3, frame_to_u, duration_seconds),
    }
    return {
        signal_name: np.abs(signal_targets[signal_name] - signal_predictions[signal_name]).astype(np.float64, copy=False)
        for signal_name in SIGNAL_ORDERS
    }


def percentile_label(value: float) -> str:
    return f"p{value:g}".replace(".", "_")


def describe(values: np.ndarray, percentiles: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(values)
    if values.size == 0:
        result: dict[str, float | int] = {"count": 0, "mean": math.nan, "std": math.nan}
        result.update({percentile_label(p): math.nan for p in percentiles})
        return result
    result = {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }
    result.update(
        {
            percentile_label(percentile): float(value)
            for percentile, value in zip(percentiles, np.percentile(values, list(percentiles)), strict=True)
        }
    )
    return result


def knot_pair_frame_table(
    episode_uid: str,
    unique_knots: np.ndarray,
    continuous_frames: np.ndarray,
    rounded_frames: np.ndarray,
    num_frames: int,
    degree: int,
    intermediate_count: int,
) -> pd.DataFrame:
    step = intermediate_count + 1
    num_unique_knots = int(len(unique_knots))
    pair_count = num_unique_knots - step
    if pair_count <= 0:
        return pd.DataFrame()
    start_knot = np.arange(pair_count, dtype=np.int64)
    end_knot = start_knot + step
    frame_delta = rounded_frames[end_knot] - rounded_frames[start_knot]
    continuous_delta = continuous_frames[end_knot] - continuous_frames[start_knot]
    return pd.DataFrame(
        {
            "episode_uid": episode_uid,
            "num_original_frames": num_frames,
            "degree": degree,
            "num_unique_knots": num_unique_knots,
            "num_knot_spans": num_unique_knots - 1,
            "unique_knots_per_frame": num_unique_knots / num_frames,
            "frames_per_unique_knot": num_frames / num_unique_knots,
            "intermediate_knots": intermediate_count,
            "unique_knot_index_step": step,
            "start_knot_index": start_knot,
            "end_knot_index": end_knot,
            "start_u": unique_knots[start_knot],
            "end_u": unique_knots[end_knot],
            "start_frame_position": continuous_frames[start_knot],
            "end_frame_position": continuous_frames[end_knot],
            "continuous_frame_span": continuous_delta,
            "start_frame_index": rounded_frames[start_knot],
            "end_frame_index": rounded_frames[end_knot],
            "frame_index_delta": frame_delta,
            "frames_strictly_between": np.maximum(frame_delta - 1, 0),
            "frames_inclusive": frame_delta + 1,
        }
    )


def ensure_output_targets(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        cfg.episode_ratios_csv,
        cfg.ratio_distribution_csv,
        cfg.pair_details_parquet,
        cfg.pair_distribution_csv,
        cfg.pair_histogram_csv,
        cfg.absolute_error_distribution_csv,
        cfg.signal_error_distribution_csv,
        cfg.signal_error_per_episode_csv,
        cfg.signal_error_per_dimension_csv,
        cfg.signal_error_per_episode_dimension_csv,
        cfg.summary_json,
        cfg.resolved_config_yaml,
    ]
    existing = [cfg.output_dir / name for name in targets if (cfg.output_dir / name).exists()]
    if existing and not cfg.overwrite:
        raise FileExistsError(f"Output files already exist; enable output.overwrite: {existing}")


def print_distribution(label: str, payload: dict[str, Any], percentiles: tuple[float, ...]) -> None:
    ordered_keys = ["count", "mean", "std"] + [percentile_label(value) for value in percentiles]
    parts = []
    for key in ordered_keys:
        value = payload.get(key)
        if isinstance(value, (int, np.integer)):
            parts.append(f"{key}={int(value)}")
        elif value is None or (isinstance(value, float) and not math.isfinite(value)):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={float(value):.6f}")
    print(f"{label}: " + " | ".join(parts))


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    episode_splines = discover_episode_splines(cfg)
    ensure_output_targets(cfg)

    episode_rows: list[dict[str, Any]] = []
    pair_chunks: list[pd.DataFrame] = []
    values_by_gap: dict[int, list[np.ndarray]] = defaultdict(list)
    absolute_error_chunks: list[np.ndarray] = []
    signal_error_chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    signal_error_by_dimension: dict[str, list[np.ndarray]] = defaultdict(list)
    signal_error_per_episode_rows: list[dict[str, Any]] = []
    signal_error_per_episode_dimension_rows: list[dict[str, Any]] = []

    with tqdm(episode_splines, desc="Analyze knot-span frame distributions", unit="episode", dynamic_ncols=True) as episode_progress:
        for episode_index, (episode_uid, spline_path) in enumerate(episode_progress, start=1):
            inner_total = 2 + len(cfg.intermediate_counts)
            with tqdm(
                total=inner_total,
                desc=f"Episode: {episode_uid}",
                unit="step",
                leave=False,
                dynamic_ncols=True,
            ) as inner_progress:
                unique_knots, continuous_frames, rounded_frames, num_frames, degree, mode = load_knot_frames(
                    episode_uid, spline_path, cfg
                )
                num_unique_knots = int(len(unique_knots))
                episode_rows.append(
                    {
                        "episode_uid": episode_uid,
                        "spline_npz": str(spline_path),
                        "spline_target_mode": mode,
                        "degree": degree,
                        "num_original_frames": num_frames,
                        "num_unique_knots": num_unique_knots,
                        "num_knot_spans": num_unique_knots - 1,
                        "unique_knots_per_frame": num_unique_knots / num_frames,
                        "frames_per_unique_knot": num_frames / num_unique_knots,
                    }
                )
                inner_progress.update(1)
                inner_progress.set_postfix(unique_knots=num_unique_knots, frames=num_frames)

                signal_absolute_errors = load_signal_absolute_errors(episode_uid, spline_path, cfg)
                absolute_error_chunks.append(signal_absolute_errors["position"].reshape(-1))
                for signal_name, error_values in signal_absolute_errors.items():
                    signal_error_chunks[signal_name].append(error_values.reshape(-1))
                    signal_error_by_dimension[signal_name].append(error_values)
                    signal_error_per_episode_rows.append(
                        {
                            "episode_uid": episode_uid,
                            "signal_type": signal_name,
                            "num_frames": int(error_values.shape[0]),
                            "num_dimensions": int(error_values.shape[1]),
                            "metric": f"absolute_{signal_name}_error_all_dimensions_all_frames_per_episode",
                            **describe(error_values.reshape(-1), cfg.absolute_error_percentiles),
                        }
                    )
                    for dim_index in range(error_values.shape[1]):
                        signal_error_per_episode_dimension_rows.append(
                            {
                                "episode_uid": episode_uid,
                                "signal_type": signal_name,
                                "dimension_index": int(dim_index),
                                "feature_name": feature_name(dim_index),
                                "num_frames": int(error_values.shape[0]),
                                "metric": f"absolute_{signal_name}_error_per_dimension_per_episode",
                                **describe(error_values[:, dim_index], cfg.absolute_error_percentiles),
                            }
                        )
                inner_progress.update(1)

                for intermediate_count in cfg.intermediate_counts:
                    table = knot_pair_frame_table(
                        episode_uid,
                        unique_knots,
                        continuous_frames,
                        rounded_frames,
                        num_frames,
                        degree,
                        intermediate_count,
                    )
                    if not table.empty:
                        values_by_gap[intermediate_count].append(table[cfg.histogram_metric].to_numpy(copy=True))
                        pair_chunks.append(table)
                    inner_progress.update(1)
                    inner_progress.set_postfix(unique_knots=num_unique_knots, gap=intermediate_count)

            episode_progress.set_postfix(unique_knots=num_unique_knots, frames=num_frames)
            tqdm.write(
                f"[{episode_index:>3}/{len(episode_splines)}] {episode_uid}: "
                f"frames={num_frames}, unique_knots={num_unique_knots}, "
                f"ratio={num_unique_knots / num_frames:.6f}"
            )

    episode_df = pd.DataFrame(episode_rows)
    pair_df = pd.concat(pair_chunks, ignore_index=True) if pair_chunks else pd.DataFrame()
    if pair_df.empty:
        raise RuntimeError("No valid unique-knot pairs were produced")
    if cfg.histogram_metric not in pair_df.columns:
        raise KeyError(f"Unknown histogram_metric={cfg.histogram_metric!r}")

    ratio_values = episode_df["unique_knots_per_frame"].to_numpy()
    ratio_distribution_payload = {"metric": "unique_knots_per_frame", **describe(ratio_values, cfg.percentiles)}
    ratio_distribution_df = pd.DataFrame([ratio_distribution_payload])

    absolute_error_values = np.concatenate(absolute_error_chunks) if absolute_error_chunks else np.empty(0, dtype=np.float64)
    absolute_error_payload = {
        "metric": "absolute_error_all_dimensions_all_frames_all_episodes",
        "signal_type": "position",
        **describe(absolute_error_values, cfg.absolute_error_percentiles),
    }
    absolute_error_distribution_df = pd.DataFrame([absolute_error_payload])
    signal_error_distribution_rows: list[dict[str, Any]] = []
    signal_error_per_dimension_rows: list[dict[str, Any]] = []
    for signal_name in SIGNAL_ORDERS:
        signal_values = np.concatenate(signal_error_chunks.get(signal_name, [])) if signal_error_chunks.get(signal_name) else np.empty(0, dtype=np.float64)
        signal_error_distribution_rows.append(
            {
                "signal_type": signal_name,
                "metric": f"absolute_{signal_name}_error_all_dimensions_all_frames_all_episodes",
                **describe(signal_values, cfg.absolute_error_percentiles),
            }
        )

        per_dimension_arrays = signal_error_by_dimension.get(signal_name, [])
        stacked = np.concatenate(per_dimension_arrays, axis=0) if per_dimension_arrays else np.empty((0, 0), dtype=np.float64)
        for dim_index in range(stacked.shape[1] if stacked.ndim == 2 else 0):
            signal_error_per_dimension_rows.append(
                {
                    "signal_type": signal_name,
                    "dimension_index": int(dim_index),
                    "feature_name": feature_name(dim_index),
                    "num_frames": int(stacked.shape[0]),
                    "metric": f"absolute_{signal_name}_error_per_dimension_all_frames_all_episodes",
                    **describe(stacked[:, dim_index], cfg.absolute_error_percentiles),
                }
            )
    signal_error_distribution_df = pd.DataFrame(signal_error_distribution_rows)
    signal_error_per_episode_df = pd.DataFrame(signal_error_per_episode_rows)
    signal_error_per_dimension_df = pd.DataFrame(signal_error_per_dimension_rows)
    signal_error_per_episode_dimension_df = pd.DataFrame(signal_error_per_episode_dimension_rows)

    distribution_rows: list[dict[str, Any]] = []
    histogram_rows: list[dict[str, Any]] = []
    for intermediate_count in cfg.intermediate_counts:
        arrays = values_by_gap.get(intermediate_count, [])
        values = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        payload = {
            "intermediate_knots": intermediate_count,
            "unique_knot_index_step": intermediate_count + 1,
            "metric": cfg.histogram_metric,
            **describe(values, cfg.percentiles),
        }
        distribution_rows.append(payload)
        if values.size:
            unique, counts = np.unique(values, return_counts=True)
            histogram_rows.extend(
                {
                    "intermediate_knots": intermediate_count,
                    "unique_knot_index_step": intermediate_count + 1,
                    "metric": cfg.histogram_metric,
                    "frame_count": float(value),
                    "pair_count": int(count),
                    "proportion": float(count / values.size),
                }
                for value, count in zip(unique, counts, strict=True)
            )

    episode_df.to_csv(cfg.output_dir / cfg.episode_ratios_csv, index=False)
    ratio_distribution_df.to_csv(cfg.output_dir / cfg.ratio_distribution_csv, index=False)
    pair_df.to_parquet(cfg.output_dir / cfg.pair_details_parquet, index=False)
    pd.DataFrame(distribution_rows).to_csv(cfg.output_dir / cfg.pair_distribution_csv, index=False)
    pd.DataFrame(histogram_rows).to_csv(cfg.output_dir / cfg.pair_histogram_csv, index=False)
    absolute_error_distribution_df.to_csv(cfg.output_dir / cfg.absolute_error_distribution_csv, index=False)
    signal_error_distribution_df.to_csv(cfg.output_dir / cfg.signal_error_distribution_csv, index=False)
    signal_error_per_episode_df.to_csv(cfg.output_dir / cfg.signal_error_per_episode_csv, index=False)
    signal_error_per_dimension_df.to_csv(cfg.output_dir / cfg.signal_error_per_dimension_csv, index=False)
    signal_error_per_episode_dimension_df.to_csv(cfg.output_dir / cfg.signal_error_per_episode_dimension_csv, index=False)

    resolved = dict(cfg.raw)
    resolved["dataset_root"] = str(cfg.dataset_root)
    resolved["source_dataset_root"] = str(cfg.source_dataset_root)
    resolved["resolved_output_dir"] = str(cfg.output_dir)
    (cfg.output_dir / cfg.resolved_config_yaml).write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    summary = {
        "dataset_root": str(cfg.dataset_root),
        "source_dataset_root": str(cfg.source_dataset_root),
        "spline_npz_name": cfg.spline_npz_name,
        "state_array_name": cfg.state_array_name,
        "allowed_target_modes": list(cfg.allowed_target_modes),
        "pair_definition": "unique knot pair (i, i + N + 1) has exactly N intermediate unique knots",
        "frame_metrics": {
            "frame_index_delta": "end_frame_index - start_frame_index",
            "frames_strictly_between": "max(frame_index_delta - 1, 0)",
            "frames_inclusive": "frame_index_delta + 1",
            "continuous_frame_span": "unrounded end_frame_position - start_frame_position",
        },
        "episodes": int(len(episode_df)),
        "total_original_frames": int(episode_df["num_original_frames"].sum()),
        "total_unique_knots": int(episode_df["num_unique_knots"].sum()),
        "total_pairs": int(len(pair_df)),
        "ratio_distribution": ratio_distribution_payload,
        "pair_distributions": distribution_rows,
        "absolute_error_distribution": absolute_error_payload,
        "signal_error_distributions": signal_error_distribution_rows,
        "outputs": {
            "episode_ratios": cfg.episode_ratios_csv,
            "ratio_distribution": cfg.ratio_distribution_csv,
            "pair_details": cfg.pair_details_parquet,
            "pair_distribution": cfg.pair_distribution_csv,
            "pair_histogram": cfg.pair_histogram_csv,
            "absolute_error_distribution": cfg.absolute_error_distribution_csv,
            "signal_error_distribution": cfg.signal_error_distribution_csv,
            "signal_error_per_episode": cfg.signal_error_per_episode_csv,
            "signal_error_per_dimension": cfg.signal_error_per_dimension_csv,
            "signal_error_per_episode_dimension": cfg.signal_error_per_episode_dimension_csv,
        },
    }
    (cfg.output_dir / cfg.summary_json).write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )

    print_distribution("Unique knot/frame ratio", ratio_distribution_payload, cfg.percentiles)
    for payload in distribution_rows:
        label = (
            f"Knot pairs with {payload['intermediate_knots']} intermediate knots "
            f"({cfg.histogram_metric})"
        )
        print_distribution(label, payload, cfg.percentiles)
    print_distribution(
        "Absolute error over all dims/frames/episodes",
        absolute_error_payload,
        cfg.absolute_error_percentiles,
    )
    for payload in signal_error_distribution_rows:
        print_distribution(
            f"Absolute {payload['signal_type']} error over all dims/frames/episodes",
            payload,
            cfg.absolute_error_percentiles,
        )
    print(f"Wrote {len(episode_df)} episode rows and {len(pair_df):,} pair rows to {cfg.output_dir}")


if __name__ == "__main__":
    main()
