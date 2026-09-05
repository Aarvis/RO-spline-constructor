"""Analyze control-point/frame coverage and reconstruction errors for action splines.

The input root may be a single spline run (``root/episodes/...``) or a
multi-degree root containing ``degree_*/episodes/...`` subdirectories. Delta
action splines are reconstructed with their stored ``state_65d[0]`` reference
before position-level errors are measured.
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


DEFAULT_CONFIG = Path(__file__).with_name("config_action_control_point_frame_distribution.yaml")
SIGNAL_ORDERS = {"action": 0, "action_velocity": 1, "action_acceleration": 2, "action_jerk": 3}


@dataclass(frozen=True)
class RunRoot:
    name: str
    root: Path


@dataclass(frozen=True)
class Config:
    source_dataset_root: Path
    run_roots: tuple[Path, ...]
    episodes_subdir: str
    arrays_subdir: str
    max_episodes: int | None
    spline_npz_name: str
    action_array_name: str
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
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    input_cfg = raw.get("input", {})
    analysis = raw.get("analysis", {})
    output = raw.get("output", {})
    roots = tuple(Path(value).expanduser().resolve() for value in raw.get("spline_run_roots", []))
    if not roots:
        raise ValueError("spline_run_roots must contain at least one spline output directory")
    counts = tuple(sorted({int(value) for value in analysis["intermediate_control_point_counts"]}))
    if not counts or counts[0] < 0:
        raise ValueError("intermediate_control_point_counts must contain non-negative integers")
    rounded_method = str(analysis.get("rounded_frame_method", "nearest"))
    if rounded_method not in {"nearest", "floor", "ceil"}:
        raise ValueError("rounded_frame_method must be nearest, floor, or ceil")
    max_episodes = args.max_episodes if args.max_episodes is not None else raw.get("max_episodes")
    return Config(
        source_dataset_root=Path(raw["source_dataset_root"]).expanduser().resolve(),
        run_roots=roots,
        episodes_subdir=str(raw.get("episodes_subdir", "episodes")),
        arrays_subdir=str(raw.get("arrays_subdir", "arrays")),
        max_episodes=int(max_episodes) if max_episodes is not None else None,
        spline_npz_name=str(input_cfg.get("spline_npz_name", "global_action_spline.npz")),
        action_array_name=str(input_cfg.get("action_array_name", "action_65d.npy")),
        timestamps_array_name=str(input_cfg.get("timestamps_array_name", "timestamps.npy")),
        fallback_fps=float(input_cfg.get("fallback_fps", 30.0)),
        use_timestamps_if_available=bool(input_cfg.get("use_timestamps_if_available", True)),
        allowed_target_modes=tuple(
            str(value)
            for value in input_cfg.get("allowed_target_modes", ["global_action", "global_delta_action_from_initial_state"])
        ),
        intermediate_counts=counts,
        rounded_frame_method=rounded_method,
        percentiles=tuple(float(value) for value in analysis.get("percentiles", [0, 1, 5, 25, 50, 75, 95, 99, 100])),
        absolute_error_percentiles=tuple(
            float(value)
            for value in analysis.get("absolute_error_percentiles", [0, 1, 5, 25, 50, 75, 95, 99, 99.9, 100])
        ),
        histogram_metric=str(analysis.get("histogram_metric", "frames_inclusive")),
        output_dir=Path(output["output_dir"]).expanduser().resolve(),
        overwrite=bool(args.overwrite or output.get("overwrite", False)),
        raw=raw,
    )


def discover_run_roots(cfg: Config) -> list[RunRoot]:
    runs: list[RunRoot] = []
    for root in cfg.run_roots:
        if (root / cfg.episodes_subdir).is_dir():
            runs.append(RunRoot(root.name, root))
            continue
        degree_roots = sorted(path for path in root.iterdir() if path.is_dir() and (path / cfg.episodes_subdir).is_dir()) if root.is_dir() else []
        if not degree_roots:
            raise FileNotFoundError(f"No {cfg.episodes_subdir!r} directory found under spline root: {root}")
        runs.extend(RunRoot(f"{root.name}/{path.name}", path) for path in degree_roots)
    return runs


def discover_episode_splines(run: RunRoot, cfg: Config) -> list[tuple[str, Path]]:
    episode_dirs = sorted(path for path in (run.root / cfg.episodes_subdir).iterdir() if path.is_dir())
    if cfg.max_episodes is not None:
        episode_dirs = episode_dirs[: cfg.max_episodes]
    paths = [(episode.name, episode / cfg.arrays_subdir / cfg.spline_npz_name) for episode in episode_dirs]
    missing = [episode_uid for episode_uid, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{run.name}: missing {cfg.spline_npz_name} for {', '.join(missing[:5])}")
    return paths


def scalar(array: np.ndarray, name: str) -> int:
    values = np.asarray(array).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    return int(values[0])


def infer_dt_seconds(source_arrays_dir: Path, num_frames: int, cfg: Config) -> tuple[float, float]:
    timestamps_path = source_arrays_dir / cfg.timestamps_array_name
    if cfg.use_timestamps_if_available and timestamps_path.is_file():
        timestamps = np.load(timestamps_path).astype(np.float64, copy=False)
        if timestamps.ndim == 1 and len(timestamps) == num_frames:
            deltas = np.diff(timestamps)
            deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
            if len(deltas):
                dt = float(np.median(deltas))
                return dt, 1.0 / dt
    if not math.isfinite(cfg.fallback_fps) or cfg.fallback_fps <= 0.0:
        raise ValueError(f"fallback_fps must be positive, got {cfg.fallback_fps!r}")
    return 1.0 / cfg.fallback_fps, cfg.fallback_fps


def round_frame_positions(values: np.ndarray, method: str) -> np.ndarray:
    if method == "nearest":
        return np.floor(values + 0.5).astype(np.int64)
    if method == "floor":
        return np.floor(values).astype(np.int64)
    return np.ceil(values).astype(np.int64)


def load_spline_metadata(episode_uid: str, path: Path, cfg: Config) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {"global_knots", "global_coefficients", "global_degree", "frame_indices", "frame_to_u"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise KeyError(f"{episode_uid}: {path.name} lacks keys {missing}")
        mode = str(np.asarray(data["spline_target_mode"]).reshape(-1)[0]) if "spline_target_mode" in data else ""
        if mode not in cfg.allowed_target_modes:
            raise ValueError(f"{episode_uid}: unsupported spline_target_mode={mode!r}")
        reference = np.asarray(data["reference_state_65d"], dtype=np.float64) if "reference_state_65d" in data else None
        return {
            "knots": np.asarray(data["global_knots"], dtype=np.float64),
            "coefficients": np.asarray(data["global_coefficients"], dtype=np.float64),
            "degree": scalar(data["global_degree"], "global_degree"),
            "frame_indices": np.asarray(data["frame_indices"], dtype=np.int64),
            "frame_to_u": np.asarray(data["frame_to_u"], dtype=np.float64),
            "mode": mode,
            "reference": reference,
        }


def compute_derivatives(signal: np.ndarray, dt_seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_order = 2 if len(signal) >= 3 else 1
    velocity = np.gradient(signal, dt_seconds, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, dt_seconds, axis=0, edge_order=edge_order)
    jerk = np.gradient(acceleration, dt_seconds, axis=0, edge_order=edge_order)
    return velocity, acceleration, jerk


def spline_derivative(spline: BSpline, order: int, frame_to_u: np.ndarray, duration_seconds: float) -> np.ndarray:
    if order > spline.k:
        return np.zeros((len(frame_to_u), spline.c.shape[1]), dtype=np.float64)
    return spline.derivative(order)(frame_to_u).astype(np.float64, copy=False) / (duration_seconds**order)


def load_signal_errors(run: RunRoot, episode_uid: str, spline_path: Path, metadata: dict[str, Any], cfg: Config) -> dict[str, np.ndarray]:
    source_arrays_dir = cfg.source_dataset_root / cfg.episodes_subdir / episode_uid / cfg.arrays_subdir
    action_path = source_arrays_dir / cfg.action_array_name
    if not action_path.is_file():
        raise FileNotFoundError(f"{run.name}/{episode_uid}: action source missing: {action_path}")
    target = np.load(action_path).astype(np.float64, copy=False)
    frame_to_u = metadata["frame_to_u"]
    if target.ndim != 2 or target.shape[0] != len(frame_to_u):
        raise ValueError(f"{run.name}/{episode_uid}: action shape {target.shape} does not match spline frame mapping")
    spline = BSpline(metadata["knots"], metadata["coefficients"], metadata["degree"])
    action_prediction = spline(frame_to_u).astype(np.float64, copy=False)
    if metadata["mode"] == "global_delta_action_from_initial_state":
        reference = metadata["reference"]
        if reference is None or reference.shape != (target.shape[1],):
            raise ValueError(f"{run.name}/{episode_uid}: delta spline lacks a compatible reference_state_65d")
        action_prediction = action_prediction + reference[None, :]
    if action_prediction.shape != target.shape:
        raise ValueError(f"{run.name}/{episode_uid}: spline output {action_prediction.shape} does not match action {target.shape}")
    dt_seconds, _ = infer_dt_seconds(source_arrays_dir, len(target), cfg)
    duration_seconds = dt_seconds * max(len(target) - 1, 1)
    target_velocity, target_acceleration, target_jerk = compute_derivatives(target, dt_seconds)
    predictions = {
        "action": action_prediction,
        "action_velocity": spline_derivative(spline, 1, frame_to_u, duration_seconds),
        "action_acceleration": spline_derivative(spline, 2, frame_to_u, duration_seconds),
        "action_jerk": spline_derivative(spline, 3, frame_to_u, duration_seconds),
    }
    targets = {
        "action": target,
        "action_velocity": target_velocity,
        "action_acceleration": target_acceleration,
        "action_jerk": target_jerk,
    }
    return {name: np.abs(targets[name] - predictions[name]) for name in SIGNAL_ORDERS}


def describe(values: np.ndarray, percentiles: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    result: dict[str, float | int] = {"count": int(values.size)}
    if not values.size:
        result.update({"mean": math.nan, "std": math.nan})
        result.update({f"p{value:g}".replace(".", "_"): math.nan for value in percentiles})
        return result
    result.update({"mean": float(np.mean(values)), "std": float(np.std(values))})
    result.update(
        {
            f"p{percentile:g}".replace(".", "_"): float(value)
            for percentile, value in zip(percentiles, np.percentile(values, list(percentiles)), strict=True)
        }
    )
    return result


def print_distribution(label: str, payload: dict[str, Any], percentiles: Iterable[float]) -> None:
    keys = ["count", "mean", "std"] + [f"p{value:g}".replace(".", "_") for value in percentiles]
    values = []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, np.integer)):
            values.append(f"{key}={int(value)}")
        elif value is None or not math.isfinite(float(value)):
            values.append(f"{key}=nan")
        else:
            values.append(f"{key}={float(value):.6g}")
    print(f"{label}: {', '.join(values)}")


def knot_pair_table(run_name: str, episode_uid: str, metadata: dict[str, Any], intermediate_count: int, method: str) -> pd.DataFrame:
    knots = metadata["knots"]
    frame_indices = metadata["frame_indices"]
    frame_to_u = metadata["frame_to_u"]
    unique_knots = np.unique(knots)
    continuous_frames = np.interp(unique_knots, frame_to_u, frame_indices.astype(np.float64))
    rounded_frames = np.clip(
        round_frame_positions(continuous_frames, method), int(frame_indices.min()), int(frame_indices.max())
    )
    step = intermediate_count + 1
    pair_count = len(unique_knots) - step
    if pair_count <= 0:
        return pd.DataFrame()
    start = np.arange(pair_count, dtype=np.int64)
    end = start + step
    delta = rounded_frames[end] - rounded_frames[start]
    return pd.DataFrame(
        {
            "run_name": run_name,
            "episode_uid": episode_uid,
            "degree": metadata["degree"],
            "spline_target_mode": metadata["mode"],
            "num_original_frames": len(frame_indices),
            "num_unique_knots": len(unique_knots),
            "intermediate_knots": intermediate_count,
            "start_knot_index": start,
            "end_knot_index": end,
            "start_frame_index": rounded_frames[start],
            "end_frame_index": rounded_frames[end],
            "continuous_frame_span": continuous_frames[end] - continuous_frames[start],
            "frame_index_delta": delta,
            "frames_strictly_between": np.maximum(delta - 1, 0),
            "frames_inclusive": delta + 1,
        }
    )


def ensure_output_dir(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if any(cfg.output_dir.iterdir()) and not cfg.overwrite:
        raise FileExistsError(f"Output directory is not empty; enable output.overwrite: {cfg.output_dir}")


def json_default(value: Any) -> Any:
    """Convert NumPy values retained from pandas group keys to JSON primitives."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    ensure_output_dir(cfg)
    runs = discover_run_roots(cfg)
    episode_rows: list[dict[str, Any]] = []
    pair_tables: list[pd.DataFrame] = []
    error_values: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    error_dimension_values: dict[tuple[str, int, str, int], list[np.ndarray]] = defaultdict(list)
    error_episode_rows: list[dict[str, Any]] = []
    error_episode_dimension_rows: list[dict[str, Any]] = []
    frame_mae_values: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    frame_max_abs_dimension_error_values: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)

    run_splines = [(run, discover_episode_splines(run, cfg)) for run in runs]
    total_episodes = sum(len(splines) for _, splines in run_splines)
    with tqdm(total=total_episodes, desc="Analyze all action spline episodes", unit="episode", dynamic_ncols=True) as overall:
        for run, splines in run_splines:
            for episode_uid, spline_path in tqdm(
                splines,
                desc=f"Analyze {run.name}",
                unit="episode",
                leave=False,
                dynamic_ncols=True,
                position=1,
            ):
                metadata = load_spline_metadata(episode_uid, spline_path, cfg)
                unique_knots = np.unique(metadata["knots"])
                num_frames = len(metadata["frame_indices"])
                episode_rows.append(
                    {
                        "run_name": run.name,
                        "episode_uid": episode_uid,
                        "degree": metadata["degree"],
                        "spline_target_mode": metadata["mode"],
                        "num_original_frames": num_frames,
                        "num_unique_knots": len(unique_knots),
                        "num_control_points": metadata["coefficients"].shape[0],
                        "control_points_per_frame": metadata["coefficients"].shape[0] / num_frames,
                        "unique_knots_per_frame": len(unique_knots) / num_frames,
                        "frames_per_unique_knot": num_frames / len(unique_knots),
                    }
                )
                for intermediate_count in cfg.intermediate_counts:
                    table = knot_pair_table(run.name, episode_uid, metadata, intermediate_count, cfg.rounded_frame_method)
                    if not table.empty:
                        pair_tables.append(table)
                errors = load_signal_errors(run, episode_uid, spline_path, metadata, cfg)
                frame_mae_values[(run.name, metadata["degree"])].append(np.mean(errors["action"], axis=1))
                frame_max_abs_dimension_error = np.max(errors["action"], axis=1)
                frame_max_abs_dimension_error_values[(run.name, metadata["degree"])].append(
                    frame_max_abs_dimension_error
                )
                for signal_name, values in errors.items():
                    base = {
                        "run_name": run.name,
                        "episode_uid": episode_uid,
                        "degree": metadata["degree"],
                        "spline_target_mode": metadata["mode"],
                        "signal_type": signal_name,
                    }
                    error_values[(run.name, metadata["degree"], signal_name)].append(values.reshape(-1))
                    error_episode_rows.append({**base, **describe(values.reshape(-1), cfg.absolute_error_percentiles)})
                    for dimension in range(values.shape[1]):
                        dimension_base = {**base, "dimension": dimension}
                        error_dimension_values[(run.name, metadata["degree"], signal_name, dimension)].append(
                            values[:, dimension]
                        )
                        error_episode_dimension_rows.append(
                            {**dimension_base, **describe(values[:, dimension], cfg.absolute_error_percentiles)}
                        )
                overall.update(1)

    episode_df = pd.DataFrame(episode_rows)
    pair_df = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()
    ratio_rows = []
    ratio_groups = list(episode_df.groupby(["run_name", "degree"], sort=True))
    for (run_name, degree), group in tqdm(ratio_groups, desc="Summarize control-point ratios", unit="group", dynamic_ncols=True):
        ratio_rows.append(
            {
                "run_name": run_name,
                "degree": degree,
                "metric": "control_points_per_frame",
                **describe(group["control_points_per_frame"].to_numpy(), cfg.percentiles),
            }
        )
        ratio_rows.append(
            {
                "run_name": run_name,
                "degree": degree,
                "metric": "unique_knots_per_frame",
                **describe(group["unique_knots_per_frame"].to_numpy(), cfg.percentiles),
            }
        )
    pair_distribution_rows = []
    pair_histogram_rows = []
    if not pair_df.empty:
        pair_groups = list(pair_df.groupby(["run_name", "degree", "intermediate_knots"], sort=True))
        for (run_name, degree, intermediate_count), group in tqdm(
            pair_groups, desc="Summarize control-point pairs", unit="group", dynamic_ncols=True
        ):
            values = group[cfg.histogram_metric].to_numpy(dtype=np.float64)
            pair_distribution_rows.append(
                {
                    "run_name": run_name,
                    "degree": degree,
                    "intermediate_knots": intermediate_count,
                    "metric": cfg.histogram_metric,
                    **describe(values, cfg.percentiles),
                }
            )
            unique_values, counts = np.unique(values, return_counts=True)
            pair_histogram_rows.extend(
                {
                    "run_name": run_name,
                    "degree": degree,
                    "intermediate_knots": intermediate_count,
                    "metric": cfg.histogram_metric,
                    "frame_count": value,
                    "pair_count": int(count),
                    "proportion": float(count / len(values)),
                }
                for value, count in zip(unique_values, counts, strict=True)
            )
    error_distribution_rows = []
    for (run_name, degree, signal_name), arrays in tqdm(
        sorted(error_values.items()), desc="Summarize signal errors", unit="signal", dynamic_ncols=True
    ):
        error_distribution_rows.append(
            {
                "run_name": run_name,
                "degree": degree,
                "signal_type": signal_name,
                **describe(np.concatenate(arrays), cfg.absolute_error_percentiles),
            }
        )
    error_dimension_rows = []
    for (run_name, degree, signal_name, dimension), arrays in tqdm(
        sorted(error_dimension_values.items()), desc="Summarize error dimensions", unit="dimension", dynamic_ncols=True
    ):
        error_dimension_rows.append(
            {
                "run_name": run_name,
                "degree": degree,
                "signal_type": signal_name,
                "dimension": dimension,
                **describe(np.concatenate(arrays), cfg.absolute_error_percentiles),
            }
        )
    frame_max_abs_dimension_error_distribution_rows = []
    for (run_name, degree), arrays in tqdm(
        sorted(frame_max_abs_dimension_error_values.items()),
        desc="Summarize framewise max errors",
        unit="run",
        dynamic_ncols=True,
    ):
        frame_max_abs_dimension_error_distribution_rows.append(
            {
                "run_name": run_name,
                "degree": degree,
                "metric": "max_abs_dimension_error_across_all_dataset_frames",
                **describe(np.concatenate(arrays), cfg.absolute_error_percentiles),
            }
        )
    frame_mae_distribution_rows = []
    for (run_name, degree), arrays in tqdm(
        sorted(frame_mae_values.items()), desc="Summarize framewise MAE-65D", unit="run", dynamic_ncols=True
    ):
        frame_mae_distribution_rows.append(
            {
                "run_name": run_name,
                "degree": degree,
                "metric": "mae_65d_across_all_dataset_frames",
                **describe(np.concatenate(arrays), cfg.absolute_error_percentiles),
            }
        )

    resolved = dict(cfg.raw)
    resolved["resolved_run_roots"] = [str(run.root) for run in runs]
    resolved["resolved_output_dir"] = str(cfg.output_dir)
    summary = {
        "run_roots": {run.name: str(run.root) for run in runs},
        "source_dataset_root": str(cfg.source_dataset_root),
        "episodes": int(len(episode_df)),
        "total_pairs": int(len(pair_df)),
        "ratio_distribution": ratio_rows,
        "signal_error_distribution": error_distribution_rows,
        "framewise_max_abs_dimension_error_distribution": frame_max_abs_dimension_error_distribution_rows,
        "framewise_mae_65d_distribution": frame_mae_distribution_rows,
    }
    output_writers = [
        ("episode ratios", lambda: episode_df.to_csv(cfg.output_dir / "episode_control_point_frame_ratios.csv", index=False)),
        ("ratio distribution", lambda: pd.DataFrame(ratio_rows).to_csv(cfg.output_dir / "control_point_frame_ratio_distribution.csv", index=False)),
        ("pair details", lambda: pair_df.to_parquet(cfg.output_dir / "control_point_pair_frame_counts.parquet", index=False)),
        ("pair distribution", lambda: pd.DataFrame(pair_distribution_rows).to_csv(cfg.output_dir / "control_point_pair_frame_distribution.csv", index=False)),
        ("pair histogram", lambda: pd.DataFrame(pair_histogram_rows).to_csv(cfg.output_dir / "control_point_pair_frame_histogram.csv", index=False)),
        ("signal errors", lambda: pd.DataFrame(error_distribution_rows).to_csv(cfg.output_dir / "action_signal_absolute_error_distribution.csv", index=False)),
        ("errors per episode", lambda: pd.DataFrame(error_episode_rows).to_csv(cfg.output_dir / "action_signal_absolute_error_per_episode.csv", index=False)),
        ("errors per dimension", lambda: pd.DataFrame(error_dimension_rows).to_csv(cfg.output_dir / "action_signal_absolute_error_per_dimension.csv", index=False)),
        ("errors per episode dimension", lambda: pd.DataFrame(error_episode_dimension_rows).to_csv(cfg.output_dir / "action_signal_absolute_error_per_episode_dimension.csv", index=False)),
        ("framewise max errors", lambda: pd.DataFrame(frame_max_abs_dimension_error_distribution_rows).to_csv(cfg.output_dir / "framewise_max_abs_dimension_error_distribution.csv", index=False)),
        ("framewise MAE-65D", lambda: pd.DataFrame(frame_mae_distribution_rows).to_csv(cfg.output_dir / "framewise_mae_65d_distribution.csv", index=False)),
        ("resolved config", lambda: (cfg.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")),
        ("summary", lambda: (cfg.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True, default=json_default), encoding="utf-8")),
    ]
    for _, write_output in tqdm(output_writers, desc="Write analysis outputs", unit="file", dynamic_ncols=True):
        write_output()
    for row in ratio_rows:
        print_distribution(
            f"{row['run_name']} degree-{row['degree']} {row['metric']}", row, cfg.percentiles
        )
    for row in pair_distribution_rows:
        print_distribution(
            f"{row['run_name']} degree-{row['degree']} knot pairs with "
            f"{row['intermediate_knots']} intermediate knots ({row['metric']})",
            row,
            cfg.percentiles,
        )
    for row in error_distribution_rows:
        print_distribution(
            f"{row['run_name']} degree-{row['degree']} absolute {row['signal_type']} error",
            row,
            cfg.absolute_error_percentiles,
        )
    for row in frame_max_abs_dimension_error_distribution_rows:
        print_distribution(
            f"{row['run_name']} degree-{row['degree']} max absolute dimension error across all dataset frames",
            row,
            cfg.absolute_error_percentiles,
        )
    for row in frame_mae_distribution_rows:
        print_distribution(
            f"{row['run_name']} degree-{row['degree']} MAE-65D across all dataset frames",
            row,
            cfg.absolute_error_percentiles,
        )
    print(f"Analyzed {len(episode_df)} spline episodes across {len(runs)} run roots.")
    print(f"Wrote analysis outputs to {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
