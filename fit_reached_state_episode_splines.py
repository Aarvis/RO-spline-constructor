from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.signal import savgol_filter
from tqdm.auto import tqdm


SIGNAL_NAMES = ("position", "velocity", "acceleration", "jerk")
EPISODE_RMSE_DISTRIBUTION_PERCENTILES = (
    ("min", 0.0),
    ("p1", 1.0),
    ("p2", 2.0),
    ("p5", 5.0),
    ("p15", 15.0),
    ("p25", 25.0),
    ("median", 50.0),
    ("p75", 75.0),
    ("p85", 85.0),
    ("p95", 95.0),
    ("p98", 98.0),
    ("p99", 99.0),
    ("max", 100.0),
)


@dataclass(frozen=True)
class Config:
    dataset_root: Path
    num_workers: int
    max_episodes: int | None
    state_array_name: str
    timestamps_array_name: str
    fallback_fps: float
    use_timestamps_if_available: bool
    smoothing_enabled: bool
    smoothing_method: str
    savgol_window: int
    savgol_polyorder: int
    normalization_center_method: str
    normalization_lower_percentile: float
    normalization_upper_percentile: float
    normalization_min_scale: float
    normalization_stats_npz_name: str
    normalization_stats_json_name: str
    normalization_stats_table_csv_name: str
    normalization_stats_table_parquet_name: str
    degree: int
    enabled_error_signals: tuple[str, ...]
    initial_internal_knots: int
    max_internal_knots: int
    max_control_points: int | None
    max_iterations: int
    min_knot_spacing_frames: int
    local_window_size: int
    endpoint_weight: float
    epsilon_position: float
    epsilon_velocity: float
    epsilon_acceleration: float
    epsilon_jerk: float
    report_episode_observation_rmse_distribution: bool
    run_name: str
    overwrite: bool
    spline_npz_name: str
    spline_index_name: str
    fit_summary_parquet_name: str
    fit_summary_csv_name: str
    resolved_config_name: str


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return str(value_float)
    return f"{value_float:.{digits}f}"


def required_signal_names(enabled_error_signals: tuple[str, ...]) -> tuple[str, ...]:
    max_index = max(SIGNAL_NAMES.index(signal_name) for signal_name in enabled_error_signals)
    return SIGNAL_NAMES[: max_index + 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build dataset-level normalization stats from reached state trajectories and fit "
            "adaptive shared-knot full-episode cubic B-splines over observation.state."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("dataset-processing/reached_state_episode_spline_constructor/config_reached_state_episode_splines.yaml"),
        help="YAML config path.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root override.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of episodes to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing spline files.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Config:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    input_cfg = raw.get("input", {})
    preprocessing_cfg = raw.get("preprocessing", {})
    smoothing_cfg = preprocessing_cfg.get("smoothing", {})
    normalization_cfg = raw.get("normalization", {})
    fit_cfg = raw.get("fit", {})
    output_cfg = raw.get("output", {})

    dataset_root_value = args.dataset_root or raw.get("dataset_root")
    if dataset_root_value is None:
        raise ValueError("dataset_root is required in config or --dataset-root")

    max_episodes = (
        int(args.max_episodes)
        if args.max_episodes is not None
        else (int(raw["max_episodes"]) if raw.get("max_episodes") is not None else None)
    )
    max_control_points = fit_cfg.get("max_control_points")
    enabled_error_signals_raw = fit_cfg.get("enabled_error_signals", list(SIGNAL_NAMES))
    enabled_error_signals = tuple(str(value).lower().strip() for value in enabled_error_signals_raw)
    invalid_signals = sorted(set(enabled_error_signals) - set(SIGNAL_NAMES))
    if invalid_signals:
        raise ValueError(f"Unsupported enabled_error_signals: {invalid_signals}. Expected subset of {SIGNAL_NAMES}.")
    if not enabled_error_signals:
        raise ValueError("fit.enabled_error_signals must contain at least one signal.")

    return Config(
        dataset_root=Path(dataset_root_value),
        num_workers=max(1, int(raw.get("num_workers", 1))),
        max_episodes=max_episodes,
        state_array_name=str(input_cfg.get("state_array_name", "state_65d.npy")),
        timestamps_array_name=str(input_cfg.get("timestamps_array_name", "timestamps.npy")),
        fallback_fps=float(input_cfg.get("fallback_fps", 30.0)),
        use_timestamps_if_available=bool(input_cfg.get("use_timestamps_if_available", True)),
        smoothing_enabled=bool(smoothing_cfg.get("enabled", True)),
        smoothing_method=str(smoothing_cfg.get("method", "savgol")).lower().strip(),
        savgol_window=int(smoothing_cfg.get("savgol_window", 7)),
        savgol_polyorder=int(smoothing_cfg.get("savgol_polyorder", 3)),
        normalization_center_method=str(normalization_cfg.get("center_method", "median")).lower().strip(),
        normalization_lower_percentile=float(normalization_cfg.get("lower_percentile", 5.0)),
        normalization_upper_percentile=float(normalization_cfg.get("upper_percentile", 95.0)),
        normalization_min_scale=float(normalization_cfg.get("min_scale", 1.0e-4)),
        normalization_stats_npz_name=str(
            normalization_cfg.get("stats_npz_name", "reached_state_spline_normalization_stats.npz")
        ),
        normalization_stats_json_name=str(
            normalization_cfg.get("stats_json_name", "reached_state_spline_normalization_summary.json")
        ),
        normalization_stats_table_csv_name=str(
            normalization_cfg.get(
                "stats_table_csv_name",
                "reached_state_spline_normalization_per_dimension.csv",
            )
        ),
        normalization_stats_table_parquet_name=str(
            normalization_cfg.get(
                "stats_table_parquet_name",
                "reached_state_spline_normalization_per_dimension.parquet",
            )
        ),
        degree=int(fit_cfg.get("degree", 3)),
        enabled_error_signals=enabled_error_signals,
        initial_internal_knots=int(fit_cfg.get("initial_internal_knots", 4)),
        max_internal_knots=int(fit_cfg.get("max_internal_knots", 64)),
        max_control_points=(int(max_control_points) if max_control_points is not None else None),
        max_iterations=int(fit_cfg.get("max_iterations", 64)),
        min_knot_spacing_frames=int(fit_cfg.get("min_knot_spacing_frames", 5)),
        local_window_size=int(fit_cfg.get("local_window_size", 11)),
        endpoint_weight=float(fit_cfg.get("endpoint_weight", 1000.0)),
        epsilon_position=float(fit_cfg.get("epsilon_position", 0.35)),
        epsilon_velocity=float(fit_cfg.get("epsilon_velocity", 0.60)),
        epsilon_acceleration=float(fit_cfg.get("epsilon_acceleration", 0.80)),
        epsilon_jerk=float(fit_cfg.get("epsilon_jerk", 1.20)),
        report_episode_observation_rmse_distribution=bool(
            fit_cfg.get("report_episode_observation_rmse_distribution", True)
        ),
        run_name=str(output_cfg.get("run_name", "sampled_reached_state_episode_spline_run")),
        overwrite=bool(args.overwrite or output_cfg.get("overwrite", False)),
        spline_npz_name=str(output_cfg.get("spline_npz_name", "reached_state_adaptive_spline.npz")),
        spline_index_name=str(output_cfg.get("spline_index_name", "reached_state_adaptive_spline_index.parquet")),
        fit_summary_parquet_name=str(
            output_cfg.get("fit_summary_parquet_name", "reached_state_spline_fit_episode_summary.parquet")
        ),
        fit_summary_csv_name=str(
            output_cfg.get("fit_summary_csv_name", "reached_state_spline_fit_episode_summary.csv")
        ),
        resolved_config_name=str(output_cfg.get("resolved_config_name", "resolved_config.yaml")),
    )


def list_episode_dirs(cfg: Config) -> list[Path]:
    episode_root = cfg.dataset_root / "episodes"
    if not episode_root.exists():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")

    episode_dirs = [
        episode_dir
        for episode_dir in sorted(path for path in episode_root.iterdir() if path.is_dir())
        if (episode_dir / "arrays" / cfg.state_array_name).exists()
    ]
    if cfg.max_episodes is not None:
        episode_dirs = episode_dirs[: cfg.max_episodes]
    return episode_dirs


def run_metadata_dir(cfg: Config) -> Path:
    return cfg.dataset_root / "metadata" / "reached_state_episode_spline_constructor" / cfg.run_name


def infer_dt(arrays_dir: Path, cfg: Config, num_frames: int) -> tuple[float, float, str]:
    timestamps_path = arrays_dir / cfg.timestamps_array_name
    if cfg.use_timestamps_if_available and timestamps_path.exists():
        timestamps = np.load(timestamps_path)
        if timestamps.ndim == 1 and timestamps.shape[0] == num_frames:
            diffs = np.diff(timestamps.astype(np.float64))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
            if diffs.size > 0:
                dt = float(np.median(diffs))
                return dt, 1.0 / dt, "timestamps"

    fps = float(cfg.fallback_fps)
    if fps <= 0.0:
        raise ValueError("fallback_fps must be positive")
    return 1.0 / fps, fps, "fallback_fps"


def ensure_odd_window(window: int) -> int:
    window = max(1, int(window))
    return window if window % 2 == 1 else window + 1


def smooth_state(state: np.ndarray, cfg: Config) -> np.ndarray:
    if not cfg.smoothing_enabled:
        return state.astype(np.float64, copy=False)
    if cfg.smoothing_method != "savgol":
        raise ValueError(f"Unsupported smoothing method: {cfg.smoothing_method!r}")

    num_frames = int(state.shape[0])
    window = ensure_odd_window(cfg.savgol_window)
    if window > num_frames:
        window = num_frames if num_frames % 2 == 1 else max(1, num_frames - 1)
    min_window = cfg.savgol_polyorder + 2
    if min_window % 2 == 0:
        min_window += 1
    if window < min_window or window <= cfg.savgol_polyorder:
        return state.astype(np.float64, copy=False)

    return savgol_filter(
        state.astype(np.float64, copy=False),
        window_length=window,
        polyorder=cfg.savgol_polyorder,
        axis=0,
        mode="interp",
    ).astype(np.float64, copy=False)


def compute_derivatives(signal: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_order = 2 if signal.shape[0] >= 3 else 1
    velocity = np.gradient(signal, dt, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, dt, axis=0, edge_order=edge_order)
    jerk = np.gradient(acceleration, dt, axis=0, edge_order=edge_order)
    return velocity.astype(np.float64, copy=False), acceleration.astype(np.float64, copy=False), jerk.astype(
        np.float64, copy=False
    )


def compute_episode_signals(state: np.ndarray, dt: float, cfg: Config) -> dict[str, np.ndarray]:
    required_signals = required_signal_names(cfg.enabled_error_signals)
    q_smooth = smooth_state(state, cfg)
    result: dict[str, np.ndarray] = {"position": q_smooth.astype(np.float32, copy=False)}
    if "velocity" not in required_signals:
        return result

    edge_order = 2 if q_smooth.shape[0] >= 3 else 1
    velocity = np.gradient(q_smooth, dt, axis=0, edge_order=edge_order).astype(np.float64, copy=False)
    result["velocity"] = velocity.astype(np.float32, copy=False)
    if "acceleration" not in required_signals:
        return result

    acceleration = np.gradient(velocity, dt, axis=0, edge_order=edge_order).astype(np.float64, copy=False)
    result["acceleration"] = acceleration.astype(np.float32, copy=False)
    if "jerk" not in required_signals:
        return result

    jerk = np.gradient(acceleration, dt, axis=0, edge_order=edge_order).astype(np.float64, copy=False)
    result["jerk"] = jerk.astype(np.float32, copy=False)
    return result


def build_normalization_stats(episode_dirs: list[Path], cfg: Config) -> dict[str, dict[str, np.ndarray]]:
    required_signals = required_signal_names(cfg.enabled_error_signals)
    signal_buffers = {signal_name: [] for signal_name in required_signals}
    episode_rows: list[dict[str, Any]] = []

    for episode_dir in tqdm(episode_dirs, desc="Normalization episodes", unit="episode", dynamic_ncols=True):
        arrays_dir = episode_dir / "arrays"
        state = np.load(arrays_dir / cfg.state_array_name).astype(np.float32, copy=False)
        if state.ndim != 2:
            raise ValueError(f"{episode_dir.name}: expected 2D state array, got shape {state.shape}")
        if state.shape[0] < 2:
            raise ValueError(f"{episode_dir.name}: need at least 2 frames, got {state.shape[0]}")

        dt, fps, dt_source = infer_dt(arrays_dir, cfg, int(state.shape[0]))
        signals = compute_episode_signals(state, dt, cfg)
        for signal_name in required_signals:
            signal_buffers[signal_name].append(signals[signal_name])
        episode_rows.append(
            {
                "episode_uid": episode_dir.name,
                "num_frames": int(state.shape[0]),
                "fps": float(fps),
                "dt": float(dt),
                "dt_source": dt_source,
            }
        )

    stats: dict[str, dict[str, np.ndarray]] = {}
    distribution_rows: list[dict[str, Any]] = []
    for signal_name in required_signals:
        values = np.concatenate(signal_buffers[signal_name], axis=0).astype(np.float32, copy=False)
        requested_percentiles = np.percentile(
            values,
            [
                0.0,
                1.0,
                cfg.normalization_lower_percentile,
                10.0,
                25.0,
                50.0,
                75.0,
                90.0,
                cfg.normalization_upper_percentile,
                99.0,
                100.0,
            ],
            axis=0,
        ).astype(np.float64)
        if cfg.normalization_center_method == "median":
            center = requested_percentiles[5]
        elif cfg.normalization_center_method == "mean":
            center = values.mean(axis=0, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported normalization center method: {cfg.normalization_center_method!r}")

        scale = requested_percentiles[8] - requested_percentiles[2]
        scale = np.maximum(scale, cfg.normalization_min_scale)
        stats[signal_name] = {
            "center": center.astype(np.float64, copy=False),
            "scale": scale.astype(np.float64, copy=False),
            "lower_percentile": requested_percentiles[2].astype(np.float64, copy=False),
            "median": requested_percentiles[5].astype(np.float64, copy=False),
            "upper_percentile": requested_percentiles[8].astype(np.float64, copy=False),
        }
        for dim_index in range(values.shape[1]):
            distribution_rows.append(
                {
                    "signal_type": signal_name,
                    "dimension_index": int(dim_index),
                    "feature_name": f"dim_{dim_index:02d}",
                    "min": float(requested_percentiles[0, dim_index]),
                    "p1": float(requested_percentiles[1, dim_index]),
                    "p5": float(requested_percentiles[2, dim_index]),
                    "p10": float(requested_percentiles[3, dim_index]),
                    "p25": float(requested_percentiles[4, dim_index]),
                    "median": float(requested_percentiles[5, dim_index]),
                    "p75": float(requested_percentiles[6, dim_index]),
                    "p90": float(requested_percentiles[7, dim_index]),
                    "p95": float(requested_percentiles[8, dim_index]),
                    "p99": float(requested_percentiles[9, dim_index]),
                    "max": float(requested_percentiles[10, dim_index]),
                    "center": float(center[dim_index]),
                    "scale": float(scale[dim_index]),
                }
            )

    metadata_dir = run_metadata_dir(cfg)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {}
    summary_payload: dict[str, Any] = {
        "dataset_root": str(cfg.dataset_root),
        "num_episodes_used": len(episode_rows),
        "num_dimensions": int(next(iter(stats.values()))["scale"].shape[0]),
        "enabled_error_signals": list(cfg.enabled_error_signals),
        "required_signals": list(required_signals),
        "signals": {},
        "episodes": episode_rows,
    }
    for signal_name, signal_stats in stats.items():
        for key, value in signal_stats.items():
            npz_payload[f"{signal_name}_{key}"] = value.astype(np.float64, copy=False)
        summary_payload["signals"][signal_name] = {
            "center_method": cfg.normalization_center_method,
            "lower_percentile": cfg.normalization_lower_percentile,
            "upper_percentile": cfg.normalization_upper_percentile,
            "min_scale": cfg.normalization_min_scale,
            "scale_min": float(np.min(signal_stats["scale"])),
            "scale_median": float(np.median(signal_stats["scale"])),
            "scale_max": float(np.max(signal_stats["scale"])),
        }

    np.savez_compressed(metadata_dir / cfg.normalization_stats_npz_name, **npz_payload)
    (metadata_dir / cfg.normalization_stats_json_name).write_text(
        json.dumps(summary_payload, indent=2),
        encoding="utf-8",
    )
    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(metadata_dir / cfg.normalization_stats_table_csv_name, index=False)
    distribution_df.to_parquet(metadata_dir / cfg.normalization_stats_table_parquet_name, index=False)

    return stats


def initial_internal_indices(num_frames: int, count: int, degree: int) -> list[int]:
    max_internal = max(0, num_frames - degree - 1)
    count = min(max(0, int(count)), max_internal)
    if count == 0 or num_frames <= 2:
        return []
    values = np.linspace(1, num_frames - 2, count + 2, dtype=np.float64)[1:-1]
    return sorted({int(round(value)) for value in values if 0 < int(round(value)) < num_frames - 1})


def make_clamped_knots(u: np.ndarray, internal_indices: list[int], degree: int) -> np.ndarray:
    internal = [float(u[index]) for index in sorted(set(internal_indices)) if 0 < index < len(u) - 1]
    return np.asarray(([0.0] * (degree + 1)) + internal + ([1.0] * (degree + 1)), dtype=np.float64)


def fit_once(u: np.ndarray, y: np.ndarray, internal_indices: list[int], degree: int, endpoint_weight: float) -> BSpline:
    knots = make_clamped_knots(u, internal_indices, degree)
    weights = np.ones(len(u), dtype=np.float64)
    weights[0] = endpoint_weight
    weights[-1] = endpoint_weight
    return make_lsq_spline(u, y, knots, k=degree, w=weights)


def derivative_or_zero(spline: BSpline, order: int, u: np.ndarray, duration: float) -> np.ndarray:
    if order > int(spline.k):
        return np.zeros((len(u), spline.c.shape[1]), dtype=np.float64)
    return spline.derivative(order)(u).astype(np.float64, copy=False) / (duration**order)


def evaluate_spline_signals(spline: BSpline, u: np.ndarray, duration: float) -> dict[str, np.ndarray]:
    position = spline(u).astype(np.float64, copy=False)
    velocity = derivative_or_zero(spline, 1, u, duration)
    acceleration = derivative_or_zero(spline, 2, u, duration)
    jerk = derivative_or_zero(spline, 3, u, duration)
    return {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
    }


def centered_window_mean(values: np.ndarray, window_size: int) -> np.ndarray:
    num_frames = int(values.shape[0])
    if num_frames == 0:
        return values.astype(np.float64, copy=False)
    width = min(ensure_odd_window(window_size), num_frames if num_frames % 2 == 1 else max(1, num_frames - 1))
    half = width // 2
    starts = np.maximum(0, np.arange(num_frames) - half)
    ends = np.minimum(num_frames, np.arange(num_frames) + half + 1)
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values.astype(np.float64, copy=False), axis=0)]
    )
    window_sums = cumulative[ends] - cumulative[starts]
    counts = (ends - starts).astype(np.float64)[:, None]
    return window_sums / counts


def compute_local_violation_metrics(
    target_signals: dict[str, np.ndarray],
    predicted_signals: dict[str, np.ndarray],
    normalization_stats: dict[str, dict[str, np.ndarray]],
    cfg: Config,
) -> dict[str, Any]:
    epsilon_by_signal = {
        "position": cfg.epsilon_position,
        "velocity": cfg.epsilon_velocity,
        "acceleration": cfg.epsilon_acceleration,
        "jerk": cfg.epsilon_jerk,
    }

    ratios_by_signal: list[np.ndarray] = []
    enabled_signal_names: list[str] = []
    signal_max_ratio: dict[str, float] = {}
    signal_max_rmse: dict[str, float] = {}
    required_signals = required_signal_names(cfg.enabled_error_signals)

    for signal_name in required_signals:
        residual = (
            target_signals[signal_name].astype(np.float64, copy=False)
            - predicted_signals[signal_name].astype(np.float64, copy=False)
        ) / normalization_stats[signal_name]["scale"][None, :]
        if signal_name in cfg.enabled_error_signals:
            local_rmse = np.sqrt(centered_window_mean(residual * residual, cfg.local_window_size))
            signal_max_rmse[signal_name] = float(np.max(local_rmse))
            ratios = local_rmse / epsilon_by_signal[signal_name]
            signal_max_ratio[signal_name] = float(np.max(ratios))
            ratios_by_signal.append(ratios.astype(np.float64, copy=False))
            enabled_signal_names.append(signal_name)
        else:
            signal_max_rmse[signal_name] = float("nan")
            signal_max_ratio[signal_name] = float("nan")

    if not ratios_by_signal:
        raise ValueError("No enabled error signals were available for local violation metrics.")

    for signal_name in SIGNAL_NAMES:
        signal_max_rmse.setdefault(signal_name, float("nan"))
        signal_max_ratio.setdefault(signal_name, float("nan"))

    ratio_stack = np.stack(ratios_by_signal, axis=0)
    signal_argmax = np.argmax(ratio_stack, axis=0).astype(np.int64)
    joint_time_max = np.max(ratio_stack, axis=0)
    overall_flat_index = int(np.argmax(joint_time_max))
    worst_frame, worst_joint = np.unravel_index(overall_flat_index, joint_time_max.shape)
    worst_signal_name = enabled_signal_names[int(signal_argmax[worst_frame, worst_joint])]
    worst_signal_index = SIGNAL_NAMES.index(worst_signal_name)

    per_frame_ratio = np.max(joint_time_max, axis=1)
    per_frame_worst_joint = np.argmax(joint_time_max, axis=1).astype(np.int64)
    per_frame_worst_signal_local = signal_argmax[np.arange(joint_time_max.shape[0]), per_frame_worst_joint].astype(
        np.int64
    )
    per_frame_worst_signal = np.asarray(
        [SIGNAL_NAMES.index(enabled_signal_names[int(index)]) for index in per_frame_worst_signal_local],
        dtype=np.int64,
    )

    return {
        "overall_max_ratio": float(joint_time_max[worst_frame, worst_joint]),
        "worst_frame": int(worst_frame),
        "worst_joint": int(worst_joint),
        "worst_signal": worst_signal_name,
        "worst_signal_index": worst_signal_index,
        "signal_max_ratio": signal_max_ratio,
        "signal_max_rmse": signal_max_rmse,
        "per_frame_ratio": per_frame_ratio.astype(np.float32, copy=False),
        "per_frame_worst_joint": per_frame_worst_joint.astype(np.int64, copy=False),
        "per_frame_worst_signal": per_frame_worst_signal.astype(np.int64, copy=False),
    }


def choose_next_knot_candidate(
    per_frame_ratio: np.ndarray,
    per_frame_worst_joint: np.ndarray,
    per_frame_worst_signal: np.ndarray,
    existing_internal_indices: list[int],
    min_spacing_frames: int,
) -> dict[str, Any] | None:
    existing = set(int(value) for value in existing_internal_indices)
    candidate_order = np.argsort(per_frame_ratio)[::-1]
    for candidate_index in candidate_order:
        frame = int(candidate_index)
        if frame <= 0 or frame >= len(per_frame_ratio) - 1:
            continue
        if frame in existing:
            continue
        if any(abs(frame - old) < min_spacing_frames for old in existing):
            continue
        signal_index = int(per_frame_worst_signal[frame])
        return {
            "frame_index": frame,
            "joint_index": int(per_frame_worst_joint[frame]),
            "signal_name": SIGNAL_NAMES[signal_index],
            "signal_index": signal_index,
            "ratio": float(per_frame_ratio[frame]),
        }
    return None


def compute_episode_observation_rmse_distribution(
    target_position: np.ndarray,
    predicted_position: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    residual = target_position.astype(np.float64, copy=False) - predicted_position.astype(np.float64, copy=False)
    per_frame_rmse = np.sqrt(np.mean(residual * residual, axis=1, dtype=np.float64)).astype(np.float64, copy=False)
    requested = np.percentile(
        per_frame_rmse,
        [percentile for _, percentile in EPISODE_RMSE_DISTRIBUTION_PERCENTILES],
    ).astype(np.float64, copy=False)
    summary = {
        label: float(value)
        for (label, _), value in zip(EPISODE_RMSE_DISTRIBUTION_PERCENTILES, requested, strict=True)
    }
    return per_frame_rmse, summary


def adaptive_fit_state_spline(
    episode_uid: str,
    target_position: np.ndarray,
    target_signals: dict[str, np.ndarray],
    normalization_stats: dict[str, dict[str, np.ndarray]],
    dt: float,
    cfg: Config,
) -> dict[str, Any]:
    num_frames = int(target_position.shape[0])
    if num_frames < 2:
        raise ValueError(f"Need at least 2 frames to fit a spline, got {num_frames}")

    degree = min(cfg.degree, num_frames - 1)
    if degree < 1:
        raise ValueError(f"Need at least 2 frames to fit a spline, got {num_frames}")

    max_internal_possible = max(0, num_frames - degree - 1)
    max_internal = min(cfg.max_internal_knots, max_internal_possible)
    if cfg.max_control_points is not None:
        max_internal = min(max_internal, max(0, cfg.max_control_points - degree - 1))

    u = np.linspace(0.0, 1.0, num_frames, dtype=np.float64)
    duration = max((num_frames - 1) * dt, np.finfo(np.float64).eps)
    internal_indices = initial_internal_indices(num_frames, cfg.initial_internal_knots, degree)[:max_internal]
    knot_history: list[dict[str, Any]] = []
    last_exception: str | None = None
    last_success: dict[str, Any] | None = None
    max_insertions_possible = max(0, max_internal - len(internal_indices))
    total_fit_iterations = min(cfg.max_iterations, max_insertions_possible) + 1
    episode_progress = tqdm(
        total=total_fit_iterations,
        desc=f"Adaptive fit: {episode_uid}",
        unit="iter",
        leave=False,
        dynamic_ncols=True,
    )

    try:
        while True:
            try:
                spline = fit_once(
                    u,
                    target_position.astype(np.float64, copy=False),
                    internal_indices,
                    degree,
                    cfg.endpoint_weight,
                )
            except Exception as exc:
                last_exception = repr(exc)
                if last_success is None:
                    raise
                final_result = dict(last_success)
                final_result["status"] = "fit_failed_after_last_success"
                final_result["last_exception"] = last_exception
                final_result["knot_history"] = knot_history
                episode_progress.set_postfix_str(
                    f"status={final_result['status']} internal={len(internal_indices)}/{max_internal}"
                )
                return final_result

            predicted_signals = evaluate_spline_signals(spline, u, duration)
            metrics = compute_local_violation_metrics(target_signals, predicted_signals, normalization_stats, cfg)
            last_success = {
                "spline": spline,
                "u": u,
                "duration_seconds": duration,
                "degree": int(degree),
                "internal_indices": np.asarray(sorted(internal_indices), dtype=np.int64),
                "predicted_signals": predicted_signals,
                "metrics": metrics,
                "status": "in_progress",
                "last_exception": None,
            }
            episode_progress.update(1)
            episode_progress.set_postfix(
                internal=f"{len(internal_indices)}/{max_internal}",
                ratio=format_float(metrics["overall_max_ratio"]),
                worst=metrics["worst_signal"],
            )

            if metrics["overall_max_ratio"] <= 1.0:
                final_result = dict(last_success)
                final_result["status"] = "tolerances_met"
                final_result["last_exception"] = last_exception
                final_result["knot_history"] = knot_history
                episode_progress.set_postfix_str(
                    f"status={final_result['status']} internal={len(internal_indices)}/{max_internal} "
                    f"ratio={format_float(metrics['overall_max_ratio'])}"
                )
                return final_result

            if len(internal_indices) >= max_internal:
                final_result = dict(last_success)
                final_result["status"] = "max_internal_knots_reached"
                final_result["last_exception"] = last_exception
                final_result["knot_history"] = knot_history
                episode_progress.set_postfix_str(
                    f"status={final_result['status']} internal={len(internal_indices)}/{max_internal} "
                    f"ratio={format_float(metrics['overall_max_ratio'])}"
                )
                return final_result

            if len(knot_history) >= cfg.max_iterations:
                final_result = dict(last_success)
                final_result["status"] = "max_iterations_reached"
                final_result["last_exception"] = last_exception
                final_result["knot_history"] = knot_history
                episode_progress.set_postfix_str(
                    f"status={final_result['status']} internal={len(internal_indices)}/{max_internal} "
                    f"ratio={format_float(metrics['overall_max_ratio'])}"
                )
                return final_result

            candidate = choose_next_knot_candidate(
                metrics["per_frame_ratio"],
                metrics["per_frame_worst_joint"],
                metrics["per_frame_worst_signal"],
                internal_indices,
                cfg.min_knot_spacing_frames,
            )
            if candidate is None:
                final_result = dict(last_success)
                final_result["status"] = "no_valid_knot_candidate"
                final_result["last_exception"] = last_exception
                final_result["knot_history"] = knot_history
                episode_progress.set_postfix_str(
                    f"status={final_result['status']} internal={len(internal_indices)}/{max_internal} "
                    f"ratio={format_float(metrics['overall_max_ratio'])}"
                )
                return final_result

            internal_indices.append(candidate["frame_index"])
            internal_indices = sorted(set(internal_indices))
            knot_history.append(
                {
                    "iteration": len(knot_history) + 1,
                    "inserted_frame_index": int(candidate["frame_index"]),
                    "inserted_u": float(u[candidate["frame_index"]]),
                    "joint_index": int(candidate["joint_index"]),
                    "reason_signal": candidate["signal_name"],
                    "ratio": float(candidate["ratio"]),
                }
            )
    finally:
        episode_progress.close()


def save_episode_spline_outputs(
    episode_dir: Path,
    cfg: Config,
    fit_result: dict[str, Any],
    target_signals: dict[str, np.ndarray],
    dt: float,
    fps: float,
    dt_source: str,
) -> dict[str, Any]:
    arrays_dir = episode_dir / "arrays"
    spline_npz_path = arrays_dir / cfg.spline_npz_name
    spline_index_path = arrays_dir / cfg.spline_index_name

    spline: BSpline = fit_result["spline"]
    metrics = fit_result["metrics"]
    num_frames = int(target_signals["position"].shape[0])
    frame_indices = np.arange(num_frames, dtype=np.int64)
    signal_name_table = np.asarray(SIGNAL_NAMES, dtype="U32")
    knot_history = fit_result.get("knot_history", [])
    num_control_points = int(np.asarray(spline.c).shape[0])
    num_knots_total = int(len(spline.t))
    num_knot_spans = int(max(0, len(np.unique(spline.t.astype(np.float64))) - 1))
    control_points_per_frame = float(num_control_points / max(num_frames, 1))
    knot_spans_per_frame = float(num_knot_spans / max(num_frames, 1))
    episode_observation_rmse_per_frame = np.full(num_frames, np.nan, dtype=np.float64)
    episode_observation_rmse_summary = {
        label: float("nan") for label, _ in EPISODE_RMSE_DISTRIBUTION_PERCENTILES
    }
    if cfg.report_episode_observation_rmse_distribution:
        episode_observation_rmse_per_frame, episode_observation_rmse_summary = (
            compute_episode_observation_rmse_distribution(
                target_signals["position"],
                fit_result["predicted_signals"]["position"],
            )
        )

    npz_payload: dict[str, np.ndarray] = {
        "global_knots": spline.t.astype(np.float64),
        "global_coefficients": np.asarray(spline.c, dtype=np.float64),
        "global_degree": np.asarray([fit_result["degree"]], dtype=np.int64),
        "frame_indices": frame_indices,
        "frame_to_u": fit_result["u"].astype(np.float64),
        "num_original_frames": np.asarray([num_frames], dtype=np.int64),
        "num_knot_spans": np.asarray([num_knot_spans], dtype=np.int64),
        "duration_seconds": np.asarray([fit_result["duration_seconds"]], dtype=np.float64),
        "dt_seconds": np.asarray([dt], dtype=np.float64),
        "fps": np.asarray([fps], dtype=np.float64),
        "dt_source": np.asarray([dt_source], dtype="U32"),
        "target_array_name": np.asarray([cfg.state_array_name], dtype="U128"),
        "spline_target_mode": np.asarray(["global_reached_state"], dtype="U64"),
        "fit_status": np.asarray([fit_result["status"]], dtype="U64"),
        "signal_names": signal_name_table,
        "required_signals": np.asarray(required_signal_names(cfg.enabled_error_signals), dtype="U32"),
        "enabled_error_signals": np.asarray(cfg.enabled_error_signals, dtype="U32"),
        "internal_knot_frame_indices": fit_result["internal_indices"].astype(np.int64),
        "final_frame_max_violation_ratio": metrics["per_frame_ratio"].astype(np.float32),
        "final_frame_worst_joint_index": metrics["per_frame_worst_joint"].astype(np.int64),
        "final_frame_worst_signal_index": metrics["per_frame_worst_signal"].astype(np.int64),
        "episode_observation_rmse_per_frame": episode_observation_rmse_per_frame.astype(np.float32, copy=False),
        "knot_insertion_frame_indices": np.asarray(
            [item["inserted_frame_index"] for item in knot_history], dtype=np.int64
        ),
        "knot_insertion_u": np.asarray([item["inserted_u"] for item in knot_history], dtype=np.float64),
        "knot_insertion_joint_indices": np.asarray([item["joint_index"] for item in knot_history], dtype=np.int64),
        "knot_insertion_signal_indices": np.asarray(
            [SIGNAL_NAMES.index(item["reason_signal"]) for item in knot_history], dtype=np.int64
        ),
        "knot_insertion_ratios": np.asarray([item["ratio"] for item in knot_history], dtype=np.float64),
    }
    timestamps_path = arrays_dir / cfg.timestamps_array_name
    if timestamps_path.exists():
        timestamps = np.load(timestamps_path)
        if timestamps.ndim == 1 and timestamps.shape[0] == num_frames:
            npz_payload["timestamps"] = timestamps.astype(np.float64, copy=False)

    np.savez_compressed(spline_npz_path, **npz_payload)

    row = {
        "episode_uid": episode_dir.name,
        "target": "global_reached_state = state_65d",
        "target_array_name": cfg.state_array_name,
        "spline_npz_name": cfg.spline_npz_name,
        "num_original_frames": num_frames,
        "target_dim": int(target_signals["position"].shape[1]),
        "degree": int(fit_result["degree"]),
        "num_internal_knots": int(len(fit_result["internal_indices"])),
        "num_knots_total": num_knots_total,
        "num_knot_spans": num_knot_spans,
        "num_control_points": num_control_points,
        "control_points_per_frame": control_points_per_frame,
        "knot_spans_per_frame": knot_spans_per_frame,
        "duration_seconds": float(fit_result["duration_seconds"]),
        "dt_seconds": float(dt),
        "fps": float(fps),
        "dt_source": dt_source,
        "enabled_error_signals": "|".join(cfg.enabled_error_signals),
        "fit_status": fit_result["status"],
        "last_exception": fit_result.get("last_exception"),
        "iterations": int(len(knot_history)),
        "overall_max_ratio": float(metrics["overall_max_ratio"]),
        "worst_frame": int(metrics["worst_frame"]),
        "worst_joint": int(metrics["worst_joint"]),
        "worst_signal": metrics["worst_signal"],
        "max_position_ratio": float(metrics["signal_max_ratio"]["position"]),
        "max_velocity_ratio": float(metrics["signal_max_ratio"]["velocity"]),
        "max_acceleration_ratio": float(metrics["signal_max_ratio"]["acceleration"]),
        "max_jerk_ratio": float(metrics["signal_max_ratio"]["jerk"]),
        "max_position_local_rmse_normalized": float(metrics["signal_max_rmse"]["position"]),
        "max_velocity_local_rmse_normalized": float(metrics["signal_max_rmse"]["velocity"]),
        "max_acceleration_local_rmse_normalized": float(metrics["signal_max_rmse"]["acceleration"]),
        "max_jerk_local_rmse_normalized": float(metrics["signal_max_rmse"]["jerk"]),
        "episode_observation_rmse_distribution_enabled": bool(cfg.report_episode_observation_rmse_distribution),
        "knot_history_json": json.dumps(knot_history),
    }
    for label, _ in EPISODE_RMSE_DISTRIBUTION_PERCENTILES:
        row[f"episode_observation_rmse_{label}"] = float(episode_observation_rmse_summary[label])
    pd.DataFrame([row]).to_parquet(spline_index_path, index=False)
    return row


def build_episode_fit_message(row: dict[str, Any], cfg: Config) -> str:
    fit_status = str(row.get("fit_status", "unknown"))
    episode_uid = str(row.get("episode_uid", "<unknown>"))
    if fit_status == "failed":
        return f"[failed] {episode_uid} | error={row.get('error_message', 'unknown')}"
    if fit_status == "skipped_existing":
        return f"[skipped_existing] {episode_uid}"

    headline = (
        f"[{fit_status}] {episode_uid} | "
        f"frames={row.get('num_original_frames', 'n/a')} | "
        f"ctrl={row.get('num_control_points', 'n/a')} | "
        f"spans={row.get('num_knot_spans', 'n/a')} | "
        f"internal_knots={row.get('num_internal_knots', 'n/a')}/{cfg.max_internal_knots} | "
        f"degree={row.get('degree', 'n/a')} | "
        f"ctrl/frame={format_float(row.get('control_points_per_frame'))} | "
        f"spans/frame={format_float(row.get('knot_spans_per_frame'))} | "
        f"required={','.join(required_signal_names(cfg.enabled_error_signals))} | "
        f"enabled={','.join(cfg.enabled_error_signals)} | "
        f"max_ratio={format_float(row.get('overall_max_ratio'))} | "
        f"worst={row.get('worst_signal', 'n/a')}@frame{row.get('worst_frame', 'n/a')}/joint{row.get('worst_joint', 'n/a')} | "
        + " | ".join(
            f"{signal_name}_rmse_n={format_float(row.get(f'max_{signal_name}_local_rmse_normalized'))}/"
            f"{format_float(getattr(cfg, f'epsilon_{signal_name}'))}"
            for signal_name in cfg.enabled_error_signals
        )
    )
    if not row.get("episode_observation_rmse_distribution_enabled", False):
        return headline

    rmse_distribution = " | ".join(
        f"{label}={format_float(row.get(f'episode_observation_rmse_{label}'))}"
        for label, _ in EPISODE_RMSE_DISTRIBUTION_PERCENTILES
    )
    return headline + "\n" + f"  observation_rmse/frame: {rmse_distribution}"


def print_run_summary(summary_rows: list[dict[str, Any]], cfg: Config) -> None:
    processed_rows = [row for row in summary_rows if row.get("fit_status") not in {"failed", "skipped_existing"}]
    failed_rows = [row for row in summary_rows if row.get("fit_status") == "failed"]
    skipped_rows = [row for row in summary_rows if row.get("fit_status") == "skipped_existing"]

    print(f"dataset_root         : {cfg.dataset_root}")
    print(f"run_metadata_dir     : {run_metadata_dir(cfg)}")
    print(f"spline_npz_name      : {cfg.spline_npz_name}")
    print(f"spline_index_name    : {cfg.spline_index_name}")
    print(f"episodes_total       : {len(summary_rows)}")
    print(f"episodes_processed   : {len(processed_rows)}")
    print(f"episodes_skipped     : {len(skipped_rows)}")
    print(f"episodes_failed      : {len(failed_rows)}")
    print(
        "epsilon_targets      : "
        + ", ".join(
            f"{signal_name}={format_float(getattr(cfg, f'epsilon_{signal_name}'))}"
            for signal_name in cfg.enabled_error_signals
        )
    )
    print(f"required_signals     : {', '.join(required_signal_names(cfg.enabled_error_signals))}")
    print(f"enabled_signals      : {', '.join(cfg.enabled_error_signals)}")
    print(
        "knot_limits          : "
        f"initial_internal={cfg.initial_internal_knots}, "
        f"max_internal={cfg.max_internal_knots}, "
        f"max_iterations={cfg.max_iterations}, "
        f"min_spacing_frames={cfg.min_knot_spacing_frames}"
    )

    if not processed_rows:
        return

    processed_df = pd.DataFrame(processed_rows)
    print(
        "aggregate_fit        : "
        f"frames_median={format_float(processed_df['num_original_frames'].median(), 1)}, "
        f"ctrl_median={format_float(processed_df['num_control_points'].median(), 1)}, "
        f"spans_median={format_float(processed_df['num_knot_spans'].median(), 1)}, "
        f"overall_ratio_max={format_float(processed_df['overall_max_ratio'].max())}"
    )
    print(
        "aggregate_rmse_n     : "
        + ", ".join(
            f"{signal_name}_max={format_float(processed_df[f'max_{signal_name}_local_rmse_normalized'].max())}"
            for signal_name in cfg.enabled_error_signals
        )
    )
    print(
        "aggregate_ratio_sig  : "
        + ", ".join(
            f"{signal_name}_max={format_float(processed_df[f'max_{signal_name}_ratio'].max())}"
            for signal_name in cfg.enabled_error_signals
        )
    )


def process_episode(
    episode_dir: Path,
    cfg: Config,
    normalization_stats: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    arrays_dir = episode_dir / "arrays"
    spline_npz_path = arrays_dir / cfg.spline_npz_name
    spline_index_path = arrays_dir / cfg.spline_index_name
    if spline_npz_path.exists() and spline_index_path.exists() and not cfg.overwrite:
        existing_row = {
            "episode_uid": episode_dir.name,
            "fit_status": "skipped_existing",
            "spline_npz_name": cfg.spline_npz_name,
            "spline_index_name": cfg.spline_index_name,
        }
        return existing_row

    state = np.load(arrays_dir / cfg.state_array_name).astype(np.float32, copy=False)
    if state.ndim != 2:
        raise ValueError(f"{episode_dir.name}: expected 2D state array, got shape {state.shape}")
    if state.shape[0] < 2:
        raise ValueError(f"{episode_dir.name}: need at least 2 state frames, got {state.shape[0]}")

    dt, fps, dt_source = infer_dt(arrays_dir, cfg, int(state.shape[0]))
    target_signals = compute_episode_signals(state, dt, cfg)
    fit_result = adaptive_fit_state_spline(
        episode_uid=episode_dir.name,
        target_position=target_signals["position"].astype(np.float64, copy=False),
        target_signals={key: value.astype(np.float64, copy=False) for key, value in target_signals.items()},
        normalization_stats=normalization_stats,
        dt=dt,
        cfg=cfg,
    )
    return save_episode_spline_outputs(episode_dir, cfg, fit_result, target_signals, dt, fps, dt_source)


def write_run_outputs(cfg: Config, summary_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    metadata_dir = run_metadata_dir(cfg)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_parquet(metadata_dir / cfg.fit_summary_parquet_name, index=False)
    summary_df.to_csv(metadata_dir / cfg.fit_summary_csv_name, index=False)

    resolved = {
        "dataset_root": str(cfg.dataset_root),
        "num_workers": cfg.num_workers,
        "max_episodes": cfg.max_episodes,
        "state_array_name": cfg.state_array_name,
        "timestamps_array_name": cfg.timestamps_array_name,
        "fallback_fps": cfg.fallback_fps,
        "use_timestamps_if_available": cfg.use_timestamps_if_available,
        "smoothing": {
            "enabled": cfg.smoothing_enabled,
            "method": cfg.smoothing_method,
            "savgol_window": cfg.savgol_window,
            "savgol_polyorder": cfg.savgol_polyorder,
        },
        "normalization": {
            "center_method": cfg.normalization_center_method,
            "lower_percentile": cfg.normalization_lower_percentile,
            "upper_percentile": cfg.normalization_upper_percentile,
            "min_scale": cfg.normalization_min_scale,
            "stats_npz_name": cfg.normalization_stats_npz_name,
            "stats_json_name": cfg.normalization_stats_json_name,
            "stats_table_csv_name": cfg.normalization_stats_table_csv_name,
            "stats_table_parquet_name": cfg.normalization_stats_table_parquet_name,
            "required_signals": list(required_signal_names(cfg.enabled_error_signals)),
        },
        "fit": {
            "degree": cfg.degree,
            "enabled_error_signals": list(cfg.enabled_error_signals),
            "initial_internal_knots": cfg.initial_internal_knots,
            "max_internal_knots": cfg.max_internal_knots,
            "max_control_points": cfg.max_control_points,
            "max_iterations": cfg.max_iterations,
            "min_knot_spacing_frames": cfg.min_knot_spacing_frames,
            "local_window_size": cfg.local_window_size,
            "endpoint_weight": cfg.endpoint_weight,
            "epsilon_position": cfg.epsilon_position,
            "epsilon_velocity": cfg.epsilon_velocity,
            "epsilon_acceleration": cfg.epsilon_acceleration,
            "epsilon_jerk": cfg.epsilon_jerk,
            "report_episode_observation_rmse_distribution": cfg.report_episode_observation_rmse_distribution,
        },
        "output": {
            "run_name": cfg.run_name,
            "overwrite": cfg.overwrite,
            "spline_npz_name": cfg.spline_npz_name,
            "spline_index_name": cfg.spline_index_name,
            "fit_summary_parquet_name": cfg.fit_summary_parquet_name,
            "fit_summary_csv_name": cfg.fit_summary_csv_name,
        },
        "cli_overrides": {
            "config": str(args.config),
            "dataset_root": None if args.dataset_root is None else str(args.dataset_root),
            "max_episodes": args.max_episodes,
            "overwrite": bool(args.overwrite),
        },
    }
    (metadata_dir / cfg.resolved_config_name).write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    episode_dirs = list_episode_dirs(cfg)
    if not episode_dirs:
        raise RuntimeError(f"No eligible episodes found under {cfg.dataset_root}")

    normalization_stats = build_normalization_stats(episode_dirs, cfg)

    summary_rows: list[dict[str, Any]] = []
    for episode_dir in tqdm(episode_dirs, desc="Fit reached-state splines", unit="episode", dynamic_ncols=True):
        try:
            row = process_episode(episode_dir, cfg, normalization_stats)
            summary_rows.append(row)
            tqdm.write(build_episode_fit_message(row, cfg))
        except Exception as exc:
            row = (
                {
                    "episode_uid": episode_dir.name,
                    "fit_status": "failed",
                    "error_message": repr(exc),
                }
            )
            summary_rows.append(row)
            tqdm.write(build_episode_fit_message(row, cfg))

    write_run_outputs(cfg, summary_rows, args)
    print_run_summary(summary_rows, cfg)


if __name__ == "__main__":
    main()
