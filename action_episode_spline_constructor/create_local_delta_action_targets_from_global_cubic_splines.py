from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import BSpline
from tqdm.auto import tqdm


ProgressCallback = Callable[[dict[str, Any]], None]
EXPECTED_GLOBAL_DEGREE = 3


@dataclass(frozen=True)
class LocalTargetConfig:
    dataset_root: Path
    num_workers: int
    max_episodes: int | None
    target_knot_spans: int
    sample_stride: int
    include_truncated_horizons: bool
    progress_update_every_samples: int
    global_spline_npz_name: str
    action_array_name: str
    state_array_name: str
    output_dir: Path | None
    output_prefix: str
    overwrite: bool
    restriction_validation_points: int
    restriction_validation_tolerance: float

    @property
    def output_tag(self) -> str:
        return f"{self.output_prefix}_knotspans{self.target_knot_spans}"

    @property
    def output_npz_name(self) -> str:
        return f"{self.output_tag}.npz"

    @property
    def output_index_name(self) -> str:
        return f"{self.output_tag}_index.parquet"

    @property
    def output_episode_summary_name(self) -> str:
        return f"{self.output_tag}_episode_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create exact local delta-action spline targets by restricting saved global cubic action splines."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration path.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Dataset root override.")
    parser.add_argument("--num-workers", type=int, default=None, help="Parallel episode worker count.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum eligible episodes to process.")
    parser.add_argument("--target-knot-spans", type=int, default=None, help="Future global knot spans per target.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional external output root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing per-episode outputs.")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> LocalTargetConfig:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    input_cfg = raw.get("input", {})
    sampling_cfg = raw.get("sampling", {})
    output_cfg = raw.get("output", {})

    dataset_root_value = args.dataset_root or raw.get("dataset_root")
    if dataset_root_value is None:
        raise ValueError("dataset_root is required in the config or --dataset-root.")

    target_knot_spans = int(
        args.target_knot_spans
        if args.target_knot_spans is not None
        else sampling_cfg.get("target_knot_spans", 10)
    )
    if target_knot_spans < 1:
        raise ValueError("sampling.target_knot_spans must be at least 1.")

    output_dir_value = args.output_dir or output_cfg.get("output_dir")
    validation_points = int(raw.get("validation", {}).get("restriction_validation_points", 5))
    if validation_points < 2:
        raise ValueError("validation.restriction_validation_points must be at least 2.")

    max_episodes_value = args.max_episodes if args.max_episodes is not None else raw.get("max_episodes")
    return LocalTargetConfig(
        dataset_root=Path(dataset_root_value),
        num_workers=max(1, int(args.num_workers if args.num_workers is not None else raw.get("num_workers", 1))),
        max_episodes=int(max_episodes_value) if max_episodes_value is not None else None,
        target_knot_spans=target_knot_spans,
        sample_stride=max(1, int(sampling_cfg.get("sample_stride", 1))),
        include_truncated_horizons=bool(sampling_cfg.get("include_truncated_horizons", False)),
        progress_update_every_samples=max(1, int(sampling_cfg.get("progress_update_every_samples", 25))),
        global_spline_npz_name=str(input_cfg.get("global_spline_npz_name", "global_action_spline.npz")),
        action_array_name=str(input_cfg.get("action_array_name", "action_65d.npy")),
        state_array_name=str(input_cfg.get("state_array_name", "state_65d.npy")),
        output_dir=Path(output_dir_value) if output_dir_value else None,
        output_prefix=str(output_cfg.get("output_prefix", "local_delta_action_cubic")),
        overwrite=bool(args.overwrite or output_cfg.get("overwrite", False)),
        restriction_validation_points=validation_points,
        restriction_validation_tolerance=float(
            raw.get("validation", {}).get("restriction_validation_tolerance", 1e-8)
        ),
    )


def list_episode_dirs(cfg: LocalTargetConfig) -> list[Path]:
    episode_root = cfg.dataset_root / "episodes"
    if not episode_root.exists():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")

    required_names = (cfg.action_array_name, cfg.state_array_name, cfg.global_spline_npz_name)
    episodes = [
        episode_dir
        for episode_dir in sorted(path for path in episode_root.iterdir() if path.is_dir())
        if all((episode_dir / "arrays" / name).exists() for name in required_names)
    ]
    if cfg.max_episodes is not None:
        if cfg.max_episodes < 1:
            raise ValueError("max_episodes must be at least 1 when provided.")
        episodes = episodes[: cfg.max_episodes]
    return episodes


def output_arrays_dir_for_episode(episode_dir: Path, cfg: LocalTargetConfig) -> Path:
    if cfg.output_dir is None:
        return episode_dir / "arrays"
    return cfg.output_dir / "episodes" / episode_dir.name / "arrays"


def summary_root(cfg: LocalTargetConfig) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir / f"{cfg.output_tag}_run"
    return cfg.dataset_root / "metadata" / f"{cfg.output_tag}_run"


def unique_knots(knots: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(knots, dtype=np.float64))
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("Global spline must contain at least two finite unique knots.")
    return values


def knot_multiplicity(knots: np.ndarray, value: float, tolerance: float = 1e-12) -> int:
    return int(np.sum(np.isclose(knots, value, rtol=0.0, atol=tolerance)))


def restrict_bspline_segment(
    global_knots: np.ndarray,
    global_coefficients: np.ndarray,
    degree: int,
    u_start: float,
    u_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict exactly, then map the global interval [u_start, u_end] to local [0, 1]."""
    if not np.isfinite(u_start) or not np.isfinite(u_end) or u_end <= u_start:
        raise ValueError(f"Invalid restriction interval [{u_start}, {u_end}].")

    spline = BSpline(
        np.asarray(global_knots, dtype=np.float64),
        np.asarray(global_coefficients, dtype=np.float64),
        int(degree),
    )
    for boundary in (float(u_start), float(u_end)):
        required_insertions = (degree + 1) - knot_multiplicity(spline.t, boundary)
        if required_insertions > 0:
            spline = spline.insert_knot(boundary, m=required_insertions)

    knots = np.asarray(spline.t, dtype=np.float64)
    coefficients = np.asarray(spline.c, dtype=np.float64)
    tolerance = 1e-10
    local_global_knots = np.concatenate(
        [
            np.full(degree + 1, u_start, dtype=np.float64),
            knots[(knots > u_start + tolerance) & (knots < u_end - tolerance)],
            np.full(degree + 1, u_end, dtype=np.float64),
        ]
    )
    expected_control_points = local_global_knots.size - degree - 1
    control_indices = np.asarray(
        [
            index
            for index in range(coefficients.shape[0])
            if knots[index] >= u_start - tolerance and knots[index + degree + 1] <= u_end + tolerance
        ],
        dtype=np.int64,
    )
    if control_indices.size != expected_control_points:
        raise RuntimeError(
            "Restricted control point count mismatch: "
            f"got {control_indices.size}, expected {expected_control_points}."
        )

    local_knots = (local_global_knots - u_start) / (u_end - u_start)
    local_knots[: degree + 1] = 0.0
    local_knots[-(degree + 1) :] = 1.0
    return local_knots.astype(np.float64), coefficients[control_indices].astype(np.float64)


def target_horizon(
    frame_to_u: np.ndarray,
    unique_global_knots: np.ndarray,
    start_frame: int,
    target_knot_spans: int,
    include_truncated_horizons: bool,
) -> tuple[int, int, float, int, float] | None:
    """Return start/end span and final in-span dataset frame for one local target."""
    u_start = float(frame_to_u[start_frame])
    start_span = int(np.searchsorted(unique_global_knots, u_start, side="right") - 1)
    start_span = max(0, min(start_span, unique_global_knots.size - 2))
    requested_end_span = start_span + target_knot_spans
    if requested_end_span >= unique_global_knots.size:
        if not include_truncated_horizons:
            return None
        end_span = unique_global_knots.size - 1
    else:
        end_span = requested_end_span

    horizon_boundary_u = float(unique_global_knots[end_span])
    end_frame = int(np.searchsorted(frame_to_u, horizon_boundary_u, side="right") - 1)
    end_frame = max(0, min(end_frame, frame_to_u.size - 1))
    if end_frame <= start_frame:
        return None
    return start_span, end_span, horizon_boundary_u, end_frame, float(frame_to_u[end_frame])


def validate_global_payload(
    episode_uid: str,
    payload: Any,
    num_frames: int,
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    required_keys = ("global_knots", "global_coefficients", "global_degree", "frame_to_u")
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"{episode_uid}: global spline is missing keys: {missing}")

    knots = np.asarray(payload["global_knots"], dtype=np.float64)
    coefficients = np.asarray(payload["global_coefficients"], dtype=np.float64)
    degree = int(np.asarray(payload["global_degree"]).reshape(-1)[0])
    frame_to_u = np.asarray(payload["frame_to_u"], dtype=np.float64)
    if degree != EXPECTED_GLOBAL_DEGREE:
        raise ValueError(f"{episode_uid}: expected saved global degree {EXPECTED_GLOBAL_DEGREE}, found {degree}.")
    if frame_to_u.shape != (num_frames,) or not np.all(np.isfinite(frame_to_u)):
        raise ValueError(f"{episode_uid}: frame_to_u must be finite with shape ({num_frames},), got {frame_to_u.shape}.")
    if np.any(np.diff(frame_to_u) <= 0):
        raise ValueError(f"{episode_uid}: frame_to_u must be strictly increasing.")
    if coefficients.ndim != 2 or coefficients.shape[1] != target_dim:
        raise ValueError(
            f"{episode_uid}: global coefficients must have shape (control_points, {target_dim}), got {coefficients.shape}."
        )
    if knots.ndim != 1 or knots.size != coefficients.shape[0] + degree + 1:
        raise ValueError(f"{episode_uid}: invalid global knot/control-point dimensions.")
    if frame_to_u[0] < knots[degree] - 1e-12 or frame_to_u[-1] > knots[-degree - 1] + 1e-12:
        raise ValueError(f"{episode_uid}: frame parameters are outside the global spline domain.")
    return knots, coefficients, frame_to_u, degree


def atomic_savez(path: Path, **payload: np.ndarray) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary_path.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)


def percentile_stats(values: list[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p1": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "p99_9": None,
            "p99_99": None,
            "max": None,
        }
    stats: dict[str, float | int | None] = {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p1": float(np.percentile(array, 1)),
        "p5": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "p99_99": float(np.percentile(array, 99.99)),
        "max": float(np.max(array)),
    }
    return stats


def format_distribution(stats: dict[str, float | int | None]) -> str:
    keys = ("count", "min", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "p99_9", "p99_99", "max")
    values = []
    for key in keys:
        value = stats[key]
        if key == "count":
            values.append(f"{key}={value}")
        elif value is None:
            values.append(f"{key}=n/a")
        else:
            values.append(f"{key}={float(value):.6g}")
    return ", ".join(values)


def process_episode(
    episode_dir: Path,
    cfg: LocalTargetConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    arrays_dir = episode_dir / "arrays"
    output_arrays_dir = output_arrays_dir_for_episode(episode_dir, cfg)
    output_npz = output_arrays_dir / cfg.output_npz_name
    output_index = output_arrays_dir / cfg.output_index_name
    output_episode_summary = output_arrays_dir / cfg.output_episode_summary_name
    if all(path.exists() for path in (output_npz, output_index, output_episode_summary)) and not cfg.overwrite:
        try:
            existing_summary = json.loads(output_episode_summary.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_summary = {}
        return {
            "episode_uid": episode_dir.name,
            "status": "skipped_existing",
            "num_frames": int(existing_summary.get("num_frames", 0)),
            "candidate_start_frames": int(existing_summary.get("candidate_start_frames", 0)),
            "num_targets": int(existing_summary.get("targets_created", 0)),
            "short_horizons_skipped": int(existing_summary.get("short_horizons_skipped", 0)),
            "frames_without_targets": int(existing_summary.get("frames_without_targets", 0)),
        }

    action = np.load(arrays_dir / cfg.action_array_name).astype(np.float64, copy=False)
    state = np.load(arrays_dir / cfg.state_array_name).astype(np.float64, copy=False)
    if action.ndim != 2 or action.shape[1] != 65:
        raise ValueError(f"{episode_dir.name}: expected {cfg.action_array_name} shape (frames, 65), got {action.shape}.")
    if state.shape != action.shape:
        raise ValueError(f"{episode_dir.name}: action/state mismatch {action.shape} versus {state.shape}.")

    with np.load(arrays_dir / cfg.global_spline_npz_name, allow_pickle=False) as global_payload:
        global_knots, global_coefficients, frame_to_u, global_degree = validate_global_payload(
            episode_dir.name, global_payload, action.shape[0], action.shape[1]
        )

    global_spline = BSpline(global_knots, global_coefficients, global_degree)
    global_absolute_prediction = np.asarray(global_spline(frame_to_u), dtype=np.float64)
    global_difference = action - global_absolute_prediction
    global_frame_mae = np.mean(np.abs(global_difference), axis=1)
    global_frame_max_abs = np.max(np.abs(global_difference), axis=1)
    global_unique_knots = unique_knots(global_knots)
    candidate_start_frames = list(range(0, action.shape[0] - 1, cfg.sample_stride))

    if progress_callback is not None:
        progress_callback(
            {
                "event": "episode_start",
                "episode_uid": episode_dir.name,
                "total_candidates": len(candidate_start_frames),
            }
        )

    rows: list[dict[str, Any]] = []
    coefficients_parts: list[np.ndarray] = []
    coefficient_offsets = [0]
    knots_parts: list[np.ndarray] = []
    knot_offsets = [0]
    sample_ids: list[int] = []
    anchor_states: list[np.ndarray] = []
    frame_index_parts: list[np.ndarray] = []
    frame_index_offsets = [0]
    v_parts: list[np.ndarray] = []
    v_offsets = [0]
    skipped_short_horizon = 0
    restriction_errors: list[float] = []
    global_mae_means: list[float] = []
    global_max_abs_maxes: list[float] = []

    for candidate_index, start_frame in enumerate(candidate_start_frames, start=1):
        horizon = target_horizon(
            frame_to_u,
            global_unique_knots,
            start_frame,
            cfg.target_knot_spans,
            cfg.include_truncated_horizons,
        )
        if horizon is None:
            skipped_short_horizon += 1
        else:
            start_span, end_span, horizon_boundary_u, end_frame, u_end = horizon
            u_start = float(frame_to_u[start_frame])
            frame_indices = np.arange(start_frame, end_frame + 1, dtype=np.int64)
            v_values = (frame_to_u[frame_indices] - u_start) / (u_end - u_start)
            v_values[0] = 0.0
            v_values[-1] = 1.0

            local_knots, absolute_coefficients = restrict_bspline_segment(
                global_knots, global_coefficients, global_degree, u_start, u_end
            )
            anchor_state = state[start_frame]
            delta_coefficients = absolute_coefficients - anchor_state[None, :]

            # Restriction and translation must reconstruct the global absolute curve exactly.
            check_v = np.linspace(0.0, 1.0, cfg.restriction_validation_points, dtype=np.float64)
            local_absolute_prediction = BSpline(local_knots, delta_coefficients, global_degree)(check_v) + anchor_state
            global_check_prediction = global_spline(u_start + check_v * (u_end - u_start))
            restriction_max_abs_error = float(np.max(np.abs(local_absolute_prediction - global_check_prediction)))
            if restriction_max_abs_error > cfg.restriction_validation_tolerance:
                raise RuntimeError(
                    f"{episode_dir.name}, frame {start_frame}: restriction error {restriction_max_abs_error:.3e} "
                    f"exceeds tolerance {cfg.restriction_validation_tolerance:.3e}."
                )

            sample_index = len(sample_ids)
            coefficients_parts.append(delta_coefficients.astype(np.float32))
            coefficient_offsets.append(coefficient_offsets[-1] + delta_coefficients.shape[0])
            knots_parts.append(local_knots)
            knot_offsets.append(knot_offsets[-1] + local_knots.size)
            sample_ids.append(start_frame)
            anchor_states.append(anchor_state.astype(np.float32))
            frame_index_parts.append(frame_indices)
            frame_index_offsets.append(frame_index_offsets[-1] + frame_indices.size)
            v_parts.append(v_values.astype(np.float32))
            v_offsets.append(v_offsets[-1] + v_values.size)

            local_global_mae = global_frame_mae[frame_indices]
            local_global_max_abs = global_frame_max_abs[frame_indices]
            global_mae_means.append(float(np.mean(local_global_mae)))
            global_max_abs_maxes.append(float(np.max(local_global_max_abs)))
            restriction_errors.append(restriction_max_abs_error)
            rows.append(
                {
                    "episode_uid": episode_dir.name,
                    "sample_id": start_frame,
                    "npz_sample_index": sample_index,
                    "status": "ok",
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "num_target_frames": int(frame_indices.size),
                    "u_start": u_start,
                    "u_end": u_end,
                    "horizon_boundary_u": horizon_boundary_u,
                    "start_knot_span_index": start_span,
                    "end_knot_span_index": end_span,
                    "target_knot_spans_requested": cfg.target_knot_spans,
                    "target_knot_spans_actual": end_span - start_span,
                    "horizon_was_truncated": bool(end_span - start_span != cfg.target_knot_spans),
                    "anchor_state_frame": start_frame,
                    "anchor_state_source": f"{cfg.state_array_name}[start_frame]",
                    "local_spline_degree": global_degree,
                    "local_num_control_points": int(delta_coefficients.shape[0]),
                    "local_num_knots_total": int(local_knots.size),
                    "target_type": "local_delta_action_spline",
                    "target_definition": f"{cfg.action_array_name}[frame] - {cfg.state_array_name}[start_frame]",
                    "reconstruction_rule": "absolute_action(v) = current_state + local_delta_action_spline(v)",
                    "fit_source": "exact_global_cubic_spline_restriction_no_refit",
                    "global_frame_mae_65d_mean": float(np.mean(local_global_mae)),
                    "global_frame_mae_65d_max": float(np.max(local_global_mae)),
                    "global_frame_max_abs_joint_max": float(np.max(local_global_max_abs)),
                    "restriction_max_abs_error": restriction_max_abs_error,
                }
            )

        if progress_callback is not None and (
            candidate_index % cfg.progress_update_every_samples == 0 or candidate_index == len(candidate_start_frames)
        ):
            progress_callback(
                {
                    "event": "episode_progress",
                    "episode_uid": episode_dir.name,
                    "completed_candidates": candidate_index,
                    "total_candidates": len(candidate_start_frames),
                    "targets_created": len(sample_ids),
                    "short_horizons_skipped": skipped_short_horizon,
                }
            )

    target_dim = action.shape[1]
    coefficient_values = (
        np.concatenate(coefficients_parts, axis=0).astype(np.float32)
        if coefficients_parts
        else np.empty((0, target_dim), dtype=np.float32)
    )
    local_knot_values = np.concatenate(knots_parts).astype(np.float64) if knots_parts else np.empty(0, dtype=np.float64)
    frame_index_values = (
        np.concatenate(frame_index_parts).astype(np.int64) if frame_index_parts else np.empty(0, dtype=np.int64)
    )
    v_values = np.concatenate(v_parts).astype(np.float32) if v_parts else np.empty(0, dtype=np.float32)
    anchor_state_values = (
        np.stack(anchor_states).astype(np.float32) if anchor_states else np.empty((0, target_dim), dtype=np.float32)
    )

    output_arrays_dir.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        output_npz,
        sample_ids=np.asarray(sample_ids, dtype=np.int64),
        coefficients=coefficient_values,
        coefficient_offsets=np.asarray(coefficient_offsets, dtype=np.int64),
        local_knots=local_knot_values,
        local_knot_offsets=np.asarray(knot_offsets, dtype=np.int64),
        local_degree=np.asarray([global_degree], dtype=np.int64),
        anchor_state_65d=anchor_state_values,
        frame_index_values=frame_index_values,
        frame_index_offsets=np.asarray(frame_index_offsets, dtype=np.int64),
        v_values=v_values,
        v_offsets=np.asarray(v_offsets, dtype=np.int64),
        target_knot_spans=np.asarray([cfg.target_knot_spans], dtype=np.int64),
        target_type=np.asarray(["local_delta_action_from_current_state"], dtype="U64"),
        source_global_spline=np.asarray([cfg.global_spline_npz_name], dtype="U256"),
        reconstruction_rule=np.asarray(["absolute_action(v) = current_state + local_delta_action_spline(v)"], dtype="U128"),
    )
    atomic_parquet(pd.DataFrame(rows), output_index)

    episode_summary = {
        "episode_uid": episode_dir.name,
        "status": "processed",
        "num_frames": int(action.shape[0]),
        "candidate_start_frames": len(candidate_start_frames),
        "targets_created": len(sample_ids),
        "short_horizons_skipped": skipped_short_horizon,
        "frames_without_targets": int(action.shape[0] - len(sample_ids)),
        "target_knot_spans": cfg.target_knot_spans,
        "include_truncated_horizons": cfg.include_truncated_horizons,
        "global_degree": global_degree,
        "global_frame_mae_65d": percentile_stats(global_frame_mae),
        "global_frame_max_abs_joint_error": percentile_stats(global_frame_max_abs),
        "per_target_mean_global_mae_65d": percentile_stats(global_mae_means),
        "per_target_max_global_max_abs_joint_error": percentile_stats(global_max_abs_maxes),
        "restriction_max_abs_error": percentile_stats(restriction_errors),
        "output_npz": str(output_npz),
        "output_index": str(output_index),
    }
    atomic_json(episode_summary, output_episode_summary)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "episode_complete",
                "episode_uid": episode_dir.name,
                "completed_candidates": len(candidate_start_frames),
                "total_candidates": len(candidate_start_frames),
                "targets_created": len(sample_ids),
                "short_horizons_skipped": skipped_short_horizon,
            }
        )

    return {
        "episode_uid": episode_dir.name,
        "status": "processed",
        "num_frames": int(action.shape[0]),
        "candidate_start_frames": len(candidate_start_frames),
        "num_targets": len(sample_ids),
        "short_horizons_skipped": skipped_short_horizon,
        "frames_without_targets": int(action.shape[0] - len(sample_ids)),
        "max_global_frame_mae_65d": float(np.max(global_frame_mae)),
        "max_global_frame_max_abs_joint_error": float(np.max(global_frame_max_abs)),
        "max_restriction_abs_error": max(restriction_errors, default=0.0),
        "output_npz": str(output_npz),
        "output_index": str(output_index),
        "episode_summary": str(output_episode_summary),
    }


class WorkerProgressRenderer:
    """Render worker progress only in the parent process to keep tqdm output coherent."""

    def __init__(self, worker_count: int) -> None:
        self.worker_slots: dict[int, int] = {}
        self.bars = [
            tqdm(total=1, desc=f"Worker {slot + 1}: idle", unit="target", leave=False, dynamic_ncols=True, position=slot + 1)
            for slot in range(worker_count)
        ]

    def close(self) -> None:
        for bar in self.bars:
            bar.close()

    def handle_event(self, event: dict[str, Any]) -> None:
        worker_id = int(event["worker_id"])
        if worker_id not in self.worker_slots:
            self.worker_slots[worker_id] = len(self.worker_slots)
        slot = self.worker_slots[worker_id]
        if slot >= len(self.bars):
            raise RuntimeError("Received progress from more workers than configured.")
        bar = self.bars[slot]
        event_type = str(event["event"])
        if event_type == "episode_start":
            total = max(1, int(event["total_candidates"]))
            bar.reset(total=total)
            bar.set_description(f"Worker {slot + 1}: {event['episode_uid']}")
            bar.set_postfix(targets=0, short_horizons=0)
        elif event_type in {"episode_progress", "episode_complete"}:
            completed = int(event["completed_candidates"])
            bar.n = min(completed, bar.total or completed)
            bar.set_postfix(
                targets=int(event["targets_created"]),
                short_horizons=int(event.get("short_horizons_skipped", 0)),
            )
            bar.refresh()
        else:
            raise ValueError(f"Unknown progress event: {event_type!r}")


def process_episode_with_progress(episode_dir: Path, cfg: LocalTargetConfig, progress_queue: Any) -> dict[str, Any]:
    worker_id = os.getpid()

    def publish(event: dict[str, Any]) -> None:
        progress_queue.put({"worker_id": worker_id, **event})

    return process_episode(episode_dir, cfg, publish)


def drain_progress_events(progress_queue: Any, renderer: WorkerProgressRenderer) -> None:
    while True:
        try:
            renderer.handle_event(progress_queue.get_nowait())
        except Empty:
            return


def update_overall_progress(progress: tqdm, result: dict[str, Any]) -> None:
    progress.update(1)
    if result["status"] == "processed":
        progress.set_postfix(
            targets=int(result["num_targets"]),
            max_mae_65d=f"{float(result['max_global_frame_mae_65d']):.6f}",
            max_abs=f"{float(result['max_global_frame_max_abs_joint_error']):.6f}",
        )


def process_all(cfg: LocalTargetConfig) -> list[dict[str, Any]]:
    episode_dirs = list_episode_dirs(cfg)
    if not episode_dirs:
        return []

    worker_count = min(cfg.num_workers, len(episode_dirs))
    renderer = WorkerProgressRenderer(worker_count)
    results: list[dict[str, Any]] = []
    try:
        if worker_count == 1:
            with tqdm(total=len(episode_dirs), desc="Create local delta-action targets", unit="episode", dynamic_ncols=True) as overall:
                for episode_dir in episode_dirs:
                    result = process_episode(
                        episode_dir,
                        cfg,
                        lambda event: renderer.handle_event({"worker_id": os.getpid(), **event}),
                    )
                    results.append(result)
                    update_overall_progress(overall, result)
            return results

        with Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(process_episode_with_progress, episode_dir, cfg, progress_queue): episode_dir
                    for episode_dir in episode_dirs
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Create local delta-action targets ({worker_count} workers)",
                    unit="episode",
                    dynamic_ncols=True,
                    position=0,
                ) as overall:
                    while futures:
                        completed, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                        drain_progress_events(progress_queue, renderer)
                        for future in completed:
                            episode_dir = futures.pop(future)
                            try:
                                result = future.result()
                            except Exception:
                                tqdm.write(f"Failed episode: {episode_dir.name}")
                                raise
                            results.append(result)
                            update_overall_progress(overall, result)
                    drain_progress_events(progress_queue, renderer)
    finally:
        renderer.close()
    return sorted(results, key=lambda result: result["episode_uid"])


def write_run_summary(cfg: LocalTargetConfig, results: list[dict[str, Any]]) -> Path:
    output_root = summary_root(cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_parquet(pd.DataFrame(results), output_root / f"{cfg.output_tag}_run_summary.parquet")
    tail_horizon_skips = [int(result.get("short_horizons_skipped", 0)) for result in results]
    frames_without_targets = [int(result.get("frames_without_targets", 0)) for result in results]
    summary = {
        "config": {
            **asdict(cfg),
            "dataset_root": str(cfg.dataset_root),
            "output_dir": str(cfg.output_dir) if cfg.output_dir is not None else None,
        },
        "episodes_found": len(results),
        "processed_episodes": sum(result["status"] == "processed" for result in results),
        "skipped_episodes": sum(result["status"].startswith("skipped") for result in results),
        "total_targets": sum(int(result.get("num_targets", 0)) for result in results),
        "total_short_horizons_skipped": sum(int(result.get("short_horizons_skipped", 0)) for result in results),
        "total_frames_without_targets": sum(frames_without_targets),
        "tail_candidate_frames_skipped_for_full_horizon_per_episode": percentile_stats(tail_horizon_skips),
        "frames_without_local_targets_per_episode": percentile_stats(frames_without_targets),
        "max_global_frame_mae_65d": max(
            (float(result.get("max_global_frame_mae_65d", 0.0)) for result in results), default=0.0
        ),
        "max_global_frame_max_abs_joint_error": max(
            (float(result.get("max_global_frame_max_abs_joint_error", 0.0)) for result in results), default=0.0
        ),
        "max_restriction_abs_error": max(
            (float(result.get("max_restriction_abs_error", 0.0)) for result in results), default=0.0
        ),
        "results": results,
    }
    summary_path = output_root / f"{cfg.output_tag}_run_summary.json"
    atomic_json(summary, summary_path)
    return summary_path


def main() -> int:
    cfg = load_config(parse_args())
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(cfg.dataset_root)

    print(f"Dataset root                : {cfg.dataset_root}")
    print(f"Global spline input         : {cfg.global_spline_npz_name}")
    print(f"Required global degree      : {EXPECTED_GLOBAL_DEGREE}")
    print(f"Target knot spans           : {cfg.target_knot_spans}")
    print(f"Sample stride               : {cfg.sample_stride}")
    print(f"Include truncated horizons  : {cfg.include_truncated_horizons}")
    print(f"Num workers                 : {cfg.num_workers}")
    print(f"Max episodes                : {cfg.max_episodes if cfg.max_episodes is not None else 'all'}")
    print(f"Output root                 : {cfg.output_dir if cfg.output_dir is not None else 'dataset episode arrays'}")
    print(f"Output NPZ                  : {cfg.output_npz_name}")
    print(f"Output index                : {cfg.output_index_name}")
    print(f"Overwrite                   : {cfg.overwrite}")

    results = process_all(cfg)
    summary_path = write_run_summary(cfg, results)
    tail_horizon_skip_stats = percentile_stats([int(result.get("short_horizons_skipped", 0)) for result in results])
    frames_without_target_stats = percentile_stats([int(result.get("frames_without_targets", 0)) for result in results])
    print(f"Episodes found              : {len(results)}")
    print(f"Processed episodes          : {sum(result['status'] == 'processed' for result in results)}")
    print(f"Skipped episodes            : {sum(result['status'].startswith('skipped') for result in results)}")
    print(f"Local targets created       : {sum(int(result.get('num_targets', 0)) for result in results)}")
    print(f"Tail starts skipped         : {sum(int(result.get('short_horizons_skipped', 0)) for result in results)}")
    print(f"Frames without targets      : {sum(int(result.get('frames_without_targets', 0)) for result in results)}")
    print(f"Tail starts/episode         : {format_distribution(tail_horizon_skip_stats)}")
    print(f"No-target frames/episode    : {format_distribution(frames_without_target_stats)}")
    print(
        "Worst global frame MAE-65D : "
        f"{max((float(result.get('max_global_frame_mae_65d', 0.0)) for result in results), default=0.0):.8f}"
    )
    print(
        "Worst global frame max abs : "
        f"{max((float(result.get('max_global_frame_max_abs_joint_error', 0.0)) for result in results), default=0.0):.8f}"
    )
    print(
        "Worst restriction error    : "
        f"{max((float(result.get('max_restriction_abs_error', 0.0)) for result in results), default=0.0):.3e}"
    )
    print(f"Run summary                 : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
