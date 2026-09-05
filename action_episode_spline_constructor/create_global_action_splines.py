from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import BSpline, make_lsq_spline
from tqdm.auto import tqdm


VALID_ERROR_METRICS = ("rmse_65d", "mae_65d", "max_abs_joint")
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class SplineConfig:
    dataset_root: Path
    degree: int
    epsilon: float
    error_metric: str
    initial_internal_knots: int
    max_internal_knots: int
    min_knot_spacing_frames: int
    endpoint_weight: float
    overwrite: bool
    output_dir: Path | None
    spline_npz_name: str
    spline_index_name: str
    target_array_name: str
    state_array_name: str
    save_delta_from_initial_state: bool
    delta_spline_npz_name: str
    delta_spline_index_name: str
    run_summary_csv_name: str
    run_summary_yaml_name: str
    num_workers: int
    max_episodes: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit adaptive clamped quartic B-splines over action_65d for every processed episode."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Processed dataset root override.")
    parser.add_argument("--epsilon", type=float, default=None, help="Max frame reconstruction error target.")
    parser.add_argument("--max-internal-knots", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of eligible episodes to process.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional external output root.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> SplineConfig:
    raw: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    spline = raw.get("spline", {})
    output = raw.get("output", {})
    dataset_root_value = args.dataset_root or raw.get("dataset_root") or raw.get("input_root")
    if dataset_root_value is None:
        raise ValueError("dataset_root is required in config or --dataset-root")

    error_metric = str(spline.get("error_metric", "rmse_65d")).strip().lower()
    if error_metric not in VALID_ERROR_METRICS:
        raise ValueError(
            f"Unsupported spline.error_metric={error_metric!r}. Expected one of {list(VALID_ERROR_METRICS)}."
        )

    output_dir_value = args.output_dir or output.get("output_dir")
    output_dir = Path(output_dir_value) if output_dir_value else None

    return SplineConfig(
        dataset_root=Path(dataset_root_value),
        degree=int(spline.get("degree", 4)),
        epsilon=float(args.epsilon if args.epsilon is not None else spline.get("epsilon", 0.001746)),
        error_metric=error_metric,
        initial_internal_knots=int(spline.get("initial_internal_knots", 4)),
        max_internal_knots=int(
            args.max_internal_knots if args.max_internal_knots is not None else spline.get("max_internal_knots", 10000)
        ),
        min_knot_spacing_frames=int(spline.get("min_knot_spacing_frames", 1)),
        endpoint_weight=float(spline.get("endpoint_weight", 1000.0)),
        overwrite=bool(args.overwrite or output.get("overwrite", False)),
        output_dir=output_dir,
        spline_npz_name=str(output.get("spline_npz_name", "global_action_spline.npz")),
        spline_index_name=str(output.get("spline_index_name", "global_action_spline_index.parquet")),
        target_array_name=str(output.get("target_array_name", "action_65d.npy")),
        state_array_name=str(output.get("state_array_name", raw.get("state_array_name", "state_65d.npy"))),
        save_delta_from_initial_state=bool(output.get("save_delta_from_initial_state", False)),
        delta_spline_npz_name=str(
            output.get("delta_spline_npz_name", "global_delta_action_from_initial_state_spline.npz")
        ),
        delta_spline_index_name=str(
            output.get("delta_spline_index_name", "global_delta_action_from_initial_state_spline_index.parquet")
        ),
        run_summary_csv_name=str(output.get("run_summary_csv_name", "global_action_spline_run_summary.csv")),
        run_summary_yaml_name=str(output.get("run_summary_yaml_name", "global_action_spline_run_summary.yaml")),
        num_workers=max(1, int(args.num_workers if args.num_workers is not None else raw.get("num_workers", 1))),
        max_episodes=(
            int(args.max_episodes)
            if args.max_episodes is not None
            else (int(raw["max_episodes"]) if raw.get("max_episodes") is not None else None)
        ),
    )


def list_episode_dirs(dataset_root: Path, target_array_name: str) -> list[Path]:
    episode_root = dataset_root / "episodes"
    if not episode_root.exists():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")
    return [
        episode_dir
        for episode_dir in sorted(path for path in episode_root.iterdir() if path.is_dir())
        if (episode_dir / "arrays" / target_array_name).exists()
    ]


def compute_frame_rmse(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    diff = y - pred
    return np.sqrt(np.mean(diff * diff, axis=1))


def compute_frame_mae(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(y - pred), axis=1)


def compute_frame_max_abs_joint_error(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.max(np.abs(y - pred), axis=1)


def compute_stop_error(y: np.ndarray, pred: np.ndarray, error_metric: str) -> np.ndarray:
    if error_metric == "rmse_65d":
        return compute_frame_rmse(y, pred)
    if error_metric == "mae_65d":
        return compute_frame_mae(y, pred)
    if error_metric == "max_abs_joint":
        return compute_frame_max_abs_joint_error(y, pred)
    raise ValueError(f"Unsupported error metric: {error_metric!r}")


def output_root_for_run(cfg: SplineConfig) -> Path:
    if cfg.output_dir is not None:
        return cfg.output_dir
    return cfg.dataset_root / "metadata" / "global_action_spline_run"


def output_arrays_dir_for_episode(episode_dir: Path, cfg: SplineConfig) -> Path:
    if cfg.output_dir is None:
        return episode_dir / "arrays"
    return cfg.output_dir / "episodes" / episode_dir.name / "arrays"


def make_clamped_knots(x: np.ndarray, internal_indices: list[int], degree: int) -> np.ndarray:
    internal = [float(x[index]) for index in sorted(set(internal_indices)) if 0 < index < len(x) - 1]
    return np.asarray(([0.0] * (degree + 1)) + internal + ([1.0] * (degree + 1)), dtype=np.float64)


def initial_internal_indices(n: int, count: int, degree: int) -> list[int]:
    max_internal = max(0, n - degree - 1)
    count = min(max(0, count), max_internal)
    if count == 0 or n <= 2:
        return []
    values = np.linspace(1, n - 2, count + 2, dtype=np.float64)[1:-1]
    return sorted({int(round(value)) for value in values if 0 < int(round(value)) < n - 1})


def fit_once(
    x: np.ndarray,
    y: np.ndarray,
    internal_indices: list[int],
    degree: int,
    endpoint_weight: float,
) -> BSpline:
    knots = make_clamped_knots(x, internal_indices, degree)
    weights = np.ones(len(x), dtype=np.float64)
    weights[0] = endpoint_weight
    weights[-1] = endpoint_weight
    return make_lsq_spline(x, y, knots, k=degree, w=weights)


def choose_next_knot_index(errors: np.ndarray, existing: set[int], min_spacing: int) -> int | None:
    for index in np.argsort(errors)[::-1]:
        candidate = int(index)
        if candidate <= 0 or candidate >= len(errors) - 1:
            continue
        if candidate in existing:
            continue
        if any(abs(candidate - old) < min_spacing for old in existing):
            continue
        return candidate
    return None


def adaptive_fit_spline(
    y: np.ndarray,
    cfg: SplineConfig,
    episode_uid: str,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    n = int(y.shape[0])
    if n < cfg.degree + 1:
        raise ValueError(f"Need at least {cfg.degree + 1} samples for degree-{cfg.degree} spline, got {n}")

    degree = cfg.degree
    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    max_possible_internal = max(0, n - degree - 1)
    max_internal = min(cfg.max_internal_knots, max_possible_internal)
    internal = initial_internal_indices(n, cfg.initial_internal_knots, degree)[:max_internal]
    status = "max_knots_reached"
    last_exc: str | None = None
    spline: BSpline | None = None
    stop_error_per_frame: np.ndarray | None = None
    rmse_per_frame: np.ndarray | None = None
    mae_per_frame: np.ndarray | None = None
    max_abs_joint_error_per_frame: np.ndarray | None = None
    total_fit_iterations = max(1, max_internal - len(internal) + 1)
    progress = None
    if progress_callback is None:
        progress = tqdm(
            total=total_fit_iterations,
            desc=f"Adaptive fit: {episode_uid}",
            unit="iter",
            leave=False,
            dynamic_ncols=True,
        )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "fit_start",
                "episode_uid": episode_uid,
                "num_frames": n,
                "total_iterations": total_fit_iterations,
            }
        )
    iteration = 0

    try:
        while True:
            try:
                spline = fit_once(x, y, internal, degree, cfg.endpoint_weight)
                prediction = spline(x)
                stop_error_per_frame = compute_stop_error(y, prediction, cfg.error_metric)
                rmse_per_frame = compute_frame_rmse(y, prediction)
                mae_per_frame = compute_frame_mae(y, prediction)
                max_abs_joint_error_per_frame = compute_frame_max_abs_joint_error(y, prediction)
                max_error = float(np.max(stop_error_per_frame))
                mean_error = float(np.mean(stop_error_per_frame))
                max_rmse_65d = float(np.max(rmse_per_frame))
                mean_rmse_65d = float(np.mean(rmse_per_frame))
                max_mae = float(np.max(mae_per_frame))
                mean_mae = float(np.mean(mae_per_frame))
                max_max_abs_joint_error = float(np.max(max_abs_joint_error_per_frame))
                mean_max_abs_joint_error = float(np.mean(max_abs_joint_error_per_frame))
                iteration += 1
                control_points_per_frame = np.asarray(spline.c).shape[0] / n
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        internal=f"{len(internal)}/{max_internal}",
                        stop_max=f"{max_error:.6f}",
                        max_mae_65d=f"{max_mae:.6f}",
                        control_points_per_frame=f"{control_points_per_frame:.4f}",
                    )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "fit_update",
                            "episode_uid": episode_uid,
                            "num_frames": n,
                            "iteration": iteration,
                            "total_iterations": total_fit_iterations,
                            "stop_max": max_error,
                            "max_mae_65d": max_mae,
                            "control_points_per_frame": control_points_per_frame,
                        }
                    )
                if max_error <= cfg.epsilon:
                    status = "epsilon_met"
                    break
                if len(internal) >= max_internal:
                    status = "max_knots_reached"
                    break
                next_index = choose_next_knot_index(stop_error_per_frame, set(internal), cfg.min_knot_spacing_frames)
                if next_index is None:
                    status = "no_valid_knot_candidate"
                    break
                internal = sorted(set(internal + [next_index]))
            except Exception as exc:
                last_exc = repr(exc)
                if internal:
                    internal = internal[:-1]
                    status = "fit_failed_after_knot_removal"
                    continue
                raise
    finally:
        if progress is not None:
            progress.close()

    if progress_callback is not None:
        progress_callback(
            {
                "event": "fit_complete",
                "episode_uid": episode_uid,
                "num_frames": n,
                "iteration": iteration,
                "total_iterations": total_fit_iterations,
            }
        )

    if any(value is None for value in (spline, stop_error_per_frame, rmse_per_frame, mae_per_frame, max_abs_joint_error_per_frame)):
        raise RuntimeError("Spline fitting failed before producing a spline.")

    return {
        "spline": spline,
        "x": x,
        "internal_indices": np.asarray(internal, dtype=np.int64),
        "stop_error_per_frame": stop_error_per_frame.astype(np.float32),
        "rmse_per_frame": rmse_per_frame.astype(np.float32),
        "mae_per_frame": mae_per_frame.astype(np.float32),
        "max_abs_joint_error_per_frame": max_abs_joint_error_per_frame.astype(np.float32),
        "max_error": max_error,
        "mean_error": mean_error,
        "max_rmse_65d": max_rmse_65d,
        "mean_rmse_65d": mean_rmse_65d,
        "max_mae": max_mae,
        "mean_mae": mean_mae,
        "max_max_abs_joint_error": max_max_abs_joint_error,
        "mean_max_abs_joint_error": mean_max_abs_joint_error,
        "degree": degree,
        "status": status,
        "last_exception": last_exc,
    }


def load_action_and_reference_state(episode_dir: Path, cfg: SplineConfig) -> tuple[np.ndarray, np.ndarray | None]:
    arrays_dir = episode_dir / "arrays"
    action = np.load(arrays_dir / cfg.target_array_name).astype(np.float32, copy=False)
    if action.ndim != 2 or action.shape[1] != 65:
        raise ValueError(f"{episode_dir.name}: expected action shape (frames, 65), got {action.shape}")
    if action.shape[0] < cfg.degree + 1:
        raise ValueError(
            f"{episode_dir.name}: need at least {cfg.degree + 1} action frames for degree-{cfg.degree}, got {action.shape[0]}"
        )
    if not cfg.save_delta_from_initial_state:
        return action, None

    state = np.load(arrays_dir / cfg.state_array_name).astype(np.float32, copy=False)
    if state.ndim != 2 or state.shape[1] != action.shape[1] or state.shape[0] < 1:
        raise ValueError(
            f"{episode_dir.name}: expected {cfg.state_array_name} with shape (frames, {action.shape[1]}), got {state.shape}"
        )
    return action, state[0].astype(np.float64)


def build_npz_payload(spline: BSpline, fit: dict[str, Any], target_dim: int, num_frames: int, error_metric: str) -> dict[str, np.ndarray]:
    return {
        "global_knots": spline.t.astype(np.float64),
        "global_coefficients": np.asarray(spline.c, dtype=np.float64),
        "global_degree": np.asarray([fit["degree"]], dtype=np.int64),
        "stop_error_metric": np.asarray([error_metric], dtype="U32"),
        "frame_indices": np.arange(num_frames, dtype=np.int64),
        "frame_to_u": fit["x"].astype(np.float64),
        "error_per_frame": fit["stop_error_per_frame"],
        "rmse_per_frame": fit["rmse_per_frame"],
        "mae_per_frame": fit["mae_per_frame"],
        "max_abs_joint_error_per_frame": fit["max_abs_joint_error_per_frame"],
        "internal_knot_frame_indices": fit["internal_indices"],
        "target_dim": np.asarray([target_dim], dtype=np.int64),
    }


def process_episode(
    episode_dir: Path,
    cfg: SplineConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    arrays_dir = episode_dir / "arrays"
    output_arrays_dir = output_arrays_dir_for_episode(episode_dir, cfg)
    out_npz = output_arrays_dir / cfg.spline_npz_name
    out_index = output_arrays_dir / cfg.spline_index_name
    out_delta_npz = output_arrays_dir / cfg.delta_spline_npz_name
    out_delta_index = output_arrays_dir / cfg.delta_spline_index_name
    required_outputs = [out_npz, out_index]
    if cfg.save_delta_from_initial_state:
        required_outputs.extend([out_delta_npz, out_delta_index])
    if all(path.exists() for path in required_outputs) and not cfg.overwrite:
        return {"episode_uid": episode_dir.name, "status": "skipped_existing", "num_splines": len(required_outputs) // 2}

    action, reference_state = load_action_and_reference_state(episode_dir, cfg)
    num_frames = int(action.shape[0])
    fit = adaptive_fit_spline(action.astype(np.float64), cfg, episode_dir.name, progress_callback)
    spline: BSpline = fit["spline"]
    output_arrays_dir.mkdir(parents=True, exist_ok=True)

    npz_payload = build_npz_payload(spline, fit, action.shape[1], num_frames, cfg.error_metric)
    npz_payload["spline_target_mode"] = np.asarray(["global_action"], dtype="U64")
    timestamps_path = arrays_dir / "timestamps.npy"
    if timestamps_path.exists():
        timestamps = np.load(timestamps_path)
        if len(timestamps) == num_frames:
            npz_payload["timestamps"] = timestamps
    np.savez_compressed(out_npz, **npz_payload)

    row = {
        "episode_uid": episode_dir.name,
        "target": "global_action_spline = action_65d",
        "target_array_name": cfg.target_array_name,
        "start_frame": 0,
        "end_frame_exclusive": num_frames,
        "num_frames": num_frames,
        "target_dim": int(action.shape[1]),
        "degree": int(fit["degree"]),
        "num_internal_knots": int(len(fit["internal_indices"])),
        "num_knots_total": int(len(spline.t)),
        "num_control_points": int(np.asarray(spline.c).shape[0]),
        "epsilon": float(cfg.epsilon),
        "error_metric": cfg.error_metric,
        "max_internal_knots": int(cfg.max_internal_knots),
        "initial_internal_knots": int(cfg.initial_internal_knots),
        "min_knot_spacing_frames": int(cfg.min_knot_spacing_frames),
        "endpoint_weight": float(cfg.endpoint_weight),
        "max_error": float(fit["max_error"]),
        "mean_error": float(fit["mean_error"]),
        "max_rmse_65d": float(fit["max_rmse_65d"]),
        "mean_rmse_65d": float(fit["mean_rmse_65d"]),
        "max_mae": float(fit["max_mae"]),
        "mean_mae": float(fit["mean_mae"]),
        "max_max_abs_joint_error": float(fit["max_max_abs_joint_error"]),
        "mean_max_abs_joint_error": float(fit["mean_max_abs_joint_error"]),
        "fit_status": fit["status"],
        "last_exception": fit["last_exception"],
        "source_episode_dir": str(episode_dir),
        "output_arrays_dir": str(output_arrays_dir),
        "npz_path": str(out_npz),
        "npz_key_prefix": "global",
        "spline_target_mode": "global_action",
    }
    pd.DataFrame([row]).to_parquet(out_index, index=False)

    num_splines = 1
    if cfg.save_delta_from_initial_state:
        if reference_state is None:
            raise RuntimeError("Initial state reference was not loaded for requested delta spline.")
        delta_payload = dict(npz_payload)
        delta_payload["global_coefficients"] = np.asarray(spline.c, dtype=np.float64) - reference_state[None, :]
        delta_payload["spline_target_mode"] = np.asarray(["global_delta_action_from_initial_state"], dtype="U64")
        delta_payload["reference_type"] = np.asarray(["state_65d[0]"], dtype="U64")
        delta_payload["reference_array_name"] = np.asarray([cfg.state_array_name], dtype="U128")
        delta_payload["reference_frame"] = np.asarray([0], dtype=np.int64)
        delta_payload["reference_state_65d"] = reference_state
        delta_payload["absolute_reconstruction_rule"] = np.asarray(
            ["action(u) = reference_state_65d + global_delta_action_from_initial_state_spline(u)"], dtype="U128"
        )
        np.savez_compressed(out_delta_npz, **delta_payload)

        delta_row = dict(row)
        delta_row.update(
            {
                "target": "global_delta_action_from_initial_state_spline = action_65d - state_65d[0]",
                "reference_array_name": cfg.state_array_name,
                "reference_frame": 0,
                "reconstruction_rule": "action(u) = reference_state_65d + global_delta_action_from_initial_state_spline(u)",
                "npz_path": str(out_delta_npz),
                "spline_target_mode": "global_delta_action_from_initial_state",
            }
        )
        pd.DataFrame([delta_row]).to_parquet(out_delta_index, index=False)
        num_splines = 2

    return {
        "episode_uid": episode_dir.name,
        "status": "processed",
        "num_splines": num_splines,
        "num_frames": num_frames,
        "num_control_points": row["num_control_points"],
        "control_points_per_frame": row["num_control_points"] / num_frames,
        "max_error": row["max_error"],
        "max_mae": row["max_mae"],
        "fit_status": row["fit_status"],
    }


class WorkerFrameProgressRenderer:
    """Keep all worker progress rendering in the parent process."""

    def __init__(self, worker_episode_counts: list[int]) -> None:
        self.episode_bars = [
            tqdm(
                total=episode_count,
                desc=f"Worker {slot + 1} episodes",
                unit="episode",
                leave=False,
                dynamic_ncols=True,
                position=slot * 2 + 1,
            )
            for slot, episode_count in enumerate(worker_episode_counts)
        ]
        self.frame_bars = [
            tqdm(
                total=None,
                desc=f"Worker {slot + 1} frame evals: idle",
                unit="frame-eval",
                leave=False,
                dynamic_ncols=True,
                position=slot * 2 + 2,
            )
            for slot in range(len(worker_episode_counts))
        ]

    def close(self) -> None:
        for bar in [*self.episode_bars, *self.frame_bars]:
            bar.close()

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        worker_index = int(event["worker_index"])
        if worker_index < 0 or worker_index >= len(self.episode_bars):
            raise ValueError(f"Invalid worker index: {worker_index}")
        episode_bar = self.episode_bars[worker_index]
        frame_bar = self.frame_bars[worker_index]
        event_type = str(event["event"])
        episode_uid = str(event["episode_uid"])
        if event_type == "episode_start":
            frame_bar.reset()
            frame_bar.set_description(f"Worker {worker_index + 1} frame evals: {episode_uid}")
            frame_bar.set_postfix(status="loading")
            return None
        if event_type == "fit_start":
            frame_bar.set_description(f"Worker {worker_index + 1} frame evals: {episode_uid}")
            frame_bar.set_postfix(frames=int(event["num_frames"]), iteration=f"0/{event['total_iterations']}")
            return None
        if event_type == "fit_update":
            frame_count = int(event["num_frames"])
            iteration = int(event["iteration"])
            total_iterations = int(event["total_iterations"])
            # Each adaptive iteration evaluates every frame in the active episode.
            frame_bar.update(frame_count)
            frame_bar.set_postfix(
                frames=frame_count,
                iteration=f"{iteration}/{total_iterations}",
                stop_max=f"{float(event['stop_max']):.6f}",
                max_mae_65d=f"{float(event['max_mae_65d']):.6f}",
                control_points_per_frame=f"{float(event['control_points_per_frame']):.4f}",
            )
            return None
        if event_type == "fit_complete":
            return None
        if event_type == "episode_complete":
            episode_bar.update(1)
            frame_bar.set_postfix(status=event["result"]["status"])
            return dict(event["result"])
        raise ValueError(f"Unknown progress event: {event_type!r}")


def process_episode_batch_with_progress(
    worker_index: int,
    episode_dirs: list[Path],
    cfg: SplineConfig,
    progress_queue: Any,
) -> list[dict[str, Any]]:
    results = []

    def publish(event: dict[str, Any]) -> None:
        progress_queue.put({"worker_index": worker_index, **event})

    for episode_dir in episode_dirs:
        progress_queue.put({"worker_index": worker_index, "event": "episode_start", "episode_uid": episode_dir.name})
        result = process_episode(episode_dir, cfg, progress_callback=publish)
        results.append(result)
        progress_queue.put(
            {"worker_index": worker_index, "event": "episode_complete", "episode_uid": episode_dir.name, "result": result}
        )
    return results


def drain_progress_events(
    progress_queue: Any,
    renderer: WorkerFrameProgressRenderer,
    overall_progress: tqdm,
) -> None:
    while True:
        try:
            result = renderer.handle_event(progress_queue.get_nowait())
            if result is not None:
                update_overall_progress(overall_progress, result)
        except Empty:
            return


def update_overall_progress(progress: tqdm, result: dict[str, Any]) -> None:
    progress.update(1)
    if result["status"] == "processed":
        progress.set_postfix(
            stop_max=f"{float(result['max_error']):.6f}",
            max_mae_65d=f"{float(result['max_mae']):.6f}",
            control_points_per_frame=f"{float(result['control_points_per_frame']):.4f}",
        )


def process_all(cfg: SplineConfig) -> list[dict[str, Any]]:
    episode_dirs = list_episode_dirs(cfg.dataset_root, cfg.target_array_name)
    if cfg.max_episodes is not None:
        if cfg.max_episodes < 1:
            raise ValueError("max_episodes must be >= 1 when provided")
        episode_dirs = episode_dirs[: cfg.max_episodes]
    if not episode_dirs:
        return []
    if cfg.num_workers == 1:
        results = []
        renderer = WorkerFrameProgressRenderer(worker_episode_counts=[len(episode_dirs)])
        try:
            with tqdm(total=len(episode_dirs), desc="Fit global action splines", unit="episode", dynamic_ncols=True, position=0) as progress:
                for episode_dir in episode_dirs:
                    renderer.handle_event({"worker_index": 0, "event": "episode_start", "episode_uid": episode_dir.name})
                    result = process_episode(
                        episode_dir,
                        cfg,
                        progress_callback=lambda event: renderer.handle_event({"worker_index": 0, **event}),
                    )
                    results.append(result)
                    renderer.handle_event(
                        {"worker_index": 0, "event": "episode_complete", "episode_uid": episode_dir.name, "result": result}
                    )
                    update_overall_progress(progress, result)
        finally:
            renderer.close()
        return results

    results = []
    worker_count = min(cfg.num_workers, len(episode_dirs))
    worker_episode_dirs = [episode_dirs[worker_index::worker_count] for worker_index in range(worker_count)]
    renderer = WorkerFrameProgressRenderer(worker_episode_counts=[len(paths) for paths in worker_episode_dirs])
    try:
        with Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(process_episode_batch_with_progress, worker_index, paths, cfg, progress_queue): worker_index
                    for worker_index, paths in enumerate(worker_episode_dirs)
                }
                with tqdm(
                    total=len(futures),
                    desc=f"Fit global action splines ({worker_count} workers)",
                    unit="episode",
                    dynamic_ncols=True,
                    position=0,
                ) as progress:
                    while futures:
                        completed, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                        drain_progress_events(progress_queue, renderer, progress)
                        for future in completed:
                            worker_index = futures.pop(future)
                            try:
                                results.extend(future.result())
                            except Exception:
                                tqdm.write(f"Failed worker {worker_index + 1}")
                                raise
                    drain_progress_events(progress_queue, renderer, progress)
    finally:
        renderer.close()
    return sorted(results, key=lambda row: row["episode_uid"])


def write_run_summary(cfg: SplineConfig, results: list[dict[str, Any]]) -> None:
    summary_root = output_root_for_run(cfg)
    summary_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(summary_root / cfg.run_summary_csv_name, index=False)
    payload = {
        "dataset_root": str(cfg.dataset_root),
        "output_root": str(summary_root),
        "external_output_dir": None if cfg.output_dir is None else str(cfg.output_dir),
        "target_array_name": cfg.target_array_name,
        "state_array_name": cfg.state_array_name,
        "epsilon": cfg.epsilon,
        "error_metric": cfg.error_metric,
        "degree": cfg.degree,
        "initial_internal_knots": cfg.initial_internal_knots,
        "max_internal_knots": cfg.max_internal_knots,
        "min_knot_spacing_frames": cfg.min_knot_spacing_frames,
        "endpoint_weight": cfg.endpoint_weight,
        "save_delta_from_initial_state": cfg.save_delta_from_initial_state,
        "num_workers": cfg.num_workers,
        "max_episodes": cfg.max_episodes,
        "num_results": len(results),
        "results": results,
    }
    (summary_root / cfg.run_summary_yaml_name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(cfg.dataset_root)

    print(f"Dataset root       : {cfg.dataset_root}")
    print(f"target array       : {cfg.target_array_name}")
    print(f"degree             : {cfg.degree}")
    print(f"epsilon            : {cfg.epsilon}")
    print(f"error metric       : {cfg.error_metric}")
    print(f"max internal knots : {cfg.max_internal_knots}")
    print(f"num workers        : {cfg.num_workers}")
    print(f"max episodes       : {cfg.max_episodes if cfg.max_episodes is not None else 'all'}")
    print(f"overwrite          : {cfg.overwrite}")
    print(f"output root        : {cfg.output_dir if cfg.output_dir is not None else 'dataset episode arrays'}")
    print(f"output npz         : {cfg.spline_npz_name}")
    print(f"output index       : {cfg.spline_index_name}")
    print(f"save delta spline  : {cfg.save_delta_from_initial_state}")
    if cfg.save_delta_from_initial_state:
        print(f"delta output npz   : {cfg.delta_spline_npz_name}")
        print(f"delta output index : {cfg.delta_spline_index_name}")

    results = process_all(cfg)
    write_run_summary(cfg, results)
    processed = sum(1 for result in results if result["status"] == "processed")
    skipped = sum(1 for result in results if result["status"].startswith("skipped"))
    print(f"Episodes found      : {len(results)}")
    print(f"Processed episodes  : {processed}")
    print(f"Skipped episodes    : {skipped}")
    if results:
        max_error = max(float(result.get("max_error") or 0.0) for result in results)
        max_mae = max(float(result.get("max_mae") or 0.0) for result in results)
        max_control_points_per_frame = max(float(result.get("control_points_per_frame") or 0.0) for result in results)
        print(f"Worst max error     : {max_error:.6f}")
        print(f"Worst max mae_65d   : {max_mae:.6f}")
        print(f"Max ctrl pts/frame  : {max_control_points_per_frame:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
