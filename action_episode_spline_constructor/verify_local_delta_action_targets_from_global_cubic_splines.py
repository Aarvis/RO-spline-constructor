from __future__ import annotations

import argparse
import json
import os
import shutil
import uuid
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
EXPECTED_LOCAL_DEGREE = 3
PERCENTILES = (0.0, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9, 99.99, 100.0)


@dataclass(frozen=True)
class VerificationConfig:
    dataset_root: Path
    num_workers: int
    max_episodes: int | None
    target_knot_spans: int
    local_target_prefix: str
    action_array_name: str
    state_array_name: str
    output_dir: Path | None
    progress_update_every_targets: int
    anchor_state_tolerance: float
    keep_intermediate_error_arrays: bool
    scratch_dir: Path

    @property
    def target_tag(self) -> str:
        return f"{self.local_target_prefix}_knotspans{self.target_knot_spans}"

    @property
    def local_target_npz_name(self) -> str:
        return f"{self.target_tag}.npz"

    @property
    def local_target_index_name(self) -> str:
        return f"{self.target_tag}_index.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local delta-action spline targets against action_65d at their stored frame/v pairings."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration path.")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--target-knot-spans", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def output_root(dataset_root: Path, output_dir: Path | None, target_tag: str) -> Path:
    if output_dir is not None:
        return output_dir / f"{target_tag}_verification"
    return dataset_root / "metadata" / f"{target_tag}_verification"


def load_config(args: argparse.Namespace) -> VerificationConfig:
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    input_cfg = raw.get("input", {})
    output_cfg = raw.get("output", {})
    validation_cfg = raw.get("validation", {})

    dataset_root_value = args.dataset_root or raw.get("dataset_root")
    if dataset_root_value is None:
        raise ValueError("dataset_root is required in the config or --dataset-root.")
    dataset_root = Path(dataset_root_value)
    target_knot_spans = int(
        args.target_knot_spans if args.target_knot_spans is not None else input_cfg.get("target_knot_spans", 10)
    )
    if target_knot_spans < 1:
        raise ValueError("input.target_knot_spans must be at least 1.")
    output_dir_value = args.output_dir or output_cfg.get("output_dir")
    selected_output_dir = Path(output_dir_value) if output_dir_value else None
    prefix = str(input_cfg.get("local_target_prefix", "local_delta_action_cubic"))
    tag = f"{prefix}_knotspans{target_knot_spans}"
    run_scratch = output_root(dataset_root, selected_output_dir, tag) / f".scratch_{uuid.uuid4().hex}"
    max_episodes_value = args.max_episodes if args.max_episodes is not None else raw.get("max_episodes")
    return VerificationConfig(
        dataset_root=dataset_root,
        num_workers=max(1, int(args.num_workers if args.num_workers is not None else raw.get("num_workers", 1))),
        max_episodes=int(max_episodes_value) if max_episodes_value is not None else None,
        target_knot_spans=target_knot_spans,
        local_target_prefix=prefix,
        action_array_name=str(input_cfg.get("action_array_name", "action_65d.npy")),
        state_array_name=str(input_cfg.get("state_array_name", "state_65d.npy")),
        output_dir=selected_output_dir,
        progress_update_every_targets=max(1, int(raw.get("progress_update_every_targets", 25))),
        anchor_state_tolerance=float(validation_cfg.get("anchor_state_tolerance", 1e-6)),
        keep_intermediate_error_arrays=bool(output_cfg.get("keep_intermediate_error_arrays", False)),
        scratch_dir=run_scratch,
    )


def output_arrays_dir_for_episode(episode_dir: Path, cfg: VerificationConfig) -> Path:
    if cfg.output_dir is None:
        return episode_dir / "arrays"
    return cfg.output_dir / "episodes" / episode_dir.name / "arrays"


def list_episode_dirs(cfg: VerificationConfig) -> list[Path]:
    episode_root = cfg.dataset_root / "episodes"
    if not episode_root.exists():
        raise FileNotFoundError(f"Episode root not found: {episode_root}")
    episodes: list[Path] = []
    for episode_dir in sorted(path for path in episode_root.iterdir() if path.is_dir()):
        source_arrays = episode_dir / "arrays"
        target_arrays = output_arrays_dir_for_episode(episode_dir, cfg)
        if all(
            path.exists()
            for path in (
                source_arrays / cfg.action_array_name,
                source_arrays / cfg.state_array_name,
                target_arrays / cfg.local_target_npz_name,
                target_arrays / cfg.local_target_index_name,
            )
        ):
            episodes.append(episode_dir)
    if cfg.max_episodes is not None:
        if cfg.max_episodes < 1:
            raise ValueError("max_episodes must be at least 1 when provided.")
        episodes = episodes[: cfg.max_episodes]
    return episodes


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary_path.replace(path)


def validate_offsets(name: str, offsets: np.ndarray, total_size: int, sample_count: int) -> None:
    if offsets.ndim != 1 or offsets.size != sample_count + 1:
        raise ValueError(f"{name} offsets must have {sample_count + 1} entries, got {offsets.shape}.")
    if offsets[0] != 0 or offsets[-1] != total_size or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{name} offsets are invalid for total size {total_size}.")


def scalar_stats(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    as_float = np.asarray(values, dtype=np.float64)
    return {
        "count": int(as_float.size),
        "mean": float(np.mean(as_float)),
        "std": float(np.std(as_float)),
        "min": float(np.min(as_float)),
        "max": float(np.max(as_float)),
    }


def process_episode(
    episode_dir: Path,
    cfg: VerificationConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    source_arrays = episode_dir / "arrays"
    target_arrays = output_arrays_dir_for_episode(episode_dir, cfg)
    action = np.load(source_arrays / cfg.action_array_name).astype(np.float64, copy=False)
    state = np.load(source_arrays / cfg.state_array_name).astype(np.float64, copy=False)
    if action.ndim != 2 or action.shape[1] != 65:
        raise ValueError(f"{episode_dir.name}: expected action shape (frames, 65), got {action.shape}.")
    if state.shape != action.shape:
        raise ValueError(f"{episode_dir.name}: action/state mismatch {action.shape} versus {state.shape}.")

    index_frame = pd.read_parquet(target_arrays / cfg.local_target_index_name)
    with np.load(target_arrays / cfg.local_target_npz_name, allow_pickle=False) as payload:
        required = (
            "sample_ids",
            "coefficients",
            "coefficient_offsets",
            "local_knots",
            "local_knot_offsets",
            "local_degree",
            "anchor_state_65d",
            "frame_index_values",
            "frame_index_offsets",
            "v_values",
            "v_offsets",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"{episode_dir.name}: local target NPZ is missing keys: {missing}")
        sample_ids = np.asarray(payload["sample_ids"], dtype=np.int64)
        coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
        coefficient_offsets = np.asarray(payload["coefficient_offsets"], dtype=np.int64)
        local_knots = np.asarray(payload["local_knots"], dtype=np.float64)
        local_knot_offsets = np.asarray(payload["local_knot_offsets"], dtype=np.int64)
        local_degree = int(np.asarray(payload["local_degree"]).reshape(-1)[0])
        stored_anchor_states = np.asarray(payload["anchor_state_65d"], dtype=np.float64)
        frame_index_values = np.asarray(payload["frame_index_values"], dtype=np.int64)
        frame_index_offsets = np.asarray(payload["frame_index_offsets"], dtype=np.int64)
        v_values = np.asarray(payload["v_values"], dtype=np.float64)
        v_offsets = np.asarray(payload["v_offsets"], dtype=np.int64)

    if local_degree != EXPECTED_LOCAL_DEGREE:
        raise ValueError(f"{episode_dir.name}: expected local degree {EXPECTED_LOCAL_DEGREE}, found {local_degree}.")
    sample_count = sample_ids.size
    if coefficients.ndim != 2 or coefficients.shape[1] != action.shape[1]:
        raise ValueError(f"{episode_dir.name}: invalid coefficient shape {coefficients.shape}.")
    if stored_anchor_states.shape != (sample_count, action.shape[1]):
        raise ValueError(f"{episode_dir.name}: invalid stored anchor state shape {stored_anchor_states.shape}.")
    validate_offsets("coefficient", coefficient_offsets, coefficients.shape[0], sample_count)
    validate_offsets("knot", local_knot_offsets, local_knots.size, sample_count)
    validate_offsets("frame index", frame_index_offsets, frame_index_values.size, sample_count)
    validate_offsets("v", v_offsets, v_values.size, sample_count)
    if frame_index_values.size != v_values.size:
        raise ValueError(f"{episode_dir.name}: frame-index and v-value counts differ.")
    if index_frame.shape[0] != sample_count:
        raise ValueError(
            f"{episode_dir.name}: index rows ({index_frame.shape[0]}) do not match NPZ samples ({sample_count})."
        )
    if "sample_id" not in index_frame or not np.array_equal(index_frame["sample_id"].to_numpy(dtype=np.int64), sample_ids):
        raise ValueError(f"{episode_dir.name}: parquet sample_id order does not match the local target NPZ.")

    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)
    episode_stem = f"{episode_dir.name}_{os.getpid()}"
    mae_path = cfg.scratch_dir / f"{episode_stem}_mae.npy"
    max_abs_path = cfg.scratch_dir / f"{episode_stem}_max_abs.npy"
    frame_count_path = cfg.scratch_dir / f"{episode_stem}_frame_counts.npy"
    total_target_frame_occurrences = int(frame_index_values.size)
    mae_values = np.lib.format.open_memmap(mae_path, mode="w+", dtype=np.float32, shape=(total_target_frame_occurrences,))
    max_abs_values = np.lib.format.open_memmap(max_abs_path, mode="w+", dtype=np.float32, shape=(total_target_frame_occurrences,))
    target_frame_counts = np.lib.format.open_memmap(frame_count_path, mode="w+", dtype=np.int32, shape=(sample_count,))

    if progress_callback is not None:
        progress_callback(
            {"event": "episode_start", "episode_uid": episode_dir.name, "total_targets": sample_count}
        )

    max_anchor_state_abs_error = 0.0
    max_frame_mae = 0.0
    max_frame_max_abs = 0.0
    for sample_index, start_frame in enumerate(sample_ids):
        coefficient_start, coefficient_end = coefficient_offsets[sample_index : sample_index + 2]
        knot_start, knot_end = local_knot_offsets[sample_index : sample_index + 2]
        frame_start, frame_end = frame_index_offsets[sample_index : sample_index + 2]
        v_start, v_end = v_offsets[sample_index : sample_index + 2]
        frame_indices = frame_index_values[frame_start:frame_end]
        local_v = v_values[v_start:v_end]
        if frame_indices.size == 0 or frame_indices.size != local_v.size:
            raise ValueError(f"{episode_dir.name}, sample {sample_index}: invalid frame/v pairing.")
        if int(start_frame) != int(frame_indices[0]) or frame_indices[-1] >= action.shape[0] or frame_indices[0] < 0:
            raise ValueError(f"{episode_dir.name}, sample {sample_index}: invalid source frame indices.")
        if np.any(np.diff(frame_indices) != 1) or np.any(np.diff(local_v) < 0):
            raise ValueError(f"{episode_dir.name}, sample {sample_index}: frame/v values must be ordered.")
        if not np.isclose(local_v[0], 0.0, atol=1e-6) or not np.isclose(local_v[-1], 1.0, atol=1e-6):
            raise ValueError(f"{episode_dir.name}, sample {sample_index}: local v endpoints must be 0 and 1.")

        local_coefficients = coefficients[coefficient_start:coefficient_end]
        knot_vector = local_knots[knot_start:knot_end]
        if knot_vector.size != local_coefficients.shape[0] + local_degree + 1:
            raise ValueError(f"{episode_dir.name}, sample {sample_index}: invalid local knot/control-point dimensions.")

        dataset_anchor_state = state[int(start_frame)]
        max_anchor_state_abs_error = max(
            max_anchor_state_abs_error,
            float(np.max(np.abs(stored_anchor_states[sample_index] - dataset_anchor_state))),
        )
        reconstructed_action = BSpline(knot_vector, local_coefficients, local_degree)(local_v) + dataset_anchor_state
        difference = action[frame_indices] - reconstructed_action
        frame_mae = np.mean(np.abs(difference), axis=1)
        frame_max_abs = np.max(np.abs(difference), axis=1)
        mae_values[frame_start:frame_end] = frame_mae.astype(np.float32)
        max_abs_values[frame_start:frame_end] = frame_max_abs.astype(np.float32)
        target_frame_counts[sample_index] = frame_indices.size
        max_frame_mae = max(max_frame_mae, float(np.max(frame_mae)))
        max_frame_max_abs = max(max_frame_max_abs, float(np.max(frame_max_abs)))

        completed = sample_index + 1
        if progress_callback is not None and (
            completed % cfg.progress_update_every_targets == 0 or completed == sample_count
        ):
            progress_callback(
                {
                    "event": "episode_progress",
                    "episode_uid": episode_dir.name,
                    "completed_targets": completed,
                    "total_targets": sample_count,
                    "target_frame_occurrences": int(frame_end),
                    "max_mae_65d": max_frame_mae,
                    "max_abs": max_frame_max_abs,
                }
            )

    mae_values.flush()
    max_abs_values.flush()
    target_frame_counts.flush()
    del mae_values, max_abs_values, target_frame_counts
    if progress_callback is not None:
        progress_callback(
            {
                "event": "episode_complete",
                "episode_uid": episode_dir.name,
                "completed_targets": sample_count,
                "total_targets": sample_count,
                "target_frame_occurrences": total_target_frame_occurrences,
                "max_mae_65d": max_frame_mae,
                "max_abs": max_frame_max_abs,
            }
        )

    return {
        "episode_uid": episode_dir.name,
        "status": "processed",
        "num_local_targets": int(sample_count),
        "target_frame_occurrences": total_target_frame_occurrences,
        "max_frame_mae_65d": max_frame_mae,
        "max_frame_max_abs": max_frame_max_abs,
        "max_anchor_state_abs_error": max_anchor_state_abs_error,
        "anchor_state_tolerance_exceeded": max_anchor_state_abs_error > cfg.anchor_state_tolerance,
        "mae_values_path": str(mae_path),
        "max_abs_values_path": str(max_abs_path),
        "frame_count_values_path": str(frame_count_path),
    }


class WorkerProgressRenderer:
    """Only the parent process renders tqdm, preventing interleaved worker output."""

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
        bar = self.bars[slot]
        event_type = str(event["event"])
        if event_type == "episode_start":
            bar.reset(total=max(1, int(event["total_targets"])))
            bar.set_description(f"Worker {slot + 1}: {event['episode_uid']}")
            bar.set_postfix(frame_occurrences=0)
            return
        if event_type in {"episode_progress", "episode_complete"}:
            completed = int(event["completed_targets"])
            bar.n = min(completed, bar.total or completed)
            bar.set_postfix(
                frame_occurrences=int(event["target_frame_occurrences"]),
                max_mae_65d=f"{float(event['max_mae_65d']):.6f}",
                max_abs=f"{float(event['max_abs']):.6f}",
            )
            bar.refresh()
            return
        raise ValueError(f"Unknown progress event: {event_type!r}")


def process_episode_with_progress(episode_dir: Path, cfg: VerificationConfig, progress_queue: Any) -> dict[str, Any]:
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
    progress.set_postfix(
        targets=int(result["num_local_targets"]),
        max_mae_65d=f"{float(result['max_frame_mae_65d']):.6f}",
        max_abs=f"{float(result['max_frame_max_abs']):.6f}",
    )


def process_all(cfg: VerificationConfig) -> list[dict[str, Any]]:
    episode_dirs = list_episode_dirs(cfg)
    if not episode_dirs:
        return []
    worker_count = min(cfg.num_workers, len(episode_dirs))
    renderer = WorkerProgressRenderer(worker_count)
    results: list[dict[str, Any]] = []
    try:
        if worker_count == 1:
            with tqdm(total=len(episode_dirs), desc="Verify local delta-action targets", unit="episode", dynamic_ncols=True) as overall:
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
                    desc=f"Verify local delta-action targets ({worker_count} workers)",
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


def streaming_moments(values: np.memmap, chunk_size: int = 1_000_000) -> tuple[float, float, float, float]:
    count = int(values.size)
    if count == 0:
        return 0.0, 0.0, 0.0, 0.0
    total = 0.0
    total_squared = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    for start in range(0, count, chunk_size):
        chunk = np.asarray(values[start : start + chunk_size], dtype=np.float64)
        total += float(np.sum(chunk))
        total_squared += float(np.dot(chunk, chunk))
        minimum = min(minimum, float(np.min(chunk)))
        maximum = max(maximum, float(np.max(chunk)))
    mean = total / count
    variance = max(0.0, total_squared / count - mean * mean)
    return mean, float(np.sqrt(variance)), minimum, maximum


def percentile_key(percentile: float) -> str:
    if percentile == int(percentile):
        return f"p{int(percentile)}"
    return f"p{str(percentile).replace('.', '_')}"


def exact_distribution(values_path: Path) -> dict[str, float | int]:
    values = np.load(values_path, mmap_mode="r+")
    count = int(values.size)
    if count == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, **{percentile_key(p): 0.0 for p in PERCENTILES}}
    mean, std, minimum, maximum = streaming_moments(values)
    distribution: dict[str, float | int] = {"count": count, "mean": mean, "std": std, "p0": minimum, "p100": maximum}
    for percentile in PERCENTILES[1:-1]:
        position = (count - 1) * percentile / 100.0
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        values.partition((lower, upper))
        lower_value = float(values[lower])
        upper_value = float(values[upper])
        distribution[percentile_key(percentile)] = lower_value + (position - lower) * (upper_value - lower_value)
    return distribution


def merge_metric_files(results: list[dict[str, Any]], key: str, output_path: Path) -> Path:
    paths = [Path(str(result[key])) for result in results]
    total = sum(int(np.load(path, mmap_mode="r").size) for path in paths)
    merged = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=(total,))
    cursor = 0
    for path in paths:
        values = np.load(path, mmap_mode="r")
        next_cursor = cursor + values.size
        merged[cursor:next_cursor] = values
        cursor = next_cursor
    merged.flush()
    del merged
    return output_path


def format_distribution(distribution: dict[str, float | int]) -> str:
    keys = ("count", "mean", "std", "p0", "p1", "p2", "p5", "p10", "p25", "p50", "p75", "p95", "p99", "p99_9", "p99_99", "p100")
    return ", ".join(
        f"{key}={distribution[key]}" if key == "count" else f"{key}={float(distribution[key]):.9g}"
        for key in keys
    )


def serialize_config(cfg: VerificationConfig) -> dict[str, Any]:
    output = asdict(cfg)
    output["dataset_root"] = str(cfg.dataset_root)
    output["output_dir"] = str(cfg.output_dir) if cfg.output_dir is not None else None
    output["scratch_dir"] = str(cfg.scratch_dir)
    return output


def write_summary(cfg: VerificationConfig, results: list[dict[str, Any]]) -> tuple[Path, dict[str, dict[str, float | int]]]:
    destination = output_root(cfg.dataset_root, cfg.output_dir, cfg.target_tag)
    destination.mkdir(parents=True, exist_ok=True)
    reported_results = [
        {key: value for key, value in result.items() if not key.endswith("_values_path")}
        for result in results
    ]
    atomic_parquet(pd.DataFrame(reported_results), destination / f"{cfg.target_tag}_per_episode.parquet")
    distributions: dict[str, dict[str, float | int]] = {}
    metric_specs = (
        ("local_target_frame_mae_65d", "mae_values_path"),
        ("local_target_frame_max_absolute_dimension_error", "max_abs_values_path"),
        ("local_target_frame_count_per_spline", "frame_count_values_path"),
    )
    for name, result_key in metric_specs:
        merged_path = cfg.scratch_dir / f"merged_{name}.npy"
        distributions[name] = exact_distribution(merge_metric_files(results, result_key, merged_path))
        if not cfg.keep_intermediate_error_arrays:
            merged_path.unlink()

    summary = {
        "config": serialize_config(cfg),
        "episodes_verified": len(results),
        "local_targets_verified": sum(int(result["num_local_targets"]) for result in results),
        "target_frame_occurrences_verified": sum(int(result["target_frame_occurrences"]) for result in results),
        "worst_anchor_state_abs_error": max(
            (float(result["max_anchor_state_abs_error"]) for result in results), default=0.0
        ),
        "episodes_exceeding_anchor_state_tolerance": sum(
            bool(result["anchor_state_tolerance_exceeded"]) for result in results
        ),
        "distributions": distributions,
        "per_episode_results": reported_results,
    }
    summary_path = destination / f"{cfg.target_tag}_verification_summary.json"
    atomic_json(summary, summary_path)
    return summary_path, distributions


def cleanup_scratch(cfg: VerificationConfig) -> None:
    if cfg.keep_intermediate_error_arrays:
        return
    if not cfg.scratch_dir.exists():
        return
    resolved_scratch = cfg.scratch_dir.resolve()
    resolved_output = output_root(cfg.dataset_root, cfg.output_dir, cfg.target_tag).resolve()
    if resolved_output not in resolved_scratch.parents or not resolved_scratch.name.startswith(".scratch_"):
        raise RuntimeError(f"Refusing to remove unexpected scratch path: {resolved_scratch}")
    shutil.rmtree(resolved_scratch)


def main() -> int:
    cfg = load_config(parse_args())
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(cfg.dataset_root)
    print(f"Dataset root              : {cfg.dataset_root}")
    print(f"Local target NPZ          : {cfg.local_target_npz_name}")
    print(f"Local target index        : {cfg.local_target_index_name}")
    print(f"Required local degree     : {EXPECTED_LOCAL_DEGREE}")
    print(f"Num workers               : {cfg.num_workers}")
    print(f"Max episodes              : {cfg.max_episodes if cfg.max_episodes is not None else 'all'}")
    print(f"Output directory          : {output_root(cfg.dataset_root, cfg.output_dir, cfg.target_tag)}")

    try:
        results = process_all(cfg)
        if not results:
            raise RuntimeError("No eligible episodes found with action/state and local target artifacts.")
        summary_path, distributions = write_summary(cfg, results)
    finally:
        cleanup_scratch(cfg)

    print(f"Episodes verified         : {len(results)}")
    print(f"Local targets verified    : {sum(int(result['num_local_targets']) for result in results)}")
    print(f"Target frame occurrences  : {sum(int(result['target_frame_occurrences']) for result in results)}")
    print(
        "Worst anchor state error : "
        f"{max((float(result['max_anchor_state_abs_error']) for result in results), default=0.0):.3e}"
    )
    print(f"Local target-frame MAE-65D: {format_distribution(distributions['local_target_frame_mae_65d'])}")
    print(
        "Local target-frame max abs dimension error: "
        f"{format_distribution(distributions['local_target_frame_max_absolute_dimension_error'])}"
    )
    print(
        "Local target frame count per spline: "
        f"{format_distribution(distributions['local_target_frame_count_per_spline'])}"
    )
    print(f"Verification summary       : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
