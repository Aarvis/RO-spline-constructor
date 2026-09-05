from __future__ import annotations

import argparse
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from multiprocessing import Manager
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from create_global_action_splines import SplineConfig, list_episode_dirs, process_episode, write_run_summary


SUMMARY_STATISTICS = ("mean", "median", "p1", "p5", "p25", "p75", "p95", "p99", "min", "max")
SUMMARY_METRICS = {
    "control_points_per_frame": "control_points_per_frame",
    "max_mae_65d": "max_mae",
    "stop_max": "max_error",
}


@dataclass(frozen=True)
class MultiDegreeConfig:
    base_spline: SplineConfig
    degrees: tuple[int, ...]
    degree_output_dir_pattern: str
    summary_csv_name: str
    episode_results_csv_name: str
    summary_yaml_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the same action_65d episodes for multiple clamped B-spline degrees."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML config path.")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Processed dataset root override.")
    parser.add_argument("--degrees", type=int, nargs="+", default=None, help="Spline degree overrides.")
    parser.add_argument("--epsilon", type=float, default=None, help="Max frame reconstruction error target.")
    parser.add_argument("--max-internal-knots", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None, help="Maximum number of eligible episodes to process.")
    parser.add_argument("--output-dir", type=Path, default=None, help="External output root override.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> MultiDegreeConfig:
    with args.config.open("r", encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    spline = raw.get("spline", {})
    output = raw.get("output", {})
    degrees_value = args.degrees if args.degrees is not None else spline.get("degrees")
    if not degrees_value:
        raise ValueError("spline.degrees must contain at least one degree, for example: [3, 4, 5]")
    degrees = tuple(dict.fromkeys(int(degree) for degree in degrees_value))
    if any(degree < 1 for degree in degrees):
        raise ValueError(f"Spline degrees must be positive integers, got {list(degrees)}")

    dataset_root_value = args.dataset_root or raw.get("dataset_root") or raw.get("input_root")
    if dataset_root_value is None:
        raise ValueError("dataset_root is required in config or --dataset-root")

    error_metric = str(spline.get("error_metric", "mae_65d")).strip().lower()
    valid_metrics = {"rmse_65d", "mae_65d", "max_abs_joint"}
    if error_metric not in valid_metrics:
        raise ValueError(f"Unsupported spline.error_metric={error_metric!r}. Expected one of {sorted(valid_metrics)}.")

    output_dir_value = args.output_dir or output.get("output_dir")
    if not output_dir_value:
        raise ValueError("output.output_dir is required so each spline degree can be written separately.")
    output_dir = Path(output_dir_value)

    base_spline = SplineConfig(
        dataset_root=Path(dataset_root_value),
        degree=degrees[0],
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
    return MultiDegreeConfig(
        base_spline=base_spline,
        degrees=degrees,
        degree_output_dir_pattern=str(output.get("degree_output_dir_pattern", "degree_{degree}")),
        summary_csv_name=str(output.get("summary_csv_name", "multi_degree_action_spline_summary.csv")),
        episode_results_csv_name=str(output.get("episode_results_csv_name", "multi_degree_action_spline_episode_results.csv")),
        summary_yaml_name=str(output.get("summary_yaml_name", "multi_degree_action_spline_summary.yaml")),
    )


def select_episode_dirs(cfg: SplineConfig) -> list[Path]:
    episode_dirs = list_episode_dirs(cfg.dataset_root, cfg.target_array_name)
    if cfg.max_episodes is not None:
        if cfg.max_episodes < 1:
            raise ValueError("max_episodes must be >= 1 when provided")
        episode_dirs = episode_dirs[: cfg.max_episodes]
    if not episode_dirs:
        raise RuntimeError("No eligible episodes found.")
    return episode_dirs


def update_completion_progress(progresses: list[tqdm], result: dict[str, Any]) -> None:
    if result["status"] != "processed":
        return
    for progress in progresses:
        progress.set_postfix(
            stop_max=f"{float(result['max_error']):.6f}",
            max_mae_65d=f"{float(result['max_mae']):.6f}",
            control_points_per_frame=f"{float(result['control_points_per_frame']):.4f}",
        )


class WorkerProgressRenderer:
    """Render worker events in the parent process so multiprocessing output stays readable."""

    def __init__(self, num_workers: int, position: int, frame_progress: tqdm) -> None:
        self.frame_progress = frame_progress
        self.worker_slots: dict[int, int] = {}
        self.bars = [
            tqdm(
                total=1,
                desc=f"Worker {slot + 1}: idle",
                unit="iter",
                leave=False,
                dynamic_ncols=True,
                position=position + slot,
            )
            for slot in range(num_workers)
        ]

    def close(self) -> None:
        for bar in self.bars:
            bar.close()

    def worker_bar(self, worker_id: int) -> tqdm:
        if worker_id not in self.worker_slots:
            if len(self.worker_slots) >= len(self.bars):
                raise RuntimeError("Received progress events from more workers than configured.")
            self.worker_slots[worker_id] = len(self.worker_slots)
        return self.bars[self.worker_slots[worker_id]]

    def handle_event(self, event: dict[str, Any]) -> None:
        worker_id = int(event.get("worker_id", 0))
        bar = self.worker_bar(worker_id)
        event_type = str(event["event"])
        episode_uid = str(event["episode_uid"])
        if event_type == "fit_start":
            bar.reset(total=max(1, int(event["total_iterations"])))
            bar.set_description(f"Worker {self.worker_slots[worker_id] + 1}: {episode_uid}")
            bar.set_postfix(frames=int(event["num_frames"]))
            return
        if event_type == "fit_update":
            bar.n = min(int(event["iteration"]), int(event["total_iterations"]))
            bar.set_postfix(
                frames=int(event["num_frames"]),
                stop_max=f"{float(event['stop_max']):.6f}",
                max_mae_65d=f"{float(event['max_mae_65d']):.6f}",
                control_points_per_frame=f"{float(event['control_points_per_frame']):.4f}",
            )
            bar.refresh()
            # One fit iteration evaluates every frame in the active episode.
            self.frame_progress.update(int(event["num_frames"]))
            return
        if event_type == "fit_complete":
            bar.n = min(int(event["iteration"]), int(event["total_iterations"]))
            bar.refresh()
            return
        raise ValueError(f"Unknown progress event: {event_type!r}")


def process_episode_with_progress(episode_dir: Path, cfg: SplineConfig, progress_queue: Any) -> dict[str, Any]:
    worker_id = os.getpid()

    def publish(event: dict[str, Any]) -> None:
        progress_queue.put({"worker_id": worker_id, **event})

    return process_episode(episode_dir, cfg, progress_callback=publish)


def drain_progress_events(progress_queue: Any, renderer: WorkerProgressRenderer) -> None:
    while True:
        try:
            renderer.handle_event(progress_queue.get_nowait())
        except Empty:
            return


def process_degree(
    episode_dirs: list[Path],
    cfg: SplineConfig,
    overall_progress: tqdm,
    frame_progress: tqdm,
) -> list[dict[str, Any]]:
    description = f"Fit degree-{cfg.degree} action splines"
    if cfg.num_workers == 1:
        results = []
        renderer = WorkerProgressRenderer(num_workers=1, position=3, frame_progress=frame_progress)
        try:
            with tqdm(total=len(episode_dirs), desc=description, unit="episode", dynamic_ncols=True, position=1) as progress:
                for episode_dir in episode_dirs:
                    result = process_episode(
                        episode_dir,
                        cfg,
                        progress_callback=lambda event: renderer.handle_event({"worker_id": os.getpid(), **event}),
                    )
                    results.append(result)
                    progress.update(1)
                    overall_progress.update(1)
                    update_completion_progress([progress, overall_progress], result)
        finally:
            renderer.close()
        return results

    results = []
    worker_count = min(cfg.num_workers, len(episode_dirs))
    renderer = WorkerProgressRenderer(num_workers=worker_count, position=3, frame_progress=frame_progress)
    try:
        with Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(process_episode_with_progress, episode_dir, cfg, progress_queue): episode_dir
                    for episode_dir in episode_dirs
                }
                with tqdm(
                    total=len(futures),
                    desc=f"{description} ({worker_count} workers)",
                    unit="episode",
                    dynamic_ncols=True,
                    position=1,
                ) as progress:
                    while futures:
                        completed, _ = wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                        drain_progress_events(progress_queue, renderer)
                        for future in completed:
                            episode_dir = futures.pop(future)
                            try:
                                result = future.result()
                            except Exception:
                                tqdm.write(f"Failed degree-{cfg.degree} episode: {episode_dir.name}")
                                raise
                            results.append(result)
                            progress.update(1)
                            overall_progress.update(1)
                            update_completion_progress([progress, overall_progress], result)
                    drain_progress_events(progress_queue, renderer)
    finally:
        renderer.close()
    return sorted(results, key=lambda row: row["episode_uid"])


def summarize_metric(values: np.ndarray) -> dict[str, float]:
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "mean": float(np.mean(values)),
        "median": float(percentiles[3]),
        "p1": float(percentiles[0]),
        "p5": float(percentiles[1]),
        "p25": float(percentiles[2]),
        "p75": float(percentiles[4]),
        "p95": float(percentiles[5]),
        "p99": float(percentiles[6]),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def build_summary_rows(degree: int, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed = [result for result in results if result["status"] == "processed"]
    rows = []
    for metric_name, source_key in SUMMARY_METRICS.items():
        values = np.asarray([float(result[source_key]) for result in processed], dtype=np.float64)
        if not len(values):
            stats = {name: float("nan") for name in SUMMARY_STATISTICS}
        else:
            stats = summarize_metric(values)
        rows.append(
            {
                "degree": degree,
                "metric": metric_name,
                "num_selected_episodes": len(results),
                "num_processed_episodes": len(processed),
                "num_skipped_episodes": len(results) - len(processed),
                **stats,
            }
        )
    return rows


def print_degree_summary(summary_rows: list[dict[str, Any]]) -> None:
    degree = summary_rows[0]["degree"]
    print(f"Degree {degree} distribution summary:")
    for row in summary_rows:
        print(
            f"  {row['metric']}: mean={row['mean']:.6f}, median={row['median']:.6f}, "
            f"p95={row['p95']:.6f}, max={row['max']:.6f}"
        )


def write_multi_degree_summary(
    cfg: MultiDegreeConfig,
    episode_dirs: list[Path],
    summary_rows: list[dict[str, Any]],
    episode_rows: list[dict[str, Any]],
) -> None:
    output_dir = cfg.base_spline.output_dir
    if output_dir is None:
        raise RuntimeError("Multi-degree output directory is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output_dir / cfg.summary_csv_name, index=False)
    pd.DataFrame(episode_rows).to_csv(output_dir / cfg.episode_results_csv_name, index=False)
    payload = {
        "dataset_root": str(cfg.base_spline.dataset_root),
        "output_root": str(output_dir),
        "degrees": list(cfg.degrees),
        "episode_uids": [episode_dir.name for episode_dir in episode_dirs],
        "num_selected_episodes": len(episode_dirs),
        "epsilon": cfg.base_spline.epsilon,
        "error_metric": cfg.base_spline.error_metric,
        "statistics": list(SUMMARY_STATISTICS),
        "metrics": list(SUMMARY_METRICS),
        "degree_summary": summary_rows,
    }
    (output_dir / cfg.summary_yaml_name).write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    if not cfg.base_spline.dataset_root.exists():
        raise FileNotFoundError(cfg.base_spline.dataset_root)

    episode_dirs = select_episode_dirs(cfg.base_spline)
    print(f"Dataset root       : {cfg.base_spline.dataset_root}")
    print(f"Selected episodes  : {len(episode_dirs)}")
    print(f"Spline degrees     : {list(cfg.degrees)}")
    print(f"Epsilon            : {cfg.base_spline.epsilon}")
    print(f"Error metric       : {cfg.base_spline.error_metric}")
    print(f"Output root        : {cfg.base_spline.output_dir}")

    summary_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    total_jobs = len(cfg.degrees) * len(episode_dirs)
    with tqdm(
        total=total_jobs,
        desc="Overall multi-degree fits",
        unit="episode",
        dynamic_ncols=True,
        position=0,
    ) as overall_progress:
        # Adaptive fitting re-evaluates all episode frames each iteration, so its total is data-dependent.
        with tqdm(
            total=None,
            desc="Frame evaluations",
            unit="frame",
            dynamic_ncols=True,
            position=2,
        ) as frame_progress:
            for degree in cfg.degrees:
                degree_dir_name = cfg.degree_output_dir_pattern.format(degree=degree)
                degree_output_dir = cfg.base_spline.output_dir / degree_dir_name
                degree_cfg = replace(cfg.base_spline, degree=degree, output_dir=degree_output_dir)
                tqdm.write(f"Degree {degree} output : {degree_output_dir}")
                results = process_degree(episode_dirs, degree_cfg, overall_progress, frame_progress)
                write_run_summary(degree_cfg, results)
                degree_summary_rows = build_summary_rows(degree, results)
                summary_rows.extend(degree_summary_rows)
                print_degree_summary(degree_summary_rows)
                episode_rows.extend({"degree": degree, **result} for result in results)

    write_multi_degree_summary(cfg, episode_dirs, summary_rows, episode_rows)
    print(f"\nMulti-degree summary: {cfg.base_spline.output_dir / cfg.summary_csv_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
