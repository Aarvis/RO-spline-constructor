from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.interpolate import BSpline, make_lsq_spline
from tqdm.auto import tqdm


VALID_ERROR_METRICS = ("rmse_65d", "mae_65d", "max_abs_joint")


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
    save_delta_from_first_state: bool
    delta_spline_npz_name: str
    delta_spline_index_name: str
    run_summary_csv_name: str
    run_summary_yaml_name: str
    num_workers: int
    max_episodes: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit one adaptive clamped cubic B-spline over observation.state (state_65d) for every processed episode."
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
        degree=int(spline.get("degree", 3)),
        epsilon=float(args.epsilon if args.epsilon is not None else spline.get("epsilon", 0.02)),
        error_metric=error_metric,
        initial_internal_knots=int(spline.get("initial_internal_knots", 4)),
        max_internal_knots=int(
            args.max_internal_knots if args.max_internal_knots is not None else spline.get("max_internal_knots", 64)
        ),
        min_knot_spacing_frames=int(spline.get("min_knot_spacing_frames", 5)),
        endpoint_weight=float(spline.get("endpoint_weight", 1000.0)),
        overwrite=bool(args.overwrite or output.get("overwrite", False)),
        output_dir=output_dir,
        spline_npz_name=str(output.get("spline_npz_name", "global_reached_state_spline.npz")),
        spline_index_name=str(output.get("spline_index_name", "global_reached_state_spline_index.parquet")),
        target_array_name=str(output.get("target_array_name", "state_65d.npy")),
        state_array_name=str(output.get("state_array_name", raw.get("state_array_name", "state_65d.npy"))),
        save_delta_from_first_state=bool(output.get("save_delta_from_first_state", False)),
        delta_spline_npz_name=str(
            output.get("delta_spline_npz_name", "global_delta_reached_state_from_first_state_spline.npz")
        ),
        delta_spline_index_name=str(
            output.get("delta_spline_index_name", "global_delta_reached_state_from_first_state_spline_index.parquet")
        ),
        run_summary_csv_name=str(output.get("run_summary_csv_name", "global_reached_state_spline_run_summary.csv")),
        run_summary_yaml_name=str(output.get("run_summary_yaml_name", "global_reached_state_spline_run_summary.yaml")),
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
    result = []
    for episode_dir in sorted(p for p in episode_root.iterdir() if p.is_dir()):
        if (episode_dir / "arrays" / target_array_name).exists():
            result.append(episode_dir)
    return result


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
    return cfg.dataset_root / "metadata" / "global_reached_state_spline_run"


def output_arrays_dir_for_episode(episode_dir: Path, cfg: SplineConfig) -> Path:
    if cfg.output_dir is None:
        return episode_dir / "arrays"
    return cfg.output_dir / "episodes" / episode_dir.name / "arrays"


def make_clamped_knots(x: np.ndarray, internal_indices: list[int], degree: int) -> np.ndarray:
    internal = [float(x[i]) for i in sorted(set(internal_indices)) if 0 < i < len(x) - 1]
    return np.asarray(([0.0] * (degree + 1)) + internal + ([1.0] * (degree + 1)), dtype=np.float64)


def initial_internal_indices(n: int, count: int, degree: int) -> list[int]:
    max_internal = max(0, n - degree - 1)
    count = min(max(0, count), max_internal)
    if count == 0 or n <= 2:
        return []
    values = np.linspace(1, n - 2, count + 2, dtype=np.float64)[1:-1]
    return sorted({int(round(v)) for v in values if 0 < int(round(v)) < n - 1})


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


def choose_next_knot_index(
    errors: np.ndarray,
    existing: set[int],
    min_spacing: int,
) -> int | None:
    for idx in np.argsort(errors)[::-1]:
        idx_int = int(idx)
        if idx_int <= 0 or idx_int >= len(errors) - 1:
            continue
        if idx_int in existing:
            continue
        if any(abs(idx_int - old) < min_spacing for old in existing):
            continue
        return idx_int
    return None


def adaptive_fit_spline(
    y: np.ndarray,
    cfg: SplineConfig,
    episode_uid: str,
) -> dict[str, Any]:
    n = int(y.shape[0])
    degree = min(cfg.degree, n - 1)
    if degree < 1:
        raise ValueError(f"Need at least 2 samples to fit a spline, got {n}")

    x = np.linspace(0.0, 1.0, n, dtype=np.float64)
    max_possible_internal = max(0, n - degree - 1)
    max_internal = min(cfg.max_internal_knots, max_possible_internal)
    internal = initial_internal_indices(n, cfg.initial_internal_knots, degree)
    internal = internal[:max_internal]
    status = "max_knots_reached"
    last_exc: str | None = None
    spline: BSpline | None = None
    stop_error_per_frame: np.ndarray | None = None
    rmse_per_frame: np.ndarray | None = None
    mae_per_frame: np.ndarray | None = None
    max_abs_joint_error_per_frame: np.ndarray | None = None
    max_error = float("nan")
    mean_error = float("nan")
    max_rmse_65d = float("nan")
    mean_rmse_65d = float("nan")
    max_max_abs_joint_error = float("nan")
    mean_max_abs_joint_error = float("nan")
    max_mae = float("nan")
    mean_mae = float("nan")
    total_fit_iterations = max(1, max_internal - len(internal) + 1)
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
                spline = fit_once(x, y, internal, degree, cfg.endpoint_weight)
                pred = spline(x)
                stop_error_per_frame = compute_stop_error(y, pred, cfg.error_metric)
                rmse_per_frame = compute_frame_rmse(y, pred)
                mae_per_frame = compute_frame_mae(y, pred)
                max_abs_joint_error_per_frame = compute_frame_max_abs_joint_error(y, pred)
                max_error = float(np.max(stop_error_per_frame))
                mean_error = float(np.mean(stop_error_per_frame))
                max_rmse_65d = float(np.max(rmse_per_frame))
                mean_rmse_65d = float(np.mean(rmse_per_frame))
                max_max_abs_joint_error = float(np.max(max_abs_joint_error_per_frame))
                mean_max_abs_joint_error = float(np.mean(max_abs_joint_error_per_frame))
                max_mae = float(np.max(mae_per_frame))
                mean_mae = float(np.mean(mae_per_frame))
                episode_progress.update(1)
                episode_progress.set_postfix(
                    internal=f"{len(internal)}/{max_internal}",
                    stop_max=f"{max_error:.6f}",
                    stop_mean=f"{mean_error:.6f}",
                    max_mae=f"{max_mae:.6f}",
                )
                if max_error <= cfg.epsilon:
                    status = "epsilon_met"
                    break
                if len(internal) >= max_internal:
                    status = "max_knots_reached"
                    break
                next_idx = choose_next_knot_index(stop_error_per_frame, set(internal), cfg.min_knot_spacing_frames)
                if next_idx is None:
                    status = "no_valid_knot_candidate"
                    break
                internal.append(next_idx)
                internal = sorted(set(internal))
            except Exception as exc:
                last_exc = repr(exc)
                if internal:
                    internal = internal[:-1]
                    status = "fit_failed_after_knot_removal"
                    continue
                raise
    finally:
        episode_progress.close()

    if (
        spline is None
        or stop_error_per_frame is None
        or rmse_per_frame is None
        or mae_per_frame is None
        or max_abs_joint_error_per_frame is None
    ):
        raise RuntimeError("Spline fitting failed before producing a spline.")

    return {
        "spline": spline,
        "x": x,
        "internal_indices": np.asarray(sorted(internal), dtype=np.int64),
        "stop_error_per_frame": stop_error_per_frame.astype(np.float32),
        "rmse_per_frame": rmse_per_frame.astype(np.float32),
        "mae_per_frame": mae_per_frame.astype(np.float32),
        "max_abs_joint_error_per_frame": max_abs_joint_error_per_frame.astype(np.float32),
        "max_error": max_error,
        "mean_error": mean_error,
        "max_rmse_65d": max_rmse_65d,
        "mean_rmse_65d": mean_rmse_65d,
        "max_max_abs_joint_error": max_max_abs_joint_error,
        "mean_max_abs_joint_error": mean_max_abs_joint_error,
        "max_mae": max_mae,
        "mean_mae": mean_mae,
        "degree": degree,
        "status": status,
        "last_exception": last_exc,
    }


def process_episode(episode_dir: Path, cfg: SplineConfig) -> dict[str, Any]:
    arrays_dir = episode_dir / "arrays"
    output_arrays_dir = output_arrays_dir_for_episode(episode_dir, cfg)
    out_npz = output_arrays_dir / cfg.spline_npz_name
    out_index = output_arrays_dir / cfg.spline_index_name
    out_delta_npz = output_arrays_dir / cfg.delta_spline_npz_name
    out_delta_index = output_arrays_dir / cfg.delta_spline_index_name

    required_outputs = [out_npz, out_index]
    if cfg.save_delta_from_first_state:
        required_outputs.extend([out_delta_npz, out_delta_index])
    if all(path.exists() for path in required_outputs) and not cfg.overwrite:
        return {"episode_uid": episode_dir.name, "status": "skipped_existing", "num_splines": len(required_outputs) // 2}

    state = np.load(arrays_dir / cfg.target_array_name).astype(np.float32, copy=False)
    if state.ndim != 2:
        raise ValueError(f"{episode_dir.name}: expected 2D state array, got shape {state.shape}")
    if state.shape[0] < 2:
        raise ValueError(f"{episode_dir.name}: need at least 2 state frames, got {state.shape[0]}")

    num_frames = int(state.shape[0])
    fit = adaptive_fit_spline(state.astype(np.float64), cfg, episode_dir.name)
    spline: BSpline = fit["spline"]
    frame_indices = np.arange(num_frames, dtype=np.int64)
    output_arrays_dir.mkdir(parents=True, exist_ok=True)

    coefficients = np.asarray(spline.c, dtype=np.float64)
    npz_payload: dict[str, np.ndarray] = {
        "global_knots": spline.t.astype(np.float64),
        "global_coefficients": coefficients,
        "global_degree": np.asarray([fit["degree"]], dtype=np.int64),
        "spline_target_mode": np.asarray(["global_reached_state"], dtype="U64"),
        "stop_error_metric": np.asarray([cfg.error_metric], dtype="U32"),
        "frame_indices": frame_indices,
        "frame_to_u": fit["x"].astype(np.float64),
        "error_per_frame": fit["stop_error_per_frame"].astype(np.float32),
        "rmse_per_frame": fit["rmse_per_frame"].astype(np.float32),
        "mae_per_frame": fit["mae_per_frame"].astype(np.float32),
        "max_abs_joint_error_per_frame": fit["max_abs_joint_error_per_frame"].astype(np.float32),
        "internal_knot_frame_indices": fit["internal_indices"].astype(np.int64),
        "target_dim": np.asarray([state.shape[1]], dtype=np.int64),
    }
    timestamps_path = arrays_dir / "timestamps.npy"
    if timestamps_path.exists():
        timestamps = np.load(timestamps_path)
        if len(timestamps) == num_frames:
            npz_payload["timestamps"] = timestamps

    np.savez_compressed(out_npz, **npz_payload)

    if cfg.save_delta_from_first_state:
        reference_state = state[0].astype(np.float64)
        delta_payload = dict(npz_payload)
        delta_payload["global_coefficients"] = coefficients - reference_state[None, :]
        delta_payload["spline_target_mode"] = np.asarray(["global_delta_reached_state_from_first_state"], dtype="U64")
        delta_payload["reference_type"] = np.asarray(["state_65d[0]"], dtype="U64")
        delta_payload["reference_array_name"] = np.asarray([cfg.state_array_name], dtype="U128")
        delta_payload["reference_frame"] = np.asarray([0], dtype=np.int64)
        delta_payload["reference_state_65d"] = reference_state.astype(np.float64)
        delta_payload["absolute_reconstruction_rule"] = np.asarray(
            ["reached_state(u) = reference_state_65d + global_delta_reached_state_spline(u)"], dtype="U128"
        )
        np.savez_compressed(out_delta_npz, **delta_payload)

    row = {
        "episode_uid": episode_dir.name,
        "target": "global_reached_state_spline = state_65d",
        "target_array_name": cfg.target_array_name,
        "start_frame": 0,
        "end_frame_exclusive": num_frames,
        "num_frames": num_frames,
        "target_dim": int(state.shape[1]),
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
        "max_max_abs_joint_error": float(fit["max_max_abs_joint_error"]),
        "mean_max_abs_joint_error": float(fit["mean_max_abs_joint_error"]),
        "max_mae": float(fit["max_mae"]),
        "mean_mae": float(fit["mean_mae"]),
        "fit_status": fit["status"],
        "last_exception": fit["last_exception"],
        "source_episode_dir": str(episode_dir),
        "output_arrays_dir": str(output_arrays_dir),
        "npz_path": str(out_npz),
        "npz_key_prefix": "global",
        "spline_target_mode": "global_reached_state",
    }
    pd.DataFrame([row]).to_parquet(out_index, index=False)

    num_splines = 1
    if cfg.save_delta_from_first_state:
        delta_row = dict(row)
        delta_row.update(
            {
                "target": "global_delta_reached_state_from_first_state = state_65d - state_65d[0]",
                "reference_array_name": cfg.state_array_name,
                "reference_frame": 0,
                "reconstruction_rule": "reached_state(u) = reference_state_65d + global_delta_reached_state_spline(u)",
                "npz_path": str(out_delta_npz),
                "spline_target_mode": "global_delta_reached_state_from_first_state",
            }
        )
        pd.DataFrame([delta_row]).to_parquet(out_delta_index, index=False)
        num_splines = 2

    return {
        "episode_uid": episode_dir.name,
        "status": "processed",
        "num_splines": num_splines,
        "max_error": row["max_error"],
        "max_mae": row["max_mae"],
        "fit_status": row["fit_status"],
    }


def process_all(cfg: SplineConfig) -> list[dict[str, Any]]:
    episode_dirs = list_episode_dirs(cfg.dataset_root, cfg.target_array_name)
    if cfg.max_episodes is not None:
        if cfg.max_episodes < 1:
            raise ValueError("max_episodes must be >= 1 when provided")
        episode_dirs = episode_dirs[: cfg.max_episodes]
    if cfg.num_workers == 1:
        results = []
        for episode_dir in tqdm(episode_dirs, desc="Fit global reached-state splines", unit="episode", dynamic_ncols=True):
            results.append(process_episode(episode_dir, cfg))
        return results

    results = []
    with ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {executor.submit(process_episode, episode_dir, cfg): episode_dir for episode_dir in episode_dirs}
        with tqdm(
            total=len(futures),
            desc=f"Fit global reached-state splines ({cfg.num_workers} workers)",
            unit="episode",
            dynamic_ncols=True,
        ) as progress:
            for future in as_completed(futures):
                episode_dir = futures[future]
                try:
                    results.append(future.result())
                except Exception:
                    tqdm.write(f"Failed episode: {episode_dir.name}")
                    raise
                progress.update(1)
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
        "save_delta_from_first_state": cfg.save_delta_from_first_state,
        "num_workers": cfg.num_workers,
        "max_episodes": cfg.max_episodes,
        "num_results": len(results),
        "results": results,
    }
    (summary_root / cfg.run_summary_yaml_name).write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    cfg = load_config(args)
    if not cfg.dataset_root.exists():
        raise FileNotFoundError(cfg.dataset_root)

    print(f"Dataset root       : {cfg.dataset_root}")
    print(f"target array       : {cfg.target_array_name}")
    print(f"epsilon           : {cfg.epsilon}")
    print(f"error metric      : {cfg.error_metric}")
    print(f"max internal knots: {cfg.max_internal_knots}")
    print(f"num workers       : {cfg.num_workers}")
    print(f"max episodes      : {cfg.max_episodes if cfg.max_episodes is not None else 'all'}")
    print(f"overwrite         : {cfg.overwrite}")
    print(f"output root       : {cfg.output_dir if cfg.output_dir is not None else 'dataset episode arrays'}")
    print(f"output npz        : {cfg.spline_npz_name}")
    print(f"output index      : {cfg.spline_index_name}")
    print(f"save delta spline : {cfg.save_delta_from_first_state}")
    if cfg.save_delta_from_first_state:
        print(f"delta output npz  : {cfg.delta_spline_npz_name}")
        print(f"delta output index: {cfg.delta_spline_index_name}")

    results = process_all(cfg)
    write_run_summary(cfg, results)
    processed = sum(1 for r in results if r["status"] == "processed")
    skipped = sum(1 for r in results if r["status"].startswith("skipped"))
    print(f"Episodes found      : {len(results)}")
    print(f"Processed episodes  : {processed}")
    print(f"Skipped episodes    : {skipped}")
    if results:
        max_error = max(float(r.get("max_error") or 0.0) for r in results)
        max_mae = max(float(r.get("max_mae") or 0.0) for r in results)
        print(f"Worst max error     : {max_error:.6f}")
        print(f"Worst max mae       : {max_mae:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
