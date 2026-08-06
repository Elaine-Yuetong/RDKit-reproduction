"""Config-driven Phase 4 coverage-sampling benchmark.

This is a thin wrapper around `phase4_generation.coverage_sampling_eval`. It
standardizes scenarios, pickers, metrics, and output formats without copying
the existing picker, eye-training, evaluation, or aggregation code.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from phase4_generation import coverage_sampling_eval as cse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("benchmark_config.yaml")
RESULTS_DIR = ROOT / "results"
CSV_OUT = RESULTS_DIR / "benchmark_results.csv"
JSON_OUT = RESULTS_DIR / "benchmark_results.json"
HTML_OUT = Path(__file__).with_name("report.html")

PICKER_ALIASES = {
    "uncertainty": "uncertainty_distance",
}

METRIC_TO_RESULT_KEY = {
    "entropy": "normalized_entropy",
    "occupied_cells": "occupied_cells",
    "mean_nn": "mean_nn_distance",
    "min_nn": "min_nn_distance",
    "coverage_radius": "coverage_radius",
}

METRIC_LABELS = {
    "entropy": "Entropy",
    "occupied_cells": "Occupied Cells",
    "mean_nn": "Mean NN Distance",
    "min_nn": "Min NN Distance",
    "coverage_radius": "Coverage Radius",
}


def load_config(path: Path) -> dict[str, Any]:
    """Load benchmark YAML config, with an explicit message if PyYAML is absent."""
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Missing dependency: install PyYAML with `./venv/bin/pip install pyyaml`.") from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return config


def _as_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_pickers(raw_pickers: list[str] | None) -> list[str]:
    """Map config-friendly picker names onto coverage_sampling_eval's names."""
    if not raw_pickers:
        return list(cse.PICKER_ORDER)
    normalized = [PICKER_ALIASES.get(str(name), str(name)) for name in raw_pickers]
    unknown = sorted(set(normalized) - set(cse.PICKER_ORDER))
    if unknown:
        raise ValueError(f"unknown picker(s): {unknown}; valid={cse.PICKER_ORDER}")
    return normalized


def normalize_metrics(raw_metrics: list[str] | None) -> list[str]:
    """Validate metric names and keep the config-facing names."""
    if not raw_metrics:
        return list(METRIC_TO_RESULT_KEY)
    raw = [str(name) for name in raw_metrics]
    unknown = sorted(set(raw) - set(METRIC_TO_RESULT_KEY))
    if unknown:
        raise ValueError(f"unknown metric(s): {unknown}; valid={sorted(METRIC_TO_RESULT_KEY)}")
    return raw


def build_eval_args(config: dict[str, Any], scenario: str, smoke: bool) -> Any:
    """Build an args object compatible with coverage_sampling_eval.run_one_seed()."""
    args = cse.build_parser().parse_args([])
    data_cfg = config.get("data", {})
    grid_cfg = config.get("grid", {})
    eval_cfg = config.get("eval", {})
    eyes_cfg = config.get("eyes", {})

    args.scenario = scenario
    if "cache" in data_cfg:
        args.cache = _as_path(data_cfg["cache"])
    if "min_atom_dist" in data_cfg:
        args.min_atom_dist = float(data_cfg["min_atom_dist"])
    if "coverage_map" in grid_cfg:
        args.coverage_map = _as_path(grid_cfg["coverage_map"])
    if "budget" in eval_cfg:
        args.budget = int(eval_cfg["budget"])
    if "n_seeds" in eval_cfg:
        args.n_seeds = int(eval_cfg["n_seeds"])
    if "seed" in eval_cfg:
        args.seed = int(eval_cfg["seed"])
    if "seed_frac" in eval_cfg:
        args.seed_frac = float(eval_cfg["seed_frac"])
    if "model" in eyes_cfg:
        args.model = str(eyes_cfg["model"])
    if "lr" in eyes_cfg:
        args.eyes_lr = float(eyes_cfg["lr"])
    if "grad_clip" in eyes_cfg:
        args.grad_clip = float(eyes_cfg["grad_clip"])
    if "epochs" in eyes_cfg:
        args.epochs = int(eyes_cfg["epochs"])
    if "batch_size" in eyes_cfg:
        args.batch_size = int(eyes_cfg["batch_size"])
    if "patience" in eyes_cfg:
        args.patience = int(eyes_cfg["patience"])

    args.smoke = bool(smoke)
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.patience = min(args.patience, 3)
        args.budget = min(args.budget, 30)
        args.n_seeds = min(args.n_seeds, 1)
    return args


def run_scenario(
    config: dict[str, Any],
    scenario: str,
    torch,
    Data,
    DataLoader,
    device,
    smoke: bool,
) -> dict[str, Any]:
    """Run one scenario by reusing coverage_sampling_eval."""
    args = build_eval_args(config, scenario, smoke)
    grid = cse._load_grid(args.coverage_map)
    expected_bins = config.get("grid", {}).get("n_bins")
    if expected_bins is not None and int(grid.get("n_bins", expected_bins)) != int(expected_bins):
        raise ValueError(
            f"grid.n_bins={expected_bins} but coverage map has n_bins={grid.get('n_bins')}"
        )

    seeds = [int(args.seed + k) for k in range(args.n_seeds)]
    seed_results = [
        cse.run_one_seed(args, run_seed, grid, torch, Data, DataLoader, device, make_figure=False)
        for run_seed in seeds
    ]
    aggregate = cse._aggregate_seed_results(seed_results)
    return {
        "scenario": scenario,
        "budget": int(seed_results[0]["budget"]),
        "n_seeds": int(args.n_seeds),
        "seeds": seeds,
        "n_bins": int(grid["n_bins"]),
        "coverage_map": str(args.coverage_map),
        "cache": str(args.cache),
        "per_seed_results": seed_results,
        "aggregate": aggregate,
    }


def tidy_rows(
    results_by_scenario: dict[str, dict[str, Any]],
    pickers: list[str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Convert aggregate benchmark results to long-format rows."""
    rows: list[dict[str, Any]] = []
    for scenario, result in results_by_scenario.items():
        aggregate = result["aggregate"]
        for picker in pickers:
            picker_stats = aggregate["pickers"][picker]
            for metric in metrics:
                result_key = METRIC_TO_RESULT_KEY[metric]
                stats = picker_stats[result_key]
                rows.append(
                    {
                        "scenario": scenario,
                        "picker": picker,
                        "metric": metric,
                        "mean": float(stats["mean"]),
                        "std": float(stats["std"]),
                        "n_seeds": int(result["n_seeds"]),
                    }
                )
    return rows


def save_tables(rows: list[dict[str, Any]]) -> None:
    """Save long-format rows as CSV and JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario", "picker", "metric", "mean", "std", "n_seeds"]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    JSON_OUT.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fmt_number(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.4f}"


def print_pivot_summary(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    """Print scenario x picker rows with metric columns."""
    by_key = {(row["scenario"], row["picker"], row["metric"]): row for row in rows}
    scenario_picker = sorted({(row["scenario"], row["picker"]) for row in rows})
    print("\nStandardized Phase 4 benchmark")
    print(f"{'scenario':12s} {'picker':24s} " + " ".join(f"{m:>18s}" for m in metrics))
    print("-" * (38 + 19 * len(metrics)))
    for scenario, picker in scenario_picker:
        values = []
        for metric in metrics:
            row = by_key[(scenario, picker, metric)]
            values.append(f"{_fmt_number(row['mean'])}+/-{_fmt_number(row['std'])}")
        print(f"{scenario:12s} {picker:24s} " + " ".join(f"{value:>18s}" for value in values))


def write_html_report(rows: list[dict[str, Any]], metrics: list[str]) -> None:
    """Write a self-contained HTML report with embedded result data."""
    key_metrics = [metric for metric in ["entropy", "coverage_radius"] if metric in metrics]
    if not key_metrics:
        key_metrics = metrics[:2]
    scenarios = sorted({row["scenario"] for row in rows})
    pickers = sorted({row["picker"] for row in rows})
    payload = {
        "rows": rows,
        "scenarios": scenarios,
        "pickers": pickers,
        "metrics": key_metrics,
        "metricLabels": {metric: METRIC_LABELS.get(metric, metric) for metric in key_metrics},
    }
    data_json = json.dumps(payload, sort_keys=True)
    title = "Phase 4 Coverage Benchmark"
    chart_blocks = "\n".join(
        f'<section><h2>{html.escape(scenario)}</h2><canvas id="chart-{html.escape(scenario)}"></canvas></section>'
        for scenario in scenarios
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172033;
      background: #f7f8fb;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 8px;
    }}
    p {{
      margin: 0 0 22px;
      color: #4d5870;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d9deea;
      border-radius: 8px;
      margin: 18px 0;
      padding: 18px;
    }}
    h2 {{
      font-size: 18px;
      margin: 0 0 14px;
    }}
    canvas {{
      width: 100%;
      min-height: 360px;
    }}
  </style>
</head>
<body>
<main>
  <h1>{title}</h1>
  <p>Grouped bars compare picker means by scenario. Entropy is higher-is-better; coverage radius is lower-is-better.</p>
  {chart_blocks}
</main>
<script>
const benchmark = {data_json};
const palette = ["#4777c2", "#d77832", "#6c9f3d", "#b75d69", "#7f6ab4"];
const rowsByScenario = new Map();
for (const row of benchmark.rows) {{
  if (!benchmark.metrics.includes(row.metric)) continue;
  if (!rowsByScenario.has(row.scenario)) rowsByScenario.set(row.scenario, []);
  rowsByScenario.get(row.scenario).push(row);
}}
for (const scenario of benchmark.scenarios) {{
  const canvas = document.getElementById(`chart-${{scenario}}`);
  const scenarioRows = rowsByScenario.get(scenario) || [];
  const datasets = benchmark.metrics.map((metric, i) => ({{
    label: benchmark.metricLabels[metric] || metric,
    data: benchmark.pickers.map((picker) => {{
      const found = scenarioRows.find((row) => row.picker === picker && row.metric === metric);
      return found ? found.mean : null;
    }}),
    backgroundColor: palette[i % palette.length],
  }}));
  new Chart(canvas, {{
    type: "bar",
    data: {{ labels: benchmark.pickers, datasets }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{ y: {{ beginAtZero: true }} }},
      plugins: {{
        legend: {{ position: "top" }},
        tooltip: {{
          callbacks: {{
            afterLabel: (ctx) => {{
              const row = scenarioRows.find((item) =>
                item.picker === ctx.label && item.metric === benchmark.metrics[ctx.datasetIndex]
              );
              return row ? `std: ${{Number(row.std).toFixed(4)}}; n_seeds: ${{row.n_seeds}}` : "";
            }}
          }}
        }}
      }}
    }}
  }});
}}
</script>
</body>
</html>
"""
    HTML_OUT.write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standardized Phase 4 coverage benchmark.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke", action="store_true", help="Tiny benchmark gate for the GPU server.")
    return parser


def main() -> None:
    cli_args = build_parser().parse_args()
    config = load_config(cli_args.config)
    scenarios = [str(name) for name in config.get("scenarios", [])]
    if not scenarios:
        raise ValueError("config.scenarios must contain at least one scenario")
    unknown_scenarios = sorted(set(scenarios) - {"random", "source"})
    if unknown_scenarios:
        raise ValueError(f"unknown scenario(s): {unknown_scenarios}; valid=['random', 'source']")

    pickers = normalize_pickers(config.get("pickers"))
    metrics = normalize_metrics(config.get("metrics"))

    torch, Data, DataLoader = cse.import_torch_stack()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"config={cli_args.config}")
    print(f"scenarios={scenarios}")
    print(f"pickers={pickers}")
    print(f"metrics={metrics}")
    if cli_args.smoke:
        print("SMOKE mode enabled: n_seeds=1, epochs<=3, budget<=30")

    results_by_scenario = {}
    for scenario in scenarios:
        results_by_scenario[scenario] = run_scenario(
            deepcopy(config),
            scenario,
            torch,
            Data,
            DataLoader,
            device,
            smoke=cli_args.smoke,
        )

    rows = tidy_rows(results_by_scenario, pickers, metrics)
    save_tables(rows)
    write_html_report(rows, metrics)
    print_pivot_summary(rows, metrics)
    print(f"\nsaved_csv={CSV_OUT}")
    print(f"saved_json={JSON_OUT}")
    print(f"saved_html={HTML_OUT}")


if __name__ == "__main__":
    main()
