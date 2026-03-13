import os
import argparse
import json
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------
# Base parameter values (DO NOT EDIT)
# ---------------------------
BASE_PARAMS: dict[str, float] = {
    "v1": 4.8318, "q1a": 1.4266, "q3a": 8.9432, "q4a": 5.9277,
    "K1": 0.1943, "K2": 1.6138, "k1L": 0.2866,  "p1": 0.8672, "p1L": 0.2378,
    "d1": 0.7843, "q1b": 3.575, "q3b": 5.5899, "q4b": 8.954, "v2": 1.6822, "K3": 2.2275,
    "K4": 0.40, "K5": 0.37, "k2": 0.35, "p2": 0.7858, "d2L": 0.2917,
    "v3": 1.113, "K6": 0.4944, "K7": 2.4087, "k3": 0.5819, "p3": 0.6142,
    "d3L": 0.5431, "v4": 2.5012, "K8": 0.3262, "K9": 1.7974, "K10": 1.1889, "k4": 0.925,
    "p4": 1.126, "de1": 0.0022,
    "Ap3": 0.3868, "Am7": 0.5503, "Ak7": 1.125,  "kmpac": 137, "kd": 7,
    "v5": 0.1129, "K11": 0.3322, "k5": 0.1591, "p5": 0.5293,  "d5L": 5.0712,
    "g1": 0.001, "g2": 0.18, "K12": 0.86, "Bp4": 0.4147, "Bm8": 0.7728, "Bk8": 0.1732,
    "kmpbc": 7162, "Cp5": 0.4567, "Cm9": 0.867, "Ck9": 0.3237, "kmcc": 13406, "K13": 8.04866,
    "K14": 0.347569, "v6": 0.438785, "k6": 0.328256, "K15": 0.336558, "p6": 0.264064,
    "d6L": 0.190108, "de2": 0.4741, "de3": 0.3765, "de4": 0.398, "de5": 0.0003, "q2": 0.5767,
    "d3D": 0.5026, "d5D": 0.4404, "d6D": 0.576502, "d2D": 0.3712, "k1D": 0.213
}


class PhasePortraitAnalyzer:
    """Analyze phase portrait frames with 6 subplots and quantify area/eccentricity."""

    def __init__(self, base_dir, output_dir=None, base_params=None):
        self.base_dir = base_dir
        self.base_params = base_params or {}

        if output_dir is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
            self.output_dir = os.path.join(base_dir, f"area_ecc_analysis_{date_str}")
        else:
            self.output_dir = output_dir

        self.plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.parameter_folders = []
        self.parameter_frames = {}

        self.pairs = [
            ("CLm", "P97m"),
            ("CLm", "P51m"),
            ("CLm", "EL"),
            ("P97m", "P51m"),
            ("P97m", "EL"),
            ("P51m", "EL"),
        ]

    @staticmethod
    def _extract_param_value(filename, param_name):
        """Extract parameter value from a phase portrait filename."""
        base = os.path.splitext(os.path.basename(filename))[0]
        pattern = rf"^phase_{re.escape(param_name)}_(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)$"
        match = re.match(pattern, base)
        if match:
            return float(match.group("value"))

        tokens = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", base)
        if tokens:
            return float(tokens[-1])

        return None

    def load_parameter_folders(self):
        """Find all parameter folders in the base directory."""
        print("Loading parameter folders...")
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if (
                os.path.isdir(item_path)
                and item != os.path.basename(self.output_dir)
                and not item.startswith("analysis_results_")
                and not item.startswith("area_ecc_analysis_")
            ):
                self.parameter_folders.append(item)
        print(f"Found {len(self.parameter_folders)} parameter folders")

    def load_parameter_frames(self):
        """Load frame files for each parameter."""
        print("Loading parameter frames...")
        valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

        for param_folder in self.parameter_folders:
            folder_path = os.path.join(self.base_dir, param_folder)
            param_frames = []

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in valid_extensions):
                    try:
                        param_value = self._extract_param_value(file, param_folder)
                    except ValueError:
                        param_value = None

                    if param_value is not None:
                        param_frames.append({
                            "path": file_path,
                            "value": param_value,
                        })

            param_frames.sort(key=lambda x: x["value"])
            self.parameter_frames[param_folder] = param_frames
            print(f"  {param_folder}: {len(param_frames)} frames")

    @staticmethod
    def _split_subplots(image, rows=2, cols=3, border_frac=0.08):
        """Split a 2x3 phase portrait image into subplot tiles and crop borders."""
        h, w = image.shape[:2]
        tile_h = h // rows
        tile_w = w // cols
        tiles = []

        for r in range(rows):
            for c in range(cols):
                y0 = r * tile_h
                y1 = (r + 1) * tile_h
                x0 = c * tile_w
                x1 = (c + 1) * tile_w

                tile = image[y0:y1, x0:x1].copy()

                crop_x = int(tile_w * border_frac)
                crop_y = int(tile_h * border_frac)
                tile = tile[crop_y:tile.shape[0] - crop_y, crop_x:tile.shape[1] - crop_x]
                tiles.append(tile)

        return tiles

    @staticmethod
    def _line_mask(tile):
        """Create a binary mask for the plotted trajectory line."""
        if tile.ndim == 3:
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        else:
            gray = tile

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return mask

    @staticmethod
    def _area_and_eccentricity(mask):
        """Compute convex hull area and eccentricity from a binary mask."""
        ys, xs = np.where(mask > 0)
        if len(xs) < 10:
            return np.nan, np.nan

        points = np.column_stack([xs, ys]).astype(np.int32)
        hull = cv2.convexHull(points)
        area = float(cv2.contourArea(hull))

        coords = points.astype(float)
        mean = coords.mean(axis=0)
        centered = coords - mean
        cov = np.cov(centered, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(eigvals)[::-1]
        if eigvals[0] <= 0:
            ecc = np.nan
        else:
            ecc = float(np.sqrt(max(0.0, 1.0 - (eigvals[1] / eigvals[0]))))

        return area, ecc

    def analyze_frame(self, frame_path):
        """Analyze a single phase portrait frame and return per-subplot metrics."""
        frame = np.array(Image.open(frame_path))
        tiles = self._split_subplots(frame, rows=2, cols=3)

        metrics = []
        for tile in tiles:
            mask = self._line_mask(tile)
            area, ecc = self._area_and_eccentricity(mask)
            metrics.append({
                "area": area,
                "eccentricity": ecc,
            })

        return metrics

    def analyze_parameter_folder(self, param_name):
        """Analyze all frames for a parameter and compute change vs base."""
        print(f"Analyzing parameter: {param_name}...")
        frames = self.parameter_frames.get(param_name, [])
        if not frames:
            print(f"No frames found for parameter {param_name}")
            return None

        base_value = self.base_params.get(param_name)
        base_index = None
        if base_value is not None:
            values = np.array([f["value"] for f in frames], dtype=float)
            base_index = int(np.argmin(np.abs(values - base_value)))
            base_value = values[base_index]
        else:
            print(f"Warning: no base value provided for {param_name}; change metrics will be NaN.")

        per_frame = []
        for frame in frames:
            metrics = self.analyze_frame(frame["path"])
            per_frame.append({
                "param_value": frame["value"],
                "metrics": metrics,
            })

        if base_index is not None:
            base_metrics = per_frame[base_index]["metrics"]
        else:
            base_metrics = None

        rows_subplots = []
        rows_mean = []

        for frame_entry in per_frame:
            value = frame_entry["param_value"]
            if base_value is None or base_value == 0:
                fold_change = np.nan
            else:
                fold_change = value / base_value

            deltas_area = []
            deltas_ecc = []

            for idx, metric in enumerate(frame_entry["metrics"]):
                pair_label = f"{self.pairs[idx][0]} vs {self.pairs[idx][1]}"
                area = metric["area"]
                ecc = metric["eccentricity"]

                if base_metrics is None:
                    delta_area = np.nan
                    delta_ecc = np.nan
                else:
                    delta_area = area - base_metrics[idx]["area"]
                    delta_ecc = ecc - base_metrics[idx]["eccentricity"]

                deltas_area.append(delta_area)
                deltas_ecc.append(delta_ecc)

                rows_subplots.append({
                    "parameter": param_name,
                    "param_value": value,
                    "fold_change": fold_change,
                    "subplot_index": idx + 1,
                    "pair": pair_label,
                    "area": area,
                    "eccentricity": ecc,
                    "delta_area": delta_area,
                    "delta_eccentricity": delta_ecc,
                    "base_param_value": base_value,
                })

            mean_delta_area = float(np.nanmean(deltas_area)) if deltas_area else np.nan
            mean_delta_ecc = float(np.nanmean(deltas_ecc)) if deltas_ecc else np.nan

            mean_area = float(np.nanmean([m["area"] for m in frame_entry["metrics"]]))
            mean_ecc = float(np.nanmean([m["eccentricity"] for m in frame_entry["metrics"]]))

            rows_mean.append({
                "parameter": param_name,
                "param_value": value,
                "fold_change": fold_change,
                "mean_area": mean_area,
                "mean_eccentricity": mean_ecc,
                "mean_delta_area": mean_delta_area,
                "mean_delta_eccentricity": mean_delta_ecc,
                "base_param_value": base_value,
            })

        return {
            "parameter": param_name,
            "base_value": base_value,
            "subplot_rows": rows_subplots,
            "mean_rows": rows_mean,
        }

    def analyze_all_parameters(self):
        """Analyze all parameter folders."""
        if not self.parameter_folders:
            self.load_parameter_folders()
        if not self.parameter_frames:
            self.load_parameter_frames()

        all_subplots = []
        all_means = []
        all_results = []

        for param_name in self.parameter_folders:
            result = self.analyze_parameter_folder(param_name)
            if not result:
                continue

            all_subplots.extend(result["subplot_rows"])
            all_means.extend(result["mean_rows"])
            all_results.append(result)

        if not all_subplots:
            print("No results to save.")
            return

        df_subplots = pd.DataFrame(all_subplots)
        df_means = pd.DataFrame(all_means)

        df_subplots.to_csv(os.path.join(self.output_dir, "subplot_metrics.csv"), index=False)
        df_means.to_csv(os.path.join(self.output_dir, "mean_metrics.csv"), index=False)

        with open(os.path.join(self.output_dir, "analysis_overview.json"), "w") as f:
            json.dump({"parameters": [r["parameter"] for r in all_results]}, f, indent=2)

        self.plot_results(df_subplots, df_means)

        print(f"Analysis complete! Results saved to {self.output_dir}")

    def plot_results(self, df_subplots, df_means):
        """Plot per-subplot and mean changes versus fold change for each parameter."""
        for param_name in sorted(df_subplots["parameter"].unique()):
            df_param = df_subplots[df_subplots["parameter"] == param_name]
            df_mean = df_means[df_means["parameter"] == param_name]

            if df_param.empty:
                continue

            self._plot_change(
                df_param,
                df_mean,
                param_name,
                metric="delta_area",
                ylabel="Change in area (px^2)",
                filename=f"{param_name}_area_change.png",
            )

            self._plot_change(
                df_param,
                df_mean,
                param_name,
                metric="delta_eccentricity",
                ylabel="Change in eccentricity",
                filename=f"{param_name}_eccentricity_change.png",
            )

    def _plot_change(self, df_param, df_mean, param_name, metric, ylabel, filename):
        plt.figure(figsize=(8, 5))

        for idx, pair in enumerate(self.pairs, start=1):
            df_sub = df_param[df_param["subplot_index"] == idx].sort_values("fold_change")
            plt.plot(
                df_sub["fold_change"],
                df_sub[metric],
                marker="o",
                linewidth=1.2,
                label=f"{pair[0]} vs {pair[1]}",
            )

        df_mean_sorted = df_mean.sort_values("fold_change")
        plt.plot(
            df_mean_sorted["fold_change"],
            df_mean_sorted[f"mean_{metric}"],
            color="black",
            linewidth=2.0,
            marker="o",
            label="Mean",
        )

        plt.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        plt.xlabel("Fold change vs base", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.title(f"{param_name}: {ylabel}", fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(loc="best", fontsize=20)
        plt.grid(True, linestyle=":", linewidth=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, filename), dpi=300)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze phase portraits for area/eccentricity changes.")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="Path to base directory containing parameter folders with frames. Defaults to ./phase_portraits.",
    )
    parser.add_argument("--output_dir", default=None, help="Path where results should be saved")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = args.base_dir or os.path.join(script_dir, "phase_portraits")
    base_dir = os.path.abspath(os.path.expanduser(base_dir))

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    print("Running area/eccentricity analyzer...")
    analyzer = PhasePortraitAnalyzer(base_dir, args.output_dir, base_params=BASE_PARAMS)
    analyzer.load_parameter_folders()
    analyzer.load_parameter_frames()
    analyzer.analyze_all_parameters()


if __name__ == "__main__":
    main()
