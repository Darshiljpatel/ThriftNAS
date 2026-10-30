"""ThriftNAS Interactive Web Demo (PROJECT_SPEC §7, Step 12).

Offline, CPU-based interactive demonstration with 4 tabs:
1. Classify: image classification with top-3 predictions and measured latency
2. Compare all models: side-by-side inference across all models
3. Results: publication metrics and Pareto frontier figures
4. Search replay: architecture parameter evolution and discovered cell genotypes
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from PIL import Image

from thriftnas.models.genotypes import load_genotype, plot_genotype
from thriftnas.utils.inference import (
    CLASSES,
    compare_all_models,
    get_available_model_names,
    get_model_metadata,
    load_model,
    measure_inference_latency,
    predict,
)

# ---------------------------------------------------------------------------
# Example images from CIFAR-10 test set
# ---------------------------------------------------------------------------

EXAMPLE_DIR = Path("data/demo_examples")
EXAMPLES: list[list[str]] = []
if EXAMPLE_DIR.exists():
    for f in sorted(EXAMPLE_DIR.glob("*.png")):
        EXAMPLES.append([str(f)])


# ---------------------------------------------------------------------------
# Tab 1: Classify logic
# ---------------------------------------------------------------------------

def classify_image(
    image: Image.Image | None,
    model_name: str,
) -> tuple[dict[str, float] | str, str, str]:
    """Run inference on an uploaded image with the chosen model."""
    if image is None:
        return "Please upload or select an image.", "-", ""

    model, err = load_model(model_name)
    if err or model is None:
        return f"Model Unavailable: {err}", "-", ""

    # Predict top-3
    top3 = predict(model, image)
    probs_dict = {cls_name: round(prob, 4) for cls_name, prob in top3}

    # Measure CPU latency (median of 20 runs)
    lat_ms = measure_inference_latency(model, image, runs=20)
    lat_str = f"{lat_ms:.2f} ms"

    # Read published model metadata from results.csv
    meta = get_model_metadata(model_name)
    info_md = (
        f"**Model:** {model_name}  \n"
        f"**Top-1 Accuracy:** {meta['Top-1 (%)']}  \n"
        f"**Parameters:** {meta['Params (M)']} M  \n"
        f"**Complexity:** {meta['MACs (M)']} MACs (M)  \n"
        f"**Size on Disk:** {meta['Size (MB)']} MB  \n"
        f"**Standard Benchmark Latency:** {meta['CPU latency (ms, mean +/- std)']} ms"
    )

    return probs_dict, lat_str, info_md


def update_model_info(model_name: str) -> str:
    """Update metadata panel when dropdown model changes."""
    meta = get_model_metadata(model_name)
    return (
        f"**Model:** {model_name}  \n"
        f"**Top-1 Accuracy:** {meta['Top-1 (%)']}  \n"
        f"**Parameters:** {meta['Params (M)']} M  \n"
        f"**Complexity:** {meta['MACs (M)']} MACs (M)  \n"
        f"**Size on Disk:** {meta['Size (MB)']} MB  \n"
        f"**Standard Benchmark Latency:** {meta['CPU latency (ms, mean +/- std)']} ms"
    )


# ---------------------------------------------------------------------------
# Tab 2: Compare all models logic
# ---------------------------------------------------------------------------

def compare_models_tab(image: Image.Image | None) -> pd.DataFrame:
    """Run one image through every model and format a comparison table."""
    if image is None:
        return pd.DataFrame([{"Message": "Please upload an image to compare."}])

    records = compare_all_models(image)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Tab 3 & 4 data loaders
# ---------------------------------------------------------------------------

def get_results_table() -> pd.DataFrame:
    """Load results/results.csv into a pandas DataFrame."""
    p = Path("results/results.csv")
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame([{"Notice": "results/results.csv not found. Run evaluate.py first."}])


def get_genotype_diagrams() -> tuple[Path | None, Path | None]:
    """Ensure normal and reduction cell diagrams are rendered."""
    genotype_path = Path("results/search_lam0/genotype.json")
    if not genotype_path.exists():
        return None, None

    g = load_genotype(genotype_path)
    out_dir = Path("results/figures/cells_lam0")
    plot_genotype(g, str(out_dir))

    n_png = out_dir / "cell_normal.png"
    r_png = out_dir / "cell_reduce.png"
    return (n_png if n_png.exists() else None, r_png if r_png.exists() else None)


# ---------------------------------------------------------------------------
# Gradio Application Definition
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    with gr.Blocks(title="ThriftNAS: Efficiency-Aware DARTS Demo") as demo:
        gr.Markdown(
            "# 🚀 ThriftNAS: Efficiency-Aware Neural Architecture Search\n"
            "An interactive demonstration of hardware-free multi-objective DARTS on CIFAR-10. "
            "All inference runs on local CPU under `torch.inference_mode()` with single-thread timing."
        )

        with gr.Tabs():
            # ===============================================================
            # TAB 1: Classify
            # ===============================================================
            with gr.TabItem("🖼️ Classify"):
                gr.Markdown(
                    "### Single Model Image Classification\n"
                    "Upload any image or pick an example below. "
                    "**Notice:** Images are automatically converted to RGB and resized to 32×32 pixels before classification."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        input_img = gr.Image(
                            type="pil",
                            label="Input Image",
                            sources=["upload", "clipboard", "webcam"],
                        )
                        model_dropdown = gr.Dropdown(
                            choices=get_available_model_names(),
                            value=get_available_model_names()[0],
                            label="Select Architecture",
                        )
                        classify_btn = gr.Button("⚡ Classify Image", variant="primary")

                        if EXAMPLES:
                            gr.Examples(
                                examples=EXAMPLES,
                                inputs=[input_img],
                                label="CIFAR-10 Test Examples",
                            )

                    with gr.Column(scale=1):
                        preds_output = gr.Label(num_top_classes=3, label="Top-3 Predictions")
                        latency_output = gr.Textbox(label="Measured CPU Inference Latency (Median of 20 runs)")
                        metadata_panel = gr.Markdown(
                            value=update_model_info(get_available_model_names()[0]),
                            label="Model Specification",
                        )

                # Bind events
                model_dropdown.change(
                    fn=update_model_info,
                    inputs=[model_dropdown],
                    outputs=[metadata_panel],
                )
                classify_btn.click(
                    fn=classify_image,
                    inputs=[input_img, model_dropdown],
                    outputs=[preds_output, latency_output, metadata_panel],
                )

            # ===============================================================
            # TAB 2: Compare All Models
            # ===============================================================
            with gr.TabItem("⚖️ Compare All Models"):
                gr.Markdown(
                    "### Side-by-Side Model Comparison\n"
                    "Evaluate the same input image across all baseline and discovered ThriftNAS architectures simultaneously.\n\n"
                    "Upload an image, capture from camera, paste from clipboard, or select a sample image below. "
                    "*Images are automatically resized to 32×32 pixels.*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_img = gr.Image(
                            type="pil",
                            label="Comparison Image",
                            sources=["upload", "clipboard", "webcam"],
                        )
                        compare_btn = gr.Button("🔍 Compare All Models", variant="primary")
                        if EXAMPLES:
                            gr.Examples(
                                examples=EXAMPLES,
                                inputs=[compare_img],
                                label="CIFAR-10 Test Examples",
                            )
                    with gr.Column(scale=2):
                        compare_table = gr.Dataframe(
                            label="Prediction & Latency Summary",
                            interactive=False,
                        )

                compare_btn.click(
                    fn=compare_models_tab,
                    inputs=[compare_img],
                    outputs=[compare_table],
                )

            # ===============================================================
            # TAB 3: Results
            # ===============================================================
            with gr.TabItem("📊 Results & Pareto Frontiers"):
                gr.Markdown("### Published Metrics & Pareto Frontier Analysis")
                results_df = gr.Dataframe(value=get_results_table(), interactive=False)

                gr.Markdown("#### Pareto Trade-Off Curves")
                with gr.Row():
                    p_macs_path = "results/figures/pareto_acc_vs_macs.png"
                    p_lat_path = "results/figures/pareto_acc_vs_latency.png"
                    if os.path.exists(p_macs_path):
                        gr.Image(value=p_macs_path, label="Accuracy vs. MACs")
                    if os.path.exists(p_lat_path):
                        gr.Image(value=p_lat_path, label="Accuracy vs. Latency")

                lam_path = "results/figures/lambda_ablation.png"
                if os.path.exists(lam_path):
                    gr.Image(value=lam_path, label="Lambda Ablation Trade-off")

            # ===============================================================
            # TAB 4: Search Replay
            # ===============================================================
            with gr.TabItem("🔄 Search Replay"):
                gr.Markdown(
                    "### Offline Search Dynamics Replay\n"
                    "Inspection of architecture parameters α and cost trajectory from logged search runs. *(No live search performed)*"
                )
                with gr.Row():
                    e_alpha_path = "results/figures/e_alpha_over_epochs.png"
                    skip_path = "results/figures/skip_count_vs_lambda.png"
                    if os.path.exists(e_alpha_path):
                        gr.Image(value=e_alpha_path, label="Expected Cost E(α) Trajectory")
                    if os.path.exists(skip_path):
                        gr.Image(value=skip_path, label="Skip Connection Count vs. Penalty λ")

                with gr.Row():
                    alpha_lam0 = "results/figures/alpha_evolution_lam0.png"
                    if os.path.exists(alpha_lam0):
                        gr.Image(value=alpha_lam0, label="Architecture Parameter α Evolution (λ = 0.0)")

                gr.Markdown("#### Discovered Cell Architectures (Graphviz)")
                cell_normal, cell_reduce = get_genotype_diagrams()
                with gr.Row():
                    if cell_normal:
                        gr.Image(value=str(cell_normal), label="Derived Normal Cell")
                    if cell_reduce:
                        gr.Image(value=str(cell_reduce), label="Derived Reduction Cell")

    return demo


def main() -> None:
    app = create_app()
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
