"""fitcheck CLI -- know before you train.

Typer-based command-line interface. Entry point: `fitcheck plan`.
Business logic lives in fitcheck.api; this module handles flag
parsing, error display, and report output.

Supports two invocation styles:
    fitcheck plan --model meta-llama/Llama-3.1-8B --method qlora --gpu 3090
    fitcheck plan "qlora meta-llama/Llama-3.1-8B on 3090"
"""

from __future__ import annotations

from typing import Optional

import typer

from fitcheck.report.formatter import print_report

app = typer.Typer(
    name="fitcheck",
    help="Know before you train -- VRAM estimation for LLM fine-tuning.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """fitcheck -- know before you train."""


@app.command()
def plan(
    spec: Optional[str] = typer.Argument(
        None,
        help='Natural language spec, e.g. "qlora meta-llama/Llama-3.1-8B on 3090"',
    ),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="HuggingFace model ID"),
    method: Optional[str] = typer.Option(None, "--method", help="Training method: full, lora, qlora"),
    gpu: Optional[str] = typer.Option(None, "--gpu", "-g", help="GPU name or alias (e.g. 3090, h100)"),
    seq_len: Optional[int] = typer.Option(None, "--seq-len", help="Sequence length for estimation"),
    lora_rank: int = typer.Option(16, "--lora-rank", help="LoRA rank"),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", help="Override solver batch search with a fixed value"
    ),
    eval_seq_len: Optional[int] = typer.Option(
        None, "--eval-seq-len", help="Max eval sequence length for KV-cache spike"
    ),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Path to local dataset file (.jsonl/.json)"
    ),
) -> None:
    """Estimate VRAM usage and find optimal training config."""
    from fitcheck.api import plan as api_plan
    from fitcheck.nlparse import parse_spec

    # Resolve inputs: NL spec provides defaults, flags override
    resolved_model = model
    resolved_method = method
    resolved_gpu = gpu
    resolved_dataset = dataset
    resolved_seq_len = seq_len

    if spec is not None:
        parsed = parse_spec(spec)
        if parsed is None:
            typer.echo(
                f'Error: Could not parse "{spec}". '
                'Expected format: "qlora meta-llama/Llama-3.1-8B on 3090"',
                err=True,
            )
            raise typer.Exit(code=1)
        # NL spec provides defaults; explicit flags override
        resolved_model = resolved_model or parsed.model_id
        resolved_method = resolved_method or parsed.method
        resolved_gpu = resolved_gpu or parsed.gpu
        resolved_dataset = resolved_dataset or parsed.dataset_path
        if resolved_seq_len is None:
            resolved_seq_len = parsed.seq_len

    # Validate required fields
    if not resolved_model or not resolved_method or not resolved_gpu:
        missing = []
        if not resolved_model:
            missing.append("--model")
        if not resolved_method:
            missing.append("--method")
        if not resolved_gpu:
            missing.append("--gpu")
        typer.echo(
            f"Error: Missing required options: {', '.join(missing)}. "
            'Provide flags or use: fitcheck plan "qlora model-id on gpu"',
            err=True,
        )
        raise typer.Exit(code=1)

    # seq_len may be None here -- api.plan() resolves: explicit > dataset p95 > default 512
    try:
        report = api_plan(
            model_id=resolved_model,
            method=resolved_method,
            gpu=resolved_gpu,
            seq_len=resolved_seq_len,
            lora_rank=lora_rank,
            batch_size=batch_size,
            eval_seq_len=eval_seq_len,
            dataset_path=resolved_dataset,
        )
    except (ValueError, KeyError, FileNotFoundError) as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)

    print_report(report)


if __name__ == "__main__":
    app()
