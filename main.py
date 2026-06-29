"""Main process for DeepANC training and inference.

Use this file when you want to see the full project workflow:

    python main.py --mode train --data-path /path/to/dataset
    python main.py --mode infer --checkpoint checkpoints/deepanc_best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import DeepANC, init_weights
from src.utils import (
    InferConfig,
    STFT,
    TrainConfig,
    WavDataset,
    apply_anc_paths,
    complex_spectrogram_input,
    count_parameters,
    evaluate,
    load_checkpoint,
    load_path_filters,
    make_dataloaders,
    plot_results,
    power_to_db,
    resolve_device,
    save_checkpoint,
    set_seed,
    train_epoch,
    write_audio,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or run inference for DeepANC active noise control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("train", "infer"),
        required=True,
        help="Process to run.",
    )
    parser.add_argument("--device", default="auto", help="Use auto, cpu, cuda, or cuda:<index>.")

    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--data-path",
        "--data_path",
        type=Path,
        help="Dataset root with Hard_Index_train/val/test CSV files.",
    )
    train_group.add_argument("--epochs", type=int, default=100)
    train_group.add_argument("--batch-size", "--batch", dest="batch_size", type=int, default=32)
    train_group.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=1e-3)
    train_group.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=5e-5)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--resume", type=Path, help="Checkpoint to resume training from.")
    train_group.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        dest="checkpoint_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for latest/best training checkpoints.",
    )

    shared_group = parser.add_argument_group("data and signal processing")
    shared_group.add_argument("--sample-rate", "--sample_rate", "--fs", dest="sample_rate", type=int, default=16000)
    shared_group.add_argument(
        "--filter-dir",
        "--filter_dir",
        dest="filter_dir",
        type=Path,
        default=Path("src/mat_files"),
        help="Directory containing Primary_path.mat and Secondary_path.mat.",
    )
    shared_group.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=0)

    infer_group = parser.add_argument_group("inference")
    infer_group.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/deepanc_best.pt"),
        help="Checkpoint used for inference.",
    )
    infer_group.add_argument(
        "--input-dir",
        "--input_dir",
        dest="input_dir",
        type=Path,
        default=Path("results/noise_samples"),
        help="Directory of audio files for inference.",
    )
    infer_group.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("wav_results"),
        help="Directory for synthesized audio and metrics.",
    )
    return parser


def make_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        filter_dir=args.filter_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        device=args.device,
        seed=args.seed,
    )


def make_infer_config(args: argparse.Namespace) -> InferConfig:
    return InferConfig(
        checkpoint=args.checkpoint,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        filter_dir=args.filter_dir,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )


def train_process(config: TrainConfig) -> None:
    """Train DeepANC and keep the experiment flow visible in this file."""

    set_seed(config.seed)

    device = resolve_device(config.device)
    model = DeepANC().to(device)
    model.apply(init_weights)
    custom_stft = STFT(requires_grad=False, device=device).to(device)

    primary_filter, secondary_filter = load_path_filters(config.filter_dir)
    primary_filter = primary_filter.to(device)
    secondary_filter = secondary_filter.to(device)

    train_loader, valid_loader, test_loader = make_dataloaders(config)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(custom_stft.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        amsgrad=True,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, config.epochs // 4),
        gamma=0.5,
    )

    start_epoch = 0
    if config.resume is not None:
        checkpoint = load_checkpoint(
            config.resume,
            model,
            optimizer=optimizer,
            stft=custom_stft,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = config.checkpoint_dir / "deepanc_latest.pt"
    best_checkpoint = config.checkpoint_dir / "deepanc_best.pt"

    best_valid_loss = float("inf")
    train_losses: list[float] = []
    valid_losses: list[float] = []
    test_losses: list[float] = []

    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Loss: {loss_fn.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")

    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(
            model,
            custom_stft,
            train_loader,
            optimizer,
            loss_fn,
            primary_filter,
            secondary_filter,
            device,
        )
        scheduler.step()

        valid_loss = evaluate(
            model,
            custom_stft,
            valid_loader,
            loss_fn,
            primary_filter,
            secondary_filter,
            device,
            desc="valid",
        )
        test_loss = evaluate(
            model,
            custom_stft,
            test_loader,
            loss_fn,
            primary_filter,
            secondary_filter,
            device,
            desc="test",
        )

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        test_losses.append(test_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(best_checkpoint, epoch, model, optimizer, custom_stft, asdict(config))

        save_checkpoint(latest_checkpoint, epoch, model, optimizer, custom_stft, asdict(config))
        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"train={train_loss:.4f} valid={valid_loss:.4f} "
            f"test={test_loss:.4f} best={best_valid_loss:.4f}"
        )

    plot_results(train_losses, valid_losses, test_losses, config.checkpoint_dir / "loss.png")


@torch.no_grad()
def infer_process(config: InferConfig) -> pd.DataFrame:
    """Run DeepANC inference and keep the audio pipeline visible in this file."""

    device = resolve_device(config.device)
    model = DeepANC().to(device)
    custom_stft = STFT(device=device).to(device)
    load_checkpoint(config.checkpoint, model, stft=custom_stft, map_location=device)
    model.eval()

    primary_filter, secondary_filter = load_path_filters(config.filter_dir)
    primary_filter = primary_filter.to(device)
    secondary_filter = secondary_filter.to(device)

    dataset = WavDataset(config.input_dir, sr=config.sample_rate)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    rows: list[dict[str, float | str]] = []
    synthesized_dir = config.output_dir / "synthesized"
    anti_noise_dir = config.output_dir / "anti_noise"

    print(f"Device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Checkpoint: {config.checkpoint}")
    print(f"Input audio: {config.input_dir}")

    for origin_audio, file_names in tqdm(dataloader, desc="infer", leave=False):
        origin_audio = origin_audio.to(device)
        real, imag = custom_stft.stft(origin_audio.squeeze(1))
        model_input = complex_spectrogram_input(real, imag, pad_frames=2)

        model_output = model(model_input)
        anti_noise = custom_stft.istft(model_output)
        desired, _, error = apply_anc_paths(
            primary_filter,
            secondary_filter,
            origin_audio,
            anti_noise,
        )

        for item_idx, file_name in enumerate(file_names):
            stem = Path(file_name).stem
            item_error = error[item_idx].detach().cpu()
            item_desired = desired[item_idx].detach().cpu()
            item_origin = origin_audio[item_idx].squeeze(0).detach().cpu()

            write_audio(synthesized_dir / f"{stem}_synthesized.wav", item_error, config.sample_rate)
            write_audio(anti_noise_dir / f"{stem}_anti_noise.wav", item_desired, config.sample_rate)

            error_db = power_to_db(item_error.numpy())
            desired_db = power_to_db(item_desired.numpy())
            origin_db = power_to_db(item_origin.numpy())
            rows.append(
                {
                    "file_name": file_name,
                    "origin_db_max": float(origin_db.max()),
                    "origin_db_avg": float(origin_db.mean()),
                    "error_db_max": float(error_db.max()),
                    "error_db_avg": float(error_db.mean()),
                    "anti_noise_db_max": float(desired_db.max()),
                    "anti_noise_db_avg": float(desired_db.mean()),
                }
            )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(config.output_dir / "db_values.csv", index=False)
    return results


def run(args: argparse.Namespace) -> None:
    if args.mode == "train":
        train_process(make_train_config(args))
    elif args.mode == "infer":
        infer_process(make_infer_config(args))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "train" and args.data_path is None:
        parser.error("--data-path is required when --mode train")
    run(args)


if __name__ == "__main__":
    main()
