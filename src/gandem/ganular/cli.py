"""Command-line interface for GANular training."""

import argparse


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the GANular 3D particle generator from 2D images."
    )
    parser.add_argument("data_dir", help="Directory containing training images")
    parser.add_argument("-o", "--output-dir", help="New directory for training outputs")
    parser.add_argument(
        "--epochs", type=_positive_int, help="Number of training epochs"
    )
    parser.add_argument(
        "--object-batch-size",
        "--batch-size",
        dest="object_batch_size",
        type=_positive_int,
        help="Generated objects per update",
    )
    parser.add_argument(
        "--views-per-object",
        "--views",
        dest="views_per_object",
        type=_positive_int,
        help="Projection views per generated object",
    )
    parser.add_argument(
        "--updates-per-epoch", type=_positive_int, help="Generator updates per epoch"
    )
    parser.add_argument(
        "--projection-sample-size",
        type=_positive_int,
        help="Fixed number of 2D projections used by each quality evaluation",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--resume-from",
        help="Full training_epoch_*.pth checkpoint to continue from",
    )
    parser.add_argument(
        "--no-quality-checks",
        action="store_true",
        help="Disable advisory training and sample quality checks",
    )
    args = parser.parse_args(argv)

    from .quality import QualityChecks
    from .train import train

    quality_checks = None
    if args.no_quality_checks or args.projection_sample_size is not None:
        quality_checks = QualityChecks(
            enabled=not args.no_quality_checks,
            **(
                {"projection_sample_size": args.projection_sample_size}
                if args.projection_sample_size is not None
                else {}
            ),
        )

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        object_batch_size=args.object_batch_size,
        views_per_object=args.views_per_object,
        updates_per_epoch=args.updates_per_epoch,
        seed=args.seed,
        resume_from=args.resume_from,
        quality_checks=quality_checks,
    )


if __name__ == "__main__":
    main()
