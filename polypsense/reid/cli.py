import argparse
import json

from polypsense.reid.engine import evaluate


def main():
    parser = get_parser()
    args = parser.parse_args()
    evaluate(args)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--wandb_project", type=str)

    parser.add_argument(
        "--task",
        type=str,
        choices=["reid", "retrieval"],
    )
    parser.add_argument("--stages", type=str, nargs="+", default=None)

    parser.add_argument("--dataset_root", type=str)
    # e.g. "data/e2e/splits/oneout_001-009"

    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=[
            "sfe-v2",
            "mve-v2",
            "timesformer",
            "surgenet",
            "videomae",
            "dinov2",
            "vjepa2",
            "hmve",
            "shmve",
            "endofm",
            "endofmlv",
            "endovit",
            "hendofm",
            "dinov3",
        ],
    )
    parser.add_argument("--encoder_ckpt", type=str)
    parser.add_argument(
        "--encoder_features_source", type=str, choices=["encoder", "projector"]
    )
    parser.add_argument(
        "--encoder_pooling",
        type=str,
        choices=[
            "none",
            "cls",
            "mean",
            "patchmean",
            "cat",
            "mean-mean",  # e.g. dinov2
            "cls-mean",  # e.g. dinov3
        ],
    )
    parser.add_argument("--encoder_out_dim", type=int)

    parser.add_argument("--counting_algorithm", type=str)
    parser.add_argument("--counting_hparams", type=json.loads, default=None)

    parser.add_argument("--tracking_workdir", type=str)

    parser.add_argument("--data_fragment_length", type=int)
    parser.add_argument("--data_aug_resize", action="store_true")
    parser.add_argument("--data_aug_anchorcrop", action="store_true")
    parser.add_argument("--data_bbox_scale_factor", type=int)
    parser.add_argument("--data_im_size", type=int)
    parser.add_argument("--data_aug_normalize", action="store_true")

    # surgenet specific
    parser.add_argument("--surgenet_arch", type=str)

    # timesformer specific
    parser.add_argument(
        "--timesformer_pooling", type=str, choices=["cls", "mean", "patchmean", "cat"]
    )

    return parser


if __name__ == "__main__":
    main()
