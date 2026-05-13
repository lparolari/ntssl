import argparse

from polypsense.e2e.engine import eval, train


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.mode and args.mode == "eval":
        eval(args)
    else:
        train(args)


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_root", type=str)

    # training
    parser.add_argument("--mode", type=str, choices=["train", "eval"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--wandb_project", type=str)

    # datamodule
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fragment_length", type=int)
    parser.add_argument("--fragment_stride", type=int)
    parser.add_argument("--fragment_drop_last", action="store_true")
    parser.add_argument("--fragment_padding_mode", type=str)
    parser.add_argument("--bbox_scale_factor", type=float)
    parser.add_argument("--bbox_scale_range", type=float, nargs=2)
    parser.add_argument("--min_tracklet_length", type=int)
    parser.add_argument("--im_size", type=int)
    parser.add_argument("--n_views", type=int)
    parser.add_argument("--eval_n_views", type=int)
    parser.add_argument("--eval_batch_size", type=int)
    parser.add_argument("--eval_fragment_length", type=int)
    parser.add_argument("--train_subset_ratio", type=float)
    parser.add_argument("--do_upsample", action="store_true")
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["multipos", "temporalknnbag", "temporaltemperaturebag"],
    )
    parser.add_argument("--sampler_ttb_tmin", type=float)
    parser.add_argument("--sampler_ttb_tmax", type=float)

    # augmentation
    parser.add_argument("--aug_resize", action="store_true")
    parser.add_argument("--aug_anchorcrop", action="store_true")
    parser.add_argument("--aug_normalize", action="store_true")
    parser.add_argument("--eval_aug_resize", action="store_true")
    parser.add_argument("--eval_aug_anchorcrop", action="store_true")

    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=[
            "sfe-v2",
            "mve-v2",
            "hmve",
            "shmve",
        ],
    )

    # projector
    parser.add_argument("--d_proj", type=int)

    # sfe
    parser.add_argument("--sfe_ckpt", type=str)
    parser.add_argument("--sfe_freeze", action="store_true", default=None)
    parser.add_argument("--sfe_backbone_arch", type=str, choices=["resnet18", "resnet50"])  # fmt: skip
    parser.add_argument("--sfe_backbone_weights", choices=["IMAGENET1K_V1", "IMAGENET1K_V2"])  # fmt: skip
    parser.add_argument("--sfe_d_model", type=int)
    parser.add_argument("--sfe_d_proj", type=int)

    # mve
    parser.add_argument("--mve_d_proj", type=int)
    parser.add_argument("--mve_d_model", type=int)
    parser.add_argument("--mve_d_feedforward", type=int)
    parser.add_argument("--mve_n_heads", type=int)
    parser.add_argument("--mve_n_layers", type=int)
    parser.add_argument("--mve_dropout", type=float)
    parser.add_argument("--mve_pooling", type=str, choices=["mean", "max", "cls"])
    parser.add_argument("--mve_use_positional_encoding", action="store_true")

    # hmve
    parser.add_argument("--hmve_d_proj", type=int)
    parser.add_argument("--hmve_d_model", type=int)
    parser.add_argument("--hmve_d_feedforward", type=int)
    parser.add_argument("--hmve_n_heads", type=int)
    parser.add_argument("--hmve_n_layers", type=int)
    parser.add_argument("--hmve_dropout", type=float)
    parser.add_argument("--hmve_use_feature_norm", action="store_true")
    parser.add_argument("--hmve_use_encoder_norm", action="store_true")
    parser.add_argument("--hmve_mask_intra_fragment_frames", action="store_true")

    # loss
    parser.add_argument(
        "--loss_type",
        type=str,
        choices=["temporally-aware", "supervised", "mil-nce"],
    )
    parser.add_argument("--loss_temperature", type=float)
    parser.add_argument("--loss_lambda", type=float)
    parser.add_argument("--loss_milnce_weights", type=float, nargs="+")

    # experiment
    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--exp_name", type=str)

    return parser


if __name__ == "__main__":
    main()
