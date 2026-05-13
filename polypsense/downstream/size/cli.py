import argparse

from polypsense.downstream.size.engine import train


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_id", type=str)
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)

    # data paths
    parser.add_argument("--dataset_type", type=str, choices=["coco"])
    parser.add_argument("--train_images", type=str)
    parser.add_argument("--train_annotations", type=str)
    parser.add_argument("--val_images", type=str)
    parser.add_argument("--val_annotations", type=str)

    # data config
    parser.add_argument("--data_fragment_length", type=int)
    parser.add_argument("--data_fragment_stride", type=int)
    parser.add_argument("--data_fragment_drop_last", action="store_true")
    parser.add_argument("--data_fragment_padding_mode", type=str)
    parser.add_argument("--data_aug_resize", action="store_true")
    parser.add_argument("--data_aug_anchorcrop", action="store_true")
    parser.add_argument("--data_aug_normalize", action="store_true")
    parser.add_argument("--data_im_size", type=int)
    parser.add_argument("--data_bbox_scale_factor", type=int)

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--opt_type", type=str, choices=["adamw", "sgd"])
    parser.add_argument("--opt_lr", type=float)
    parser.add_argument("--opt_weight_decay", type=float)
    parser.add_argument("--lr_sched_type", type=str, choices=["none", "onecycle"])
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)

    # task
    parser.add_argument("--n_classes", type=int)
    parser.add_argument("--pos_weight", type=float)
    parser.add_argument("--cls_type", type=str, choices=["linear", "mlp"])
    parser.add_argument("--cls_hidden_dim", type=int)
    parser.add_argument("--cls_init", type=str, choices=["default", "normal"])
    parser.add_argument("--cls_use_layer_norm", action="store_true")

    # stages
    parser.add_argument("--stages", type=str, nargs="+", default=None)

    # encoder
    parser.add_argument("--train_encoder", action="store_true")
    parser.add_argument(
        "--encoder_type",
        type=str,
        choices=[
            "surgenet",
            "mve-v2",
            "hmve",
            "shmve",
            "endofm",
            "endofmlv",
            "endovit",
            "dinov2",
            "vjepa2",
            "hendofm",
            "dinov3",
        ],
    )
    parser.add_argument("--encoder_ckpt", type=str)
    parser.add_argument("--encoder_out_dim", type=int)
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

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
