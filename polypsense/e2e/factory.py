def get_model(args):
    if args.encoder_type and args.encoder_type == "sfe-v2":
        return get_sfe_v2(args)
    elif args.encoder_type and args.encoder_type == "mve-v2":
        return get_mve_v2(args)
    elif args.encoder_type and args.encoder_type == "hmve":
        return get_hmve(args)
    else:
        raise ValueError(f"Unknown encoder type: {args.encoder_type}")


def get_sfe_v2(args):
    from polypsense.e2e.model import SFEv2Module

    if args.ckpt_path:
        return SFEv2Module.load_from_checkpoint(args.ckpt_path, map_location="cpu")

    return SFEv2Module(
        backbone_arch=args.sfe_backbone_arch,
        backbone_weights=args.sfe_backbone_weights,
        d_model=args.sfe_d_model,
        d_proj=args.sfe_d_proj,
        lr=args.lr,
        loss_type=args.loss_type,
        loss_temperature=args.loss_temperature,
        loss_lambda=args.loss_lambda,
        loss_gamma=args.loss_gamma,
    )


def get_mve_v2(args):
    from polypsense.e2e.model import MVEv2Module

    if args.ckpt_path:
        return MVEv2Module.load_from_checkpoint(args.ckpt_path, map_location="cpu")

    return MVEv2Module(
        sfe_backbone_arch=args.sfe_backbone_arch,
        sfe_backbone_weights=args.sfe_backbone_weights,
        sfe_ckpt=args.sfe_ckpt,
        sfe_freeze=args.sfe_freeze,
        d_in=args.sfe_d_model,
        d_proj=args.mve_d_proj,
        d_model=args.mve_d_model,
        d_feedforward=args.mve_d_feedforward,
        n_heads=args.mve_n_heads,
        n_layers=args.mve_n_layers,
        dropout=args.mve_dropout,
        pooling_strategy=args.mve_pooling,
        lr=args.lr,
        loss_type=args.loss_type,
        loss_temperature=args.loss_temperature,
        loss_lambda=args.loss_lambda,
        loss_gamma=args.loss_gamma,
        loss_epsilon=args.loss_epsilon,
        loss_supervision=args.loss_supervision,
        max_seq_len=args.fragment_length,
        use_positional_encoding=args.mve_use_positional_encoding,
        batch_size=args.batch_size,
        n_views=args.n_views,
    )


def get_hmve(args):
    from polypsense.e2e.modeling.hmve import HierarchicalMVEEncoder

    if args.ckpt_path:
        return HierarchicalMVEEncoder.load_from_checkpoint(
            args.ckpt_path, map_location="cpu"
        )

    return HierarchicalMVEEncoder(
        backbone_arch=args.sfe_backbone_arch,
        backbone_weights=args.sfe_backbone_weights,
        d_in=args.sfe_d_model,
        d_proj=args.d_proj,
        d_model=args.hmve_d_model,
        d_feedforward=args.hmve_d_feedforward,
        n_heads=args.hmve_n_heads,
        n_layers=args.hmve_n_layers,
        dropout=args.hmve_dropout,
        use_feature_norm=args.hmve_use_feature_norm,
        use_encoder_norm=args.hmve_use_encoder_norm,
        mask_intra_fragment_frames=args.hmve_mask_intra_fragment_frames,
        loss_lr=args.lr,
        loss_temperature=args.loss_temperature,
        loss_weights=args.loss_milnce_weights,
        n_views=args.n_views,
    )
