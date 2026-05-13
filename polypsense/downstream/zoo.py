from polypsense.e2e.model import MVEv2Module as MVEv2Backbone
from polypsense.e2e.modeling.endofm import get_endofm
from polypsense.e2e.modeling.hmve import (
    HierarchicalMVEEncoder as HierarchicalMVEBackbone,
)
from polypsense.e2e.modeling.surgenet import SurgenetBackbone
from polypsense.e2e.modeling.endofmlv import get_endofmlv
from polypsense.e2e.modeling.endovit import get_endovit
from polypsense.e2e.modeling.dinov2 import DinoV2 as DinoV2Backbone
from polypsense.e2e.modeling.dinov3 import DinoV3 as DinoV3Backbone
from polypsense.e2e.modeling.vjepa2 import VJEPA2 as VJEPA2Backbone


def get_encoder(args):
    if args.encoder_type == "surgenet":
        return get_surgenet(encoder_ckpt=args.encoder_ckpt)
    elif args.encoder_type == "mve-v2":
        return get_mve_v2(encoder_ckpt=args.encoder_ckpt)
    elif args.encoder_type == "hmve":
        return get_hmve(encoder_ckpt=args.encoder_ckpt)
    elif args.encoder_type == "endofm":
        return get_endofm(args.encoder_ckpt)
    elif args.encoder_type == "endofmlv":
        return get_endofmlv(args.encoder_ckpt)
    elif args.encoder_type == "endovit":
        return get_endovit(args.encoder_ckpt)
    elif args.encoder_type == "dinov2":
        return get_dinov2(args.encoder_ckpt)
    elif args.encoder_type == "dinov3":
        return get_dinov3(args.encoder_ckpt)
    elif args.encoder_type == "vjepa2":
        return get_vjepa2(args.encoder_ckpt)
    else:
        raise NotImplementedError(f"Encoder type {args.encoder_type} not implemented.")


def get_surgenet(encoder_ckpt: str):
    return SurgenetBackbone(
        backbone_arch="caformer_s18",
        pretrained_weights=encoder_ckpt,
    )


def get_mve_v2(encoder_ckpt: str):
    return MVEv2Backbone.load_from_checkpoint(encoder_ckpt, map_location="cpu")


def get_hmve(encoder_ckpt: str):
    return HierarchicalMVEBackbone.load_from_checkpoint(
        encoder_ckpt, map_location="cpu"
    )


def get_dinov2(encoder_ckpt: str):
    if encoder_ckpt.endswith(".ckpt"):
        raise NotImplementedError(
            "Loading DinoV2 from .ckpt not implemented. "
            "Please use pretrained weights instead."
        )

    return DinoV2Backbone(encoder_ckpt)


def get_dinov3(encoder_ckpt: str):
    if encoder_ckpt.endswith(".ckpt"):
        raise NotImplementedError(
            "Loading DinoV3 from .ckpt not implemented. "
            "Please use pretrained weights instead."
        )

    return DinoV3Backbone(encoder_ckpt)


def get_vjepa2(encoder_ckpt: str):
    if encoder_ckpt.endswith(".ckpt"):
        raise NotImplementedError(
            "Loading VJEPA2 from .ckpt not implemented. "
            "Please use pretrained weights instead."
        )

    return VJEPA2Backbone(encoder_ckpt)
