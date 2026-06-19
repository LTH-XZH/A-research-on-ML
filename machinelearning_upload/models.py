import timm

# ---------------- CNN ----------------
def get_cnn():
    return timm.create_model(
        "convnext_tiny",
        pretrained=False,
        num_classes=10
    )

# ---------------- ViT ----------------
def get_vit():
    return timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=False,
        num_classes=10
    )

# ---------------- Swin Transformer v2 ----------------
def get_swin():
    return timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=10,
        img_size=32
    )