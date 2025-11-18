import torch


def get_device():
    """
    Returns:
        device_kwargs (dict)
        use_qlora (bool)
    QLoRA can only be used on CUDA with bitsandbytes installed.
    Fallback: normal LoRA on MPS or CPU.
    """
    # Check if bitsandbytes is installed
    try:
        import bitsandbytes

        bnb_available = True
    except ImportError:
        bnb_available = False

    # CUDA + bitsandbytes → use QLoRA
    if torch.cuda.is_available() and bnb_available:
        print("🚀 Using CUDA GPU with QLoRA")
        device_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "load_in_4bit": True,
        }
        use_qlora = True

    # Apple MPS → normal LoRA, FP16
    elif torch.backends.mps.is_available():
        print("🍏 Using Apple MPS (normal LoRA, FP16)")
        device_kwargs = {
            "device_map": {"": 0},
            "torch_dtype": torch.float16,
            "load_in_4bit": False,
        }
        use_qlora = False

    # CPU → normal LoRA, FP32
    else:
        print("🧠 Using CPU (normal LoRA, FP32)")
        device_kwargs = {
            "device_map": "cpu",
            "torch_dtype": torch.float32,
            "load_in_4bit": False,
        }
        use_qlora = False

    return device_kwargs, use_qlora
