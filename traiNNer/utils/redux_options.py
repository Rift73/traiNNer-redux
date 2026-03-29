from typing import Annotated, Any, Literal

from msgspec import Meta, Struct, field

from traiNNer.utils.types import PixelFormat


class StrictStruct(Struct, forbid_unknown_fields=True):
    pass


class WandbOptions(StrictStruct):
    resume_id: str | None = None
    project: str | None = None


class DatasetOptions(StrictStruct):
    name: Annotated[
        str,
        Meta(
            description="Name of the dataset. It should be unique compared to other datasets in this config, but the exact name isn't very important."
        ),
    ]
    type: str
    io_backend: dict[str, Any] = field(default_factory=lambda: {"type": "disk"})

    num_worker_per_gpu: Annotated[
        int | None,
        Meta(
            description="Number of subprocesses to use for data loading with PyTorch dataloader."
        ),
    ] = None
    batch_size_per_gpu: Annotated[
        int | None,
        Meta(
            description="Increasing stabilizes training but going too high can cause issues. Use multiple of 8 for best performance with AMP. A higher batch size, like 32 or 64 is more important when training from scratch, while smaller batches like 8 can be used when training with a quality pretrain model."
        ),
    ] = None
    accum_iter: Annotated[
        int,
        Meta(
            description="Using values larger than 1 simulates higher batch size by trading performance for reduced VRAM usage. If accum_iter = 4 and batch_size_per_gpu = 6 then effective batch size = 4 * 6 = 24 but performance may be as much as 4 times as slow."
        ),
    ] = 1

    use_hflip: Annotated[
        bool, Meta(description="Randomly flip the images horizontally.")
    ] = True
    use_rot: Annotated[bool, Meta(description="Randomly rotate the images.")] = True
    mean: list[float] | None = None
    std: list[float] | None = None
    gt_size: int | None = None
    lq_size: Annotated[
        int | None,
        Meta(
            description="During training, a square of this size is cropped from LR images. Larger is usually better but uses more VRAM. Previously gt_size, use lq_size = gt_size / scale to convert. Use multiple of 8 for best performance with AMP."
        ),
    ] = None
    color: Literal["y"] | None = None
    phase: str | None = None
    scale: int | None = None
    dataset_enlarge_ratio: Annotated[
        Literal["auto"] | int,
        Meta(
            description="Increase if the dataset is less than 1000 images to avoid slowdowns. Auto will automatically enlarge small datasets only."
        ),
    ] = "auto"
    prefetch_mode: str | None = None
    pin_memory: bool = True
    persistent_workers: bool = True
    num_prefetch_queue: int = 1
    prefetch_factor: int | None = 2

    clip_size: Annotated[
        int | None,
        Meta(
            description="Number of frames per clip in `PairedVideoDataset`. Must match the `clip_size` option for video generator networks such as `tscunet`."
        ),
    ] = None

    dataroot_gt: Annotated[
        str | list[str] | None,
        Meta(
            description="Path to the HR (high res) images in your training dataset. Specify one or multiple folders, separated by commas."
        ),
    ] = None
    target_dataroot_gt: Annotated[
        str | list[str] | None,
        Meta(
            description="Optional path to aligned target HR images used as supervision targets while `dataroot_gt` remains the source for OTF LR synthesis. Must match `dataroot_gt` one-to-one in relative paths."
        ),
    ] = None
    dataroot_lq: Annotated[
        str | list[str] | None,
        Meta(
            description="Path to the LR (low res) images in your training dataset. Specify one or multiple folders, separated by commas."
        ),
    ] = None
    paired_dataroot_gt: Annotated[
        str | list[str] | None,
        Meta(
            description="Path to paired HR images for RealESRGANPairedDataset. When set, the paired sub-dataset uses this instead of dataroot_gt, allowing separate folders for OTF and paired training."
        ),
    ] = None
    meta_info: str | None = None
    filename_tmpl: Annotated[
        str,
        Meta(
            description="Filename template to use for LR images. Commonly used values might be `{}x2` or `{}x4`, which should be used if the LR dataset filename is in the format filename.png while the LR dataset filename is in the format `filename_x2.png` or `filename_x4.png`. This is common on some research datasets such as DIV2K or DF2K."
        ),
    ] = "{}"

    blur_kernel_size: int = 12
    kernel_list: list[str] = field(
        default_factory=lambda: [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
    )
    kernel_prob: list[float] = field(
        default_factory=lambda: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    )
    kernel_range: tuple[int, int] = (5, 17)
    sinc_prob: float = 0
    blur_sigma: tuple[float, float] = (0.2, 2)
    betag_range: tuple[float, float] = (0.5, 4)
    betap_range: tuple[float, float] = (1, 2)

    blur_kernel_size2: int = 12
    kernel_list2: list[str] = field(
        default_factory=lambda: [
            "iso",
            "aniso",
            "generalized_iso",
            "generalized_aniso",
            "plateau_iso",
            "plateau_aniso",
        ]
    )
    kernel_prob2: list[float] = field(
        default_factory=lambda: [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    )
    kernel_range2: tuple[int, int] = (5, 17)
    sinc_prob2: float = 0
    blur_sigma2: tuple[float, float] = (0.2, 1)
    betag_range2: tuple[float, float] = (0.5, 4)
    betap_range2: tuple[float, float] = (1, 2)

    final_sinc_prob: float = 0
    final_kernel_range: tuple[int, int] = (5, 17)


class PathOptions(StrictStruct):
    experiments_root: str | None = None
    models: str | None = None
    resume_models: str | None = None
    training_states: str | None = None
    log: str | None = None
    visualization: str | None = None
    results_root: str | None = None

    pretrain_network_g: Annotated[
        str | None,
        Meta(
            description="Path to the pretrain model for the generator. `pth` and `safetensors` formats are supported."
        ),
    ] = None
    pretrain_network_g_teacher: Annotated[
        str | None,
        Meta(
            description="Path to the pretrain model for the teacher of the generator. `pth` and `safetensors` formats are supported."
        ),
    ] = None
    pretrain_network_g_path: str | None = None
    pretrain_network_ae_path: str | None = None
    param_key_g: str | None = None
    strict_load_g: Annotated[
        bool,
        Meta(
            description="Whether to load the pretrain model for the generator in strict mode. It should be enabled in most cases, unless you want to partially load a pretrain of a different scale or with slightly different hyperparameters."
        ),
    ] = True
    resume_state: str | None = None
    pretrain_network_g_ema: str | None = None

    pretrain_network_d: Annotated[
        str | None,
        Meta(
            description="Path to the pretrain model for the discriminator. `pth` and `safetensors` formats are supported."
        ),
    ] = None
    param_key_d: str | None = None
    strict_load_d: Annotated[
        bool,
        Meta(
            description="Whether to load the pretrain model for the discriminator in strict mode. It should be enabled in most cases."
        ),
    ] = True
    pretrain_network_ae: Annotated[
        str | None,
        Meta(
            description="Path to the pretrain model for the autoencoder. `pth` and `safetensors` formats are supported."
        ),
    ] = None
    pretrain_network_ae_ema: str | None = None
    pretrain_network_ae_decoder: Annotated[
        str | None,
        Meta(
            description="Path to the pretrain model for the decoder of the autoencoder. `pth` and `safetensors` formats are supported."
        ),
    ] = None
    pretrain_network_ae_decoder_ema: str | None = None
    ignore_resume_networks: list[str] | None = None


class OnnxOptions(StrictStruct):
    dynamo: bool = False
    opset: int = 20
    shape: str = "1x3xHxW"
    verify: bool = True
    dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    bf16_exclude_depthwise: bool = True
    optimize: bool = True
    wrap_5d_to_4d: bool = True


class TrainOptions(StrictStruct):
    total_iter: Annotated[
        int, Meta(description="The total number of iterations to train.")
    ]
    adaptive_d: Annotated[
        bool,
        Meta(
            description="Whether the discriminator updates adaptively. That is, discriminator updates are paused whenever the generator falls behind the discriminator (whenever smoothed l_g_gan increases). Can mitigate GAN collapse by preventing the discriminator from overpowering the generator."
        ),
    ] = False
    adaptive_d_ema_decay: float = 0.999
    adaptive_d_threshold: float = 1.02
    optim_g: Annotated[
        dict[str, Any] | None,
        Meta(description="The optimizer to use for the generator model."),
    ] = None
    ema_decay: Annotated[
        float,
        Meta(
            description="The decay factor to use for EMA (exponential moving average). Set to 0 to disable EMA."
        ),
    ] = 0
    ema_switch_iter: Annotated[
        float,
        Meta(
            description="How often to switch EMA model to online model, in iterations. Set to 0 to disable."
        ),
    ] = 0
    ema_update_after_step: int = 0  # no warmup
    ema_power: float = 10  # no warmup
    grad_clip: Annotated[
        bool,
        Meta(
            description="Whether or not to enable gradient clipping, which can improve stability when using higher learning rates, but can also cause issues if the grad clip max norm is too small."
        ),
    ] = False
    grad_clip_max_norm: Annotated[
        float, Meta(description="Max norm of gradients for gradient clipping.")
    ] = 100.0
    warmup_iter: Annotated[
        int,
        Meta(
            description="Gradually ramp up learning rates until this iteration, to stabilize early training. Use -1 to disable."
        ),
    ] = -1
    restart_scheduler: Annotated[
        bool,
        Meta(
            description="When resuming, if restart_scheduler is true then the scheduler is recreated using the schedule in the training config. If restart_scheduler is false then scheduler is read from saved state and training config schedule is ignored."
        ),
    ] = False
    scheduler: Annotated[
        dict[str, Any] | None,
        Meta(
            description="Options for the optimizer scheduler. If there are multiple optimizers, both will use the same scheduler options."
        ),
    ] = None
    optim_include_params_g: Annotated[
        list[str] | None,
        Meta(
            description="Optional regex patterns for generator parameter names to include in the optimizer. If set, only matching trainable parameters are optimized."
        ),
    ] = None
    optim_exclude_params_g: Annotated[
        list[str] | None,
        Meta(
            description="Optional regex patterns for generator parameter names to exclude from the optimizer after inclusion filtering."
        ),
    ] = None
    optim_d: Annotated[
        dict[str, Any] | None,
        Meta(description="The optimizer to use for the discriminator model."),
    ] = None
    optim_ae: Annotated[
        dict[str, Any] | None,
        Meta(description="The optimizer to use for the autoencoder model."),
    ] = None

    # new losses format
    losses: Annotated[
        list[dict[str, Any]] | None,
        Meta(description="The list of loss functions to optimize."),
    ] = None

    # old losses format
    pixel_opt: dict[str, Any] | None = None
    mssim_opt: dict[str, Any] | None = None
    ms_ssim_l1_opt: dict[str, Any] | None = None
    perceptual_opt: dict[str, Any] | None = None
    contextual_opt: dict[str, Any] | None = None
    dists_opt: dict[str, Any] | None = None
    hr_inversion_opt: dict[str, Any] | None = None
    dinov2_opt: dict[str, Any] | None = None
    topiq_opt: dict[str, Any] | None = None
    pd_opt: dict[str, Any] | None = None
    fd_opt: dict[str, Any] | None = None
    ldl_opt: dict[str, Any] | None = None
    hsluv_opt: dict[str, Any] | None = None
    gan_opt: dict[str, Any] | None = None
    color_opt: dict[str, Any] | None = None
    luma_opt: dict[str, Any] | None = None
    avg_opt: dict[str, Any] | None = None
    bicubic_opt: dict[str, Any] | None = None

    use_moa: Annotated[
        bool,
        Meta(
            description="Whether to enable mixture of augmentations, which augments the dataset on the fly to create more variety and help the model generalize."
        ),
    ] = False
    moa_augs: Annotated[
        list[str],
        Meta(
            description="The list of augmentations to choose from, only one is selected per iteration."
        ),
    ] = field(default_factory=lambda: ["none", "mixup", "cutmix", "resizemix"])
    moa_probs: Annotated[
        list[float],
        Meta(
            description="The probability each augmentation in moa_augs will be applied. Total should add up to 1."
        ),
    ] = field(default_factory=lambda: [0.4, 0.084, 0.084, 0.084, 0.348])
    moa_debug: Annotated[
        bool,
        Meta(
            description="Save images before and after augment to debug/moa folder inside of the root training directory."
        ),
    ] = False
    moa_debug_limit: Annotated[
        int,
        Meta(
            description="The max number of iterations to save augmentation images for."
        ),
    ] = 100


class ValOptions(StrictStruct):
    val_enabled: Annotated[
        bool,
        Meta(
            description="Whether to enable validations. If disabled, all validation settings below are ignored."
        ),
    ]
    save_img: Annotated[
        bool,
        Meta(
            description="Whether to save the validation images during validation, in the experiments/<name>/visualization folder."
        ),
    ]
    tile_size: Annotated[
        int,
        Meta(
            description="Tile size of input to use during validation, reduce VRAM usage but slower inference. 0 to disable."
        ),
    ] = 0
    tile_overlap: Annotated[
        int,
        Meta(
            description="Tile overlap to use during validation, larger is slower but reduces tile seams."
        ),
    ] = 0
    val_freq: Annotated[
        int | None, Meta(description="How often to run validations, in iterations.")
    ] = None
    suffix: Annotated[
        str | None, Meta(description="Optional suffix to append to saved filenames.")
    ] = None

    metrics_enabled: Annotated[
        bool, Meta(description="Whether to run metrics calculations during validation.")
    ] = False
    metrics: dict[str, Any] | None = None
    pbar: bool = True


class LogOptions(StrictStruct):
    print_freq: Annotated[
        int, Meta(description="How often to print logs to the console, in iterations.")
    ]
    save_checkpoint_freq: Annotated[
        int,
        Meta(
            description="How often to save model checkpoints and training states, in iterations."
        ),
    ]
    use_tb_logger: Annotated[
        bool, Meta(description="Whether or not to enable TensorBoard logging.")
    ]
    save_checkpoint_format: Annotated[
        Literal["safetensors", "pth"],
        Meta(description="Format to save model checkpoints."),
    ] = "safetensors"
    wandb: WandbOptions | None = None


class ReduxOptions(StrictStruct):
    name: Annotated[
        str,
        Meta(
            description="Name of the experiment. It should be a unique name if you want to run a new experiment. If you enable auto resume, the experiment with this name will be resumed instead of starting a new training run."
        ),
    ]
    scale: Annotated[
        int,
        Meta(
            description="Scale of the model. Most architectures support a scale of 1, 2, 3, 4, or 8. A scale of 1 can be used for restoration models that don't change the resolution of the input image. A scale of 2 means the width and height of the input image are doubled, so a 640x480 input will be upscaled to 1280x960."
        ),
    ]
    num_gpu: Annotated[
        Literal["auto"] | int,
        Meta(
            description="The number of GPUs to use for training, if using multiple GPUs."
        ),
    ]
    path: PathOptions

    input_pixel_format: Annotated[
        PixelFormat, Meta(description="Input pixel format.")
    ] = "rgb"
    output_pixel_format: Annotated[
        PixelFormat, Meta(description="Output pixel format.")
    ] = "rgb"

    network_g: Annotated[
        dict[str, Any] | None, Meta(description="The options for the generator model.")
    ] = None
    network_g_teacher: Annotated[
        dict[str, Any] | None,
        Meta(description="The options for the teacher generator model."),
    ] = None
    network_d: Annotated[
        dict[str, Any] | None,
        Meta(description="The options for the discriminator model."),
    ] = None
    network_ae: Annotated[
        dict[str, Any] | None,
        Meta(description="The options for the autoencoder model."),
    ] = None

    manual_seed: Annotated[
        int | None,
        Meta(
            description="Random seed for training, useful for removing randomness when testing the effect of different settings."
        ),
    ] = None
    deterministic: Annotated[
        bool,
        Meta(
            description="Enables torch.use_deterministic_algorithms. Slows down training, only use when reproducibility is critical."
        ),
    ] = False
    dist: bool | None = None
    launcher: str | None = None
    rank: int | None = None
    world_size: int | None = None
    auto_resume: bool | None = None
    watch: bool = False
    start_iter: int = 0
    is_train: bool | None = None
    root_path: str | None = None
    switch_iter_per_epoch: int = 1

    use_amp: Annotated[
        bool, Meta(description="Speed up training and reduce VRAM usage. NVIDIA only.")
    ] = False
    amp_bf16: Annotated[
        bool,
        Meta(
            description="Use bf16 instead of fp16 for AMP, RTX 3000 series or newer only. Only recommended if fp16 doesn't work."
        ),
    ] = False
    use_channels_last: Annotated[
        bool,
        Meta(
            description="Enable channels last memory format while using AMP. Reduces VRAM and speeds up training for most architectures, but some architectures are slower with channels last."
        ),
    ] = True
    fast_matmul: Annotated[
        bool, Meta(description="Trade precision for performance.")
    ] = False
    use_compile: Annotated[
        bool,
        Meta(
            description="Enable torch.compile for the generator model, which takes time on startup to compile the model, but can speed up training after the model is compiled. However, compilation must be redone when starting training each time, as the compiled model is not saved, so for models that take too long to compile it may not worth it."
        ),
    ] = False
    compile_mode: Annotated[
        Literal[
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ],
        Meta(
            description="Mode to use with torch.compile. See https://docs.pytorch.org/docs/stable/generated/torch.compile.html for more info."
        ),
    ] = "max-autotune-no-cudagraphs"
    detect_anomaly: Annotated[
        bool,
        Meta(
            description="Whether or not to enable anomaly detection, which can be useful for debugging NaNs that occur during training. Has a significant performance hit and should be disabled when not debugging."
        ),
    ] = False

    high_order_degradation: Annotated[
        bool,
        Meta(
            description="Whether or not to enable OTF (on the fly) degradations, which generates LRs on the fly."
        ),
    ] = False
    high_order_degradations_debug: Annotated[
        bool,
        Meta(
            description="Whether or not to enable debugging for OTF, which saves the OTF generated LR images so they can be inspected to view the effect of different OTF settings."
        ),
    ] = False
    high_order_degradations_debug_limit: Annotated[
        int,
        Meta(
            description="The maximum number of OTF images to save when debugging is enabled."
        ),
    ] = 100
    dataroot_lq_prob: Annotated[
        float,
        Meta(description="Probability of using paired LR data instead of OTF LR data."),
    ] = 0
    otf_shared_hf_noise_normalize: Annotated[
        bool,
        Meta(
            description="Whether to zero-center and normalize the shared high-frequency noise field before scaling it. Disable to use the raw `(beta - 0.5) * 2 * alpha` style field."
        ),
    ] = True
    otf_shared_hf_noise_prob: Annotated[
        float,
        Meta(
            description="Probability of adding the same shared high-frequency noise field to both GT and LQ after the final OTF crop, between 0 and 1."
        ),
    ] = 0
    otf_shared_hf_noise_alpha_range: Annotated[
        tuple[float, float],
        Meta(
            description="Amplitude range for the shared high-frequency noise field, in the format `[min_alpha, max_alpha]`."
        ),
    ] = (0.01, 0.05)
    otf_shared_hf_noise_beta_shape_range: Annotated[
        tuple[float, float],
        Meta(
            description="Range used to sample the beta `a` shape parameter, and also the `b` shape parameter when no `otf_shared_hf_noise_beta_offset_range` is set."
        ),
    ] = (2, 5)
    otf_shared_hf_noise_beta_offset_range: Annotated[
        tuple[float, float] | None,
        Meta(
            description="Optional positive offset range used to derive `b = a + offset` instead of sampling `b` independently. Use `[1, 5]` to approximate the external hf_noise snippet more closely."
        ),
    ] = None
    otf_shared_hf_noise_gray_prob: Annotated[
        float,
        Meta(
            description="Probability that the shared high-frequency noise field is grayscale and broadcast to all channels, between 0 and 1."
        ),
    ] = 1

    # ── Dithering (applied after crop, before dequeue — last degradation) ──────
    dithering_prob: Annotated[
        float,
        Meta(
            description="Probability of applying dithering to the LQ, between 0 and 1. Applied after crop as the last degradation step."
        ),
    ] = 0
    dithering_types: Annotated[
        list[str],
        Meta(
            description="List of dithering algorithms to choose from. Options: quantize, floydsteinberg, jarvisjudiceninke, stucki, atkinson, burkes, sierra, tworowsierra, sierralite, order, riemersma."
        ),
    ] = field(default_factory=lambda: ["quantize"])
    dithering_quantize_range: Annotated[
        tuple[int, int],
        Meta(
            description="Range of quantization levels per channel for dithering, in the format `[min_levels, max_levels]`."
        ),
    ] = (2, 10)
    dithering_map_size: Annotated[
        list[int],
        Meta(
            description="List of map sizes to choose from for ordered dithering."
        ),
    ] = field(default_factory=lambda: [4, 8])
    dithering_history_range: Annotated[
        tuple[int, int],
        Meta(
            description="Range of history values for Riemersma dithering."
        ),
    ] = (10, 15)
    dithering_ratio_range: Annotated[
        tuple[float, float],
        Meta(
            description="Range of decay ratio values for Riemersma dithering."
        ),
    ] = (0.1, 0.9)

    # ── Channel Shift (applied before resize1) ────────────────────────────────
    shift_prob: Annotated[
        float,
        Meta(
            description="Probability of applying channel shift to the LQ, between 0 and 1. Applied before the first resize."
        ),
    ] = 0
    shift_types: Annotated[
        list[str],
        Meta(
            description="List of color spaces to apply shift in. One is randomly selected. Options: rgb, yuv, cmyk."
        ),
    ] = field(default_factory=lambda: ["rgb"])
    shift_percent: Annotated[
        bool,
        Meta(
            description="If true, shift amounts are percentages of image dimensions. If false, shift amounts are in pixels."
        ),
    ] = False
    shift_rgb_r: Annotated[
        list[list[int]],
        Meta(description="Shift range for R channel: [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_rgb_g: Annotated[
        list[list[int]],
        Meta(description="Shift range for G channel: [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_rgb_b: Annotated[
        list[list[int]],
        Meta(description="Shift range for B channel: [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_yuv_y: Annotated[
        list[list[int]],
        Meta(description="Shift range for Y channel (YUV): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_yuv_u: Annotated[
        list[list[int]],
        Meta(description="Shift range for U channel (YUV): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_yuv_v: Annotated[
        list[list[int]],
        Meta(description="Shift range for V channel (YUV): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_cmyk_c: Annotated[
        list[list[int]],
        Meta(description="Shift range for C channel (CMYK): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_cmyk_m: Annotated[
        list[list[int]],
        Meta(description="Shift range for M channel (CMYK): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_cmyk_y: Annotated[
        list[list[int]],
        Meta(description="Shift range for Y channel (CMYK): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])
    shift_cmyk_k: Annotated[
        list[list[int]],
        Meta(description="Shift range for K channel (CMYK): [[x_min, x_max], [y_min, y_max]]."),
    ] = field(default_factory=lambda: [[0, 0], [0, 0]])

    # ── Chroma Subsampling (applied before resize1, after shift) ──────────────
    subsampling_prob: Annotated[
        float,
        Meta(
            description="Probability of applying chroma subsampling to the LQ, between 0 and 1. Applied before the first resize."
        ),
    ] = 0
    subsampling_down_algorithms: Annotated[
        list[str],
        Meta(
            description="List of downscaling algorithms to choose from for subsampling. Options: nearest, box, hermite, linear, lagrange, cubic_catrom, cubic_mitchell, cubic_bspline, lanczos, gauss."
        ),
    ] = field(default_factory=lambda: ["nearest"])
    subsampling_up_algorithms: Annotated[
        list[str],
        Meta(
            description="List of upscaling algorithms to choose from for subsampling."
        ),
    ] = field(default_factory=lambda: ["nearest"])
    subsampling_formats: Annotated[
        list[str],
        Meta(
            description="List of chroma subsampling formats to choose from. Options: 4:4:4, 4:2:2, 4:1:1, 4:2:0, 4:1:0, 4:4:0, 4:2:1, 4:1:2, 4:1:3."
        ),
    ] = field(default_factory=lambda: ["4:4:4"])
    subsampling_blur_range: Annotated[
        tuple[float, float] | None,
        Meta(
            description="Optional Gaussian blur sigma range for chroma channels after subsampling, in the format `[min_sigma, max_sigma]`. Null to disable."
        ),
    ] = None
    subsampling_ycbcr_type: Annotated[
        list[str],
        Meta(
            description="List of YCbCr standards to choose from. Options: 601, 709, 2020, 240."
        ),
    ] = field(default_factory=lambda: ["601"])

    lq_usm: Annotated[
        bool, Meta(description="Whether to enable unsharp mask on the LQ image.")
    ] = False
    lq_usm_radius_range: Annotated[
        tuple[int, int],
        Meta(
            description="For the unsharp mask of the LQ image, use a radius randomly selected from this range."
        ),
    ] = (1, 25)

    blur_prob: Annotated[
        float,
        Meta(
            description="Probability of applying the first blur to the LQ, between 0 and 1."
        ),
    ] = 0
    thicklines_prob: Annotated[
        float,
        Meta(
            description="Probability of applying custom ThickLines filter to the LQ, between 0 and 1."
        ),
    ] = 0
    resize_prob: Annotated[
        list[float],
        Meta(
            description="List of 3 probabilities for the first resize which should add up to 1: the probability of upscaling, the probability of downscaling, and the probability of no resize."
        ),
    ] = field(default_factory=lambda: [0.2, 0.7, 0.1])
    resize_mode_list: Annotated[
        list[Literal["bilinear", "bicubic", "nearest-exact", "lanczos", "area"]],
        Meta(description="List of possible resize modes to use for the first resize."),
    ] = field(
        default_factory=lambda: ["bilinear", "bicubic", "nearest-exact", "lanczos"]
    )
    resize_mode_prob: Annotated[
        list[float],
        Meta(
            description="List of probabilities for the first resize of selecting the corresponding resize mode in `resize_mode_list`."
        ),
    ] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    resize_range: Annotated[
        tuple[float, float],
        Meta(
            description="The resize range for the first resize, in the format `[min_resize, max_resize]`."
        ),
    ] = (0.4, 1.5)
    gaussian_noise_prob: Annotated[
        float,
        Meta(
            description="The probability of applying the first gaussian noise to the LQ, between 0 and 1."
        ),
    ] = 0
    noise_range: tuple[float, float] = (0, 0)
    poisson_scale_range: tuple[float, float] = (0, 0)
    gray_noise_prob: float = 0
    jpeg_prob: Annotated[
        float,
        Meta(
            description="The probability of applying compression (JPEG or other configured algorithm) to the LQ in stage 1, between 0 and 1."
        ),
    ] = 1
    jpeg_range: Annotated[
        tuple[float, float],
        Meta(
            description="The range of JPEG quality to apply for the first JPEG degradation, in the format `[min_quality, max_quality]`."
        ),
    ] = (75, 95)
    compress_algorithms: Annotated[
        list[str],
        Meta(
            description="List of compression algorithms to choose from in stage 1. One is randomly selected per iteration. Options: jpeg, webp, h264, hevc, mpeg2, mpeg4, vp9."
        ),
    ] = field(default_factory=lambda: ["jpeg"])
    compress_algorithm_probs: Annotated[
        list[float],
        Meta(
            description="Probability weights for each algorithm in compress_algorithms. Must match length of compress_algorithms."
        ),
    ] = field(default_factory=lambda: [1.0])
    compress_webp_range: Annotated[
        tuple[int, int],
        Meta(description="Quality range for WebP compression in stage 1 (1-100)."),
    ] = (50, 95)
    compress_h264_range: Annotated[
        tuple[int, int],
        Meta(description="CRF range for H.264 compression in stage 1 (0-51, lower = better)."),
    ] = (20, 40)
    compress_hevc_range: Annotated[
        tuple[int, int],
        Meta(description="CRF range for HEVC compression in stage 1 (0-51, lower = better)."),
    ] = (20, 40)
    compress_mpeg2_range: Annotated[
        tuple[int, int],
        Meta(description="qscale range for MPEG-2 compression in stage 1 (1-31, lower = better)."),
    ] = (2, 20)
    compress_mpeg4_range: Annotated[
        tuple[int, int],
        Meta(description="qscale range for MPEG-4 compression in stage 1 (1-31, lower = better)."),
    ] = (2, 20)
    compress_vp9_range: Annotated[
        tuple[int, int],
        Meta(description="CRF range for VP9 compression in stage 1 (0-63, lower = better)."),
    ] = (20, 50)
    compress_video_sampling: Annotated[
        list[str],
        Meta(
            description="List of video chroma subsampling options for video codecs in stage 1. Options: 444, 422, 420."
        ),
    ] = field(default_factory=lambda: ["444", "422", "420"])

    blur_prob2: Annotated[
        float,
        Meta(
            description="Probability of applying the second blur to the LQ, between 0 and 1."
        ),
    ] = 0
    resize_prob2: Annotated[
        list[float],
        Meta(
            description="List of 3 probabilities for the second resize which should add up to 1: the probability of upscaling, the probability of downscaling, and the probability of no resize."
        ),
    ] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    resize_mode_list2: Annotated[
        list[Literal["bilinear", "bicubic", "nearest-exact", "lanczos", "area"]],
        Meta(description="List of possible resize modes to use for the second resize."),
    ] = field(
        default_factory=lambda: ["bilinear", "bicubic", "nearest-exact", "lanczos"]
    )
    resize_mode_prob2: Annotated[
        list[float],
        Meta(
            description="List of probabilities for the second resize of selecting the corresponding resize mode in `resize_mode_list2`."
        ),
    ] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    resize_range2: Annotated[
        tuple[float, float],
        Meta(
            description="The resize range for the second resize, in the format `[min_resize, max_resize]`."
        ),
    ] = (0.6, 1.2)
    gaussian_noise_prob2: Annotated[
        float,
        Meta(
            description="The probability of applying the second gaussian noise to the LQ, between 0 and 1."
        ),
    ] = 0
    noise_range2: tuple[float, float] = (0, 0)
    poisson_scale_range2: tuple[float, float] = (0, 0)
    gray_noise_prob2: float = 0
    jpeg_prob2: Annotated[
        float,
        Meta(
            description="The probability of applying compression (JPEG or other configured algorithm) to the LQ in stage 2, between 0 and 1."
        ),
    ] = 1
    jpeg_range2: Annotated[
        list[float],
        Meta(
            description="The range of JPEG quality to apply for the second JPEG degradation, in the format `[min_quality, max_quality]`."
        ),
    ] = field(default_factory=lambda: [75, 95])
    compress_algorithms2: Annotated[
        list[str],
        Meta(
            description="List of compression algorithms to choose from in stage 2. Same options as compress_algorithms."
        ),
    ] = field(default_factory=lambda: ["jpeg"])
    compress_algorithm_probs2: Annotated[
        list[float],
        Meta(
            description="Probability weights for each algorithm in compress_algorithms2."
        ),
    ] = field(default_factory=lambda: [1.0])
    compress_webp_range2: Annotated[
        tuple[int, int],
        Meta(description="Quality range for WebP compression in stage 2 (1-100)."),
    ] = (50, 95)
    compress_h264_range2: Annotated[
        tuple[int, int],
        Meta(description="CRF range for H.264 compression in stage 2."),
    ] = (20, 40)
    compress_hevc_range2: Annotated[
        tuple[int, int],
        Meta(description="CRF range for HEVC compression in stage 2."),
    ] = (20, 40)
    compress_mpeg2_range2: Annotated[
        tuple[int, int],
        Meta(description="qscale range for MPEG-2 compression in stage 2."),
    ] = (2, 20)
    compress_mpeg4_range2: Annotated[
        tuple[int, int],
        Meta(description="qscale range for MPEG-4 compression in stage 2."),
    ] = (2, 20)
    compress_vp9_range2: Annotated[
        tuple[int, int],
        Meta(description="CRF range for VP9 compression in stage 2."),
    ] = (20, 50)
    compress_video_sampling2: Annotated[
        list[str],
        Meta(
            description="List of video chroma subsampling options for video codecs in stage 2."
        ),
    ] = field(default_factory=lambda: ["444", "422", "420"])

    resize_mode_list3: Annotated[
        list[Literal["bilinear", "bicubic", "nearest-exact", "lanczos", "area"]],
        Meta(description="List of possible resize modes to use for the final resize."),
    ] = field(
        default_factory=lambda: ["bilinear", "bicubic", "nearest-exact", "lanczos"]
    )
    resize_mode_prob3: Annotated[
        list[float],
        Meta(
            description="List of probabilities for the final resize of selecting the corresponding resize mode in `resize_mode_list3`."
        ),
    ] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])

    queue_size: Annotated[
        int,
        Meta(
            description="Queue size for OTF processing, must be a multiple of `batch_size_per_gpu`."
        ),
    ] = 120
    datasets: dict[str, DatasetOptions] = {}
    train: TrainOptions | None = None
    val: ValOptions | None = None
    logger: LogOptions | None = None
    dist_params: dict[str, Any] | None = field(
        default_factory=lambda: {"backend": "nccl", "port": 29500}
    )
    onnx: OnnxOptions | None = None

    find_unused_parameters: bool = False
    contents: str | None = None
