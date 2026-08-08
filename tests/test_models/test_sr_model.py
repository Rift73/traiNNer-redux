# ruff: noqa: SLF001

import os
import shutil
import sys
import tempfile
from os import path as osp
from types import SimpleNamespace

import torch
import traiNNer.models.sr_model as sr_model_module
from pytest import MonkeyPatch
from spandrel.architectures.ESRGAN import ESRGAN
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from traiNNer.data.paired_image_dataset import PairedImageDataset
from traiNNer.losses.gan_loss import GANLoss
from traiNNer.losses.mssim_loss import MSSIMLoss
from traiNNer.losses.perceptual_loss import PerceptualLoss
from traiNNer.models.sr_model import SRModel
from traiNNer.utils.config import Config
from traiNNer.utils.redux_options import DatasetOptions
from traiNNer.utils.types import DataFeed


def test_srmodel(monkeypatch: MonkeyPatch) -> None:
    options_path = "./options/train/ESRGAN/test_ESRGAN_gan.yml"
    shutil.copy("./options/_templates/train/ESRGAN/ESRGAN_gan.yml", options_path)
    args = ["", "-opt", options_path]
    monkeypatch.setattr(sys, "argv", args)

    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir, osp.pardir))
    opt, _ = Config.load_config_from_file(root_path, is_train=True)
    opt.num_gpu = 0
    opt.use_amp = False
    opt.path.pretrain_network_g = None
    opt.path.pretrain_network_g_ema = None
    assert opt.val is not None
    opt.val.val_enabled = True
    opt.val.metrics_enabled = True

    # build model
    model = SRModel(opt)
    # test attributes
    assert model.__class__.__name__ == "SRModel"
    assert isinstance(model.net_g, ESRGAN)
    assert isinstance(model.losses["l_g_mssim"], MSSIMLoss)
    assert isinstance(model.losses["l_g_perceptual"], PerceptualLoss)
    assert isinstance(model.losses["l_g_gan"], GANLoss)
    assert isinstance(model.optimizers[0], AdamW)
    assert model.ema_decay == 0.999

    # prepare data
    gt = torch.rand((1, 3, 32, 32), dtype=torch.float32)
    lq = torch.rand((1, 3, 8, 8), dtype=torch.float32)
    data: DataFeed = {"gt": gt, "lq": lq}
    model.feed_data(data)
    assert model.lq is not None
    assert model.gt is not None
    # check data shape
    assert model.lq.shape == (1, 3, 8, 8)
    assert model.gt.shape == (1, 3, 32, 32)

    # ----------------- test optimize_parameters -------------------- #
    model.optimize_parameters(1, 0, True)
    assert model.output is not None
    assert model.output.shape == (1, 3, 32, 32)
    assert isinstance(model.log_dict, dict)
    # check returned keys
    expected_keys = [
        "l_g_mssim",
        "l_g_cosim",
        "l_g_perceptual_conv3_4",
        "l_g_perceptual_conv4_4",
        "l_g_perceptual_conv5_4",
        "l_g_hsluv_hue",
        "l_g_hsluv_saturation",
        "l_g_hsluv_lightness",
        "l_g_gan",
        "l_g_total",
    ]
    assert set(expected_keys).issubset(set(model.log_dict.keys()))

    # ----------------- test save -------------------- #
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt.path.models = tmpdir
        model.opt.path.training_states = tmpdir
        model.save(0, 1)

    # ----------------- test the test function -------------------- #
    model.test()
    assert model.output.shape == (1, 3, 32, 32)
    # delete net_g_ema
    model.net_g_ema = None
    model.test()
    assert model.output.shape == (1, 3, 32, 32)
    assert model.net_g.training is True  # should back to training mode after testing

    # ----------------- test nondist_validation -------------------- #
    # construct dataloader
    dataset_opt = DatasetOptions(
        name="Test",
        dataroot_gt=["datasets/val/dataset1/hr"],
        dataroot_lq=["datasets/val/dataset1/lr"],
        io_backend={"type": "disk"},
        scale=4,
        phase="val",
        batch_size_per_gpu=6,
        use_hflip=True,
        use_rot=True,
        num_worker_per_gpu=8,
        type="PairedImageDataset",
    )
    dataset = PairedImageDataset(dataset_opt)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    assert model.is_train is True
    with tempfile.TemporaryDirectory() as tmpdir:
        model.opt.path.visualization = tmpdir
        model.nondist_validation(
            dataloader, 1, None, save_img=True, multi_val_datasets=False
        )
        assert model.is_train is True
        # check metric_results
        assert "topiq" in model.metric_results
        assert isinstance(model.metric_results["topiq"], float | Tensor)

    # in validation mode
    with tempfile.TemporaryDirectory() as tmpdir:
        assert model.opt.val is not None
        model.opt.is_train = False
        model.opt.val.suffix = "test"
        model.opt.path.visualization = tmpdir
        model.opt.val.pbar = True
        model.nondist_validation(
            dataloader, 1, None, save_img=True, multi_val_datasets=False
        )
        # check metric_results
        assert "topiq" in model.metric_results
        assert isinstance(model.metric_results["topiq"], float | Tensor)

        # if opt['val']['suffix'] is None
        model.opt.val.suffix = None
        model.opt.name = "demo"
        model.opt.path.visualization = tmpdir
        model.nondist_validation(
            dataloader, 1, None, save_img=True, multi_val_datasets=False
        )
        # check metric_results
        assert "topiq" in model.metric_results
        assert isinstance(model.metric_results["topiq"], float | Tensor)

    os.remove(options_path)


class _CountingIdentity(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eval_transitions = 0
        self.train_transitions = 0

    def train(self, mode: bool = True) -> "_CountingIdentity":
        if mode:
            self.train_transitions += 1
        else:
            self.eval_transitions += 1
        super().train(mode)
        return self

    def forward(self, value: Tensor) -> Tensor:
        return value


def _validation_lifecycle_model(
    network: nn.Module, *, is_train: bool
) -> SRModel:
    model = object.__new__(SRModel)
    model.opt = SimpleNamespace(
        is_train=is_train,
        input_pixel_format=None,
        output_pixel_format=None,
        val=SimpleNamespace(tile_size=0),
    )
    model.device = torch.device("cpu")
    model.amp_dtype = torch.bfloat16
    model.use_amp = False
    model.use_compile = False
    model.is_train = is_train
    model.net_g = network
    model.net_g_ema = None
    model.optimizers_schedule_free = []
    model.optimizer_g = None
    model._validation_inference_net = None
    model._validation_prepared_key = None
    model.lq = torch.rand(1, 3, 8, 8)
    model.gt = None
    model.output = None
    return model


def test_validation_reuses_one_eval_transition_per_weight_identity(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sr_model_module, "rgb2pixelformat_pt", lambda value, _format: value
    )
    monkeypatch.setattr(
        sr_model_module,
        "pixelformat2rgb_pt",
        lambda value, _target, _format: value,
    )
    network = _CountingIdentity()
    model = _validation_lifecycle_model(network, is_train=False)

    with model._validation_inference_context(current_iter=-1):
        model.test()
        model.test()

    assert network.eval_transitions == 1
    assert network.train_transitions == 0
    assert network.training is False

    # A second dataset using the same weights reuses the prepared eval state.
    with model._validation_inference_context(current_iter=-1):
        model.test()
    assert network.eval_transitions == 1

    # An EMA update invalidates the identity and causes exactly one refresh.
    model._validation_prepared_key = None
    with model._validation_inference_context(current_iter=-1):
        model.test()
    assert network.eval_transitions == 2


def test_training_validation_restores_network_and_model_modes(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sr_model_module, "rgb2pixelformat_pt", lambda value, _format: value
    )
    monkeypatch.setattr(
        sr_model_module,
        "pixelformat2rgb_pt",
        lambda value, _target, _format: value,
    )
    network = _CountingIdentity()
    model = _validation_lifecycle_model(network, is_train=True)

    with model._validation_inference_context(current_iter=1000):
        assert model.is_train is False
        assert network.training is False
        model.test()
        model.test()

    assert model.is_train is True
    assert network.training is True
    assert network.eval_transitions == 1
    assert network.train_transitions == 1
