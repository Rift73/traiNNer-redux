"""CPU-only tests for validation image schedules and their call sites."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest
import yaml
from traiNNer.utils.redux_options import DatasetOptions, ValOptions

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("iteration", [0, 1000, 300000, 300405, -1])
def test_no_filters_preserve_saving_every_validation(iteration: int) -> None:
    val = ValOptions(val_enabled=True, save_img=True)
    assert val.should_save_img(iteration)


@pytest.mark.parametrize("iteration", [-1, 10000, 15000, 25000, 305000])
def test_global_save_switch_remains_authoritative(iteration: int) -> None:
    val = ValOptions(
        val_enabled=True,
        save_img=False,
        save_img_freq=25000,
        save_img_iters=[10000, 15000],
    )
    dataset = DatasetOptions(
        name="Dataset1", type="pairedimagedataset", save_img_freq=5000
    )
    assert not val.should_save_img(iteration, dataset)


@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1000, False), (10000, True), (15000, True), (25000, False), (300000, False)],
)
def test_explicit_only_schedule_is_backward_compatible(
    iteration: int, expected: bool
) -> None:
    val = ValOptions(val_enabled=True, save_img=True, save_img_iters=[10000, 15000])
    assert val.should_save_img(iteration) is expected


def test_empty_explicit_list_without_interval_saves_only_standalone_tests() -> None:
    val = ValOptions(val_enabled=True, save_img=True, save_img_iters=[])
    assert not val.should_save_img(300000)
    assert val.should_save_img(-1)


@pytest.mark.parametrize("extras", [None, [], [10000, 15000]])
def test_interval_and_explicit_list_are_additive(extras: list[int] | None) -> None:
    val = ValOptions(
        val_enabled=True, save_img=True, save_img_freq=25000, save_img_iters=extras
    )
    for iteration in range(1000, 100001, 1000):
        expected = iteration % 25000 == 0 or iteration in (extras or [])
        assert val.should_save_img(iteration) is expected


@pytest.mark.parametrize(
    ("iteration", "global_save", "dataset_save"),
    [
        (10000, True, True),
        (15000, True, True),
        (300000, True, True),
        (300405, False, False),
        (301000, False, False),
        (305000, False, True),
        (310000, False, True),
        (325000, True, True),
        (850000, True, True),
        (-1, True, True),
    ],
)
def test_dataset_interval_override_and_absolute_iteration_alignment(
    iteration: int, global_save: bool, dataset_save: bool
) -> None:
    val = ValOptions(
        val_enabled=True,
        save_img=True,
        save_img_freq=25000,
        save_img_iters=[10000, 15000],
    )
    dataset = DatasetOptions(
        name="Dataset1", type="pairedimagedataset", save_img_freq=5000
    )
    assert val.should_save_img(iteration) is global_save
    assert val.should_save_img(iteration, dataset) is dataset_save


def test_dataset_filters_override_independently() -> None:
    val = ValOptions(
        val_enabled=True, save_img=True, save_img_freq=25000, save_img_iters=[1000]
    )
    interval_only = DatasetOptions(
        name="A", type="pairedimagedataset", save_img_freq=5000
    )
    assert val.should_save_img(1000, interval_only)  # Inherit the global explicit list.
    assert val.should_save_img(5000, interval_only)

    list_only = DatasetOptions(
        name="B", type="pairedimagedataset", save_img_iters=[2000]
    )
    assert not val.should_save_img(1000, list_only)
    assert val.should_save_img(2000, list_only)
    assert val.should_save_img(25000, list_only)  # Inherit the global interval.

    empty_list = DatasetOptions(name="C", type="pairedimagedataset", save_img_iters=[])
    assert not val.should_save_img(1000, empty_list)
    assert val.should_save_img(25000, empty_list)


@pytest.mark.parametrize("frequency", [0, -1, 0.5])
@pytest.mark.parametrize("options_type", [DatasetOptions, ValOptions])
def test_frequency_requires_a_positive_integer(
    frequency: int | float, options_type: type[DatasetOptions] | type[ValOptions]
) -> None:
    raw = (
        {"name": "Dataset1", "type": "pairedimagedataset"}
        if options_type is DatasetOptions
        else {"val_enabled": True, "save_img": True}
    )
    raw["save_img_freq"] = frequency
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(raw, options_type)


@pytest.mark.parametrize("options_type", [DatasetOptions, ValOptions])
def test_editor_schema_matches_image_saving_fields(
    options_type: type[DatasetOptions] | type[ValOptions],
) -> None:
    schema = json.loads((ROOT / "schemas/redux-config.schema.json").read_bytes())
    expected = msgspec.json.schema(options_type)["$defs"][options_type.__name__]
    actual = schema["$defs"][options_type.__name__]
    for field in ("save_img_freq", "save_img_iters"):
        assert actual["properties"][field] == expected["properties"][field]


def test_yaml_schedule_preserves_global_saves_and_overrides_dataset1() -> None:
    raw = yaml.safe_load("""
datasets:
  val_1: {name: Urban100, type: pairedimagedataset}
  val_2: {name: Manga109, type: pairedimagedataset}
  val_3: {name: Dataset1, type: pairedimagedataset, save_img_freq: 5000}
val:
  val_enabled: true
  val_freq: 1000
  save_img: true
  save_img_freq: 25000
  save_img_iters: [10000, 15000]
""")
    val = msgspec.convert(raw["val"], ValOptions)
    datasets = msgspec.convert(raw["datasets"], dict[str, DatasetOptions])
    assert val.val_freq == 1000
    assert val.save_img_freq == 25000
    assert val.save_img_iters == [10000, 15000]
    assert datasets["val_3"].save_img_freq == 5000

    old_global_saves = {10000, 15000, *range(25000, 850001, 25000)}
    for iteration in range(1000, 850001, 1000):
        for key in ("val_1", "val_2"):
            assert val.should_save_img(iteration, datasets[key]) is (
                iteration in old_global_saves
            )
        assert val.should_save_img(iteration, datasets["val_3"]) is (
            iteration % 5000 == 0
        )


@pytest.mark.parametrize("source_name", ["train.py", "test.py"])
def test_actual_validation_loops_pass_each_dataset_schedule(source_name: str) -> None:
    """Execute only the real validation for-loops, without training or CUDA."""
    tree = ast.parse((ROOT / source_name).read_text())
    variable = "val_loader" if source_name == "train.py" else "test_loader"
    loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == variable
    ]
    assert len(loops) == (2 if source_name == "train.py" else 1)
    val = ValOptions(val_enabled=True, save_img=True, save_img_freq=25000)
    datasets = [
        DatasetOptions(name="Urban100", type="pairedimagedataset"),
        DatasetOptions(name="Manga109", type="pairedimagedataset"),
        DatasetOptions(name="Dataset1", type="pairedimagedataset", save_img_freq=5000),
    ]
    loaders = [SimpleNamespace(dataset=SimpleNamespace(opt=opt)) for opt in datasets]
    for loop in loops:
        model = Mock()
        namespace = {
            "opt": SimpleNamespace(val=val),
            "BaseDataset": SimpleNamespace,
            "val_loaders": loaders,
            "test_loaders": loaders,
            "current_iter": 305000,
            "tb_logger": None,
            "multi_val_datasets": True,
            "model": model,
            "logger": Mock(),
        }
        module = ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[]))
        exec(compile(module, source_name, "exec"), namespace)
        calls = model.validation.call_args_list
        assert len(calls) == 3
        if source_name == "train.py":
            assert [call.args[3] for call in calls] == [False, False, True]
        else:
            assert [call.kwargs["save_img"] for call in calls] == [True, True, True]
