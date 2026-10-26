"""CPU regression tests for evaluation input dtype conversion."""

import warnings

import pytest
import torch

from kernelbench.eval import _process_input_tensor


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool]
)
def test_integer_and_boolean_inputs_are_preserved(dtype, precision):
    values = [0, 1] if dtype != torch.int64 else [0, 2**40 + 1]
    original = torch.tensor(values, dtype=dtype)
    actual = _process_input_tensor(original, "cpu", precision=precision)
    assert actual.dtype == dtype
    assert torch.equal(actual, original)


@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_floating_inputs_use_requested_precision(precision):
    original = torch.tensor([1.25, -2.5], dtype=torch.float64)
    actual = _process_input_tensor(original, "cpu", precision=precision)
    assert actual.dtype == precision
    assert torch.equal(actual, original.to(precision))


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("precision", [torch.float32, torch.float16, torch.bfloat16])
def test_complex_inputs_keep_existing_real_conversion(dtype, precision):
    original = torch.tensor([2 + 3j, -1 - 4j], dtype=dtype)
    # PyTorch warns when casting complex tensors to real tensors. Warning
    # emission may occur only once per process; verify the result directly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = _process_input_tensor(original, "cpu", precision=precision)
    assert actual.dtype == precision
    assert torch.equal(actual, original.real.to(precision))


def test_cross_entropy_class_indices_remain_valid():
    logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    targets = torch.tensor([2, 0], dtype=torch.int64)
    actual = torch.nn.functional.cross_entropy(
        _process_input_tensor(logits, "cpu"),
        _process_input_tensor(targets, "cpu"),
    )
    assert torch.equal(actual, torch.nn.functional.cross_entropy(logits, targets))


def test_non_tensor_input_is_unchanged():
    original = [1, 2]
    assert _process_input_tensor(original, "cpu") is original
