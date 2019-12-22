import torch

from syft.exceptions import PureFrameworkTensorFoundError
from syft.frameworks.torch.tensors.decorators.logging import LoggingTensor
from syft.frameworks.torch.tensors.interpreters.autograd import AutogradTensor
from syft.frameworks.torch.tensors.interpreters.native import TorchTensor
from syft.frameworks.torch.tensors.interpreters.paillier import PaillierTensor
from syft.generic.frameworks.hook.hook_args import get_child
from syft.generic.frameworks.hook.hook_args import one
from syft.generic.frameworks.hook.hook_args import register_ambiguous_function
from syft.generic.frameworks.hook.hook_args import register_ambiguous_method
from syft.generic.frameworks.hook.hook_args import register_backward_func
from syft.generic.frameworks.hook.hook_args import register_forward_func
from syft.generic.frameworks.hook.hook_args import register_type_rule

type_rule = {
    torch.Tensor: one,
    torch.nn.Parameter: one,
    AutogradTensor: one,
    LoggingTensor: one,
    PaillierTensor: one,
}

forward_func = {
    torch.Tensor: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    torch.nn.Parameter: lambda i: i.child
    if hasattr(i, "child")
    else (_ for _ in ()).throw(PureFrameworkTensorFoundError),
    AutogradTensor: get_child,
    LoggingTensor: get_child,
    PaillierTensor: get_child,
}

backward_func = {
    TorchTensor: lambda i: i.wrap(),
    torch.Tensor: lambda i: i.wrap(),
    torch.nn.Parameter: lambda i: torch.nn.Parameter(data=i),
    AutogradTensor: lambda i: AutogradTensor(data=i).on(i, wrap=False),
    LoggingTensor: lambda i: LoggingTensor().on(i, wrap=False),
    PaillierTensor: lambda i: PaillierTensor().on(i, wrap=False),
}

# Methods or functions whose signature changes a lot and that we don't want to "cache", because
# they have an arbitrary number of tensors in args which can trigger unexpected behaviour
ambiguous_methods = {
    "__getitem__",
    "__setitem__",
    "_getitem_public",
    "add_",
    "backward",
    "chunk",
    "new",
    "permute",
    "reshape",
    "sub_",
    "view",
}

ambiguous_functions = {
    "torch.unbind",
    "unbind",
    "torch.stack",
    "stack",
    "torch.cat",
    "cat",
    "torch.mean",
    "torch.sum",
    "torch.chunk",
    "chunk",
    "torch.functional.split",
    "split",
    "backward",
}

register_ambiguous_method(*ambiguous_methods)
register_ambiguous_function(*ambiguous_functions)
register_type_rule(type_rule)
register_forward_func(forward_func)
register_backward_func(backward_func)
