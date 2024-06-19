import dataclasses
import logging
from typing import Optional, Any, Callable, Dict, List, Sequence, Tuple, Union

import torch

try:
    import brt
except ImportError:
    ...

log = logging.getLogger(__name__)


@dataclasses.dataclass
class CompiledArtifact:
    byre_file: str
    # TODO. serialize Session object.
    #byre_session: object
    none_indices: List[int]
    hash_key: Optional[str] = None
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str] = None


class ByteIRFunction:
    """
    Wrap the byteir compiled function and runtime as a callable object for dynamo, as dynamo caches callable in guards.
    """

    def __init__(self, module_path_or_session, none_indices):
        if isinstance(module_path_or_session, brt.Session):
            self._session = module_path_or_session
        else:
            self._session = brt.Session(alloc_func=caching_allocator_alloc,
                                        free_func=caching_allocator_delete)
            self._session.load(module_path_or_session)
        self._none_indices = none_indices
        self._req = self._session.new_request_context(
            torch.cuda.current_stream()._as_parameter_.value)
        self.input_arg_offsets = self._session.get_input_arg_offsets()
        self.output_arg_offsets = self._session.get_output_arg_offsets()

    def __call__(self, *inputs):
        from brt.utils import brt_dtype_to_torch_dtype

        log.debug(f"***** Run function compiled through byteir ******")

        # FIXME. byteir requires all inputs on device side, move host side tensor to device.
        # Preprocess the strided tensor as byteir does not support yet.
        new_inputs = []

        for i in range(0, len(inputs)):
            _t = inputs[i]
            if not _t.is_cuda:
                log.warning(f"device error: type={type(_t)}, {_t.device}")
                _t = _t.to("cuda")
            new_inputs.append(_t.contiguous())

        device = new_inputs[0].device

        results = [
            torch.empty(
                self._session.get_static_shape(offset),
                dtype=brt_dtype_to_torch_dtype(
                    self._session.get_data_type(offset)),
                device=device,
            ) for offset in self._session.get_output_arg_offsets()
        ]

        inputOffsetAndArg = [None] * len(new_inputs)
        outputOffsetAndArg = [None] * len(results)
        for idx, (offset, inp) in enumerate(zip(self.input_arg_offsets, new_inputs)):
            inputOffsetAndArg[idx] = (offset, inp.data_ptr())
        for idx, (offset, out) in enumerate(zip(self.output_arg_offsets, results)):
            outputOffsetAndArg[idx] = (offset, out.data_ptr())
        self._req.bind_args(inputOffsetAndArg)
        self._req.bind_args(outputOffsetAndArg)
        self._req.finish_io_binding()
        self._req.run()
        self._req.sync()

        # add None results to return values
        rets = []
        none_cnt = 0
        result_cnt = 0
        for i in range(len(results) + len(self._none_indices)):
            if none_cnt < len(
                    self._none_indices) and i == self._none_indices[none_cnt]:
                rets.append(None)
                none_cnt += 1
            else:
                rets.append(results[result_cnt])
                result_cnt += 1
        if len(rets) == 1:
            return rets[0]
        return rets
