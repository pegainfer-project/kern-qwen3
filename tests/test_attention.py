"""Scatter into shuffled pages, then attend across ragged sequences on a real GPU.

The oracle uses PyTorch SDPA over contiguous tensors, independent of page layout.
"""

import ctypes as C
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


class Cuda:
    def __init__(self, path):
        self.lib = C.CDLL("libcuda.so.1")
        torch.empty(1, device="cuda")
        self.module = C.c_void_p()
        self.check(self.lib.cuModuleLoad(C.byref(self.module), str(path).encode()))

    def check(self, status):
        if status:
            raise RuntimeError(f"CUDA driver status {status}")

    def launch(self, name, grid, block, *arguments):
        function = C.c_void_p()
        self.check(self.lib.cuModuleGetFunction(C.byref(function), self.module, name.encode()))
        values = [C.c_void_p(x.data_ptr()) if isinstance(x, torch.Tensor) else C.c_int(x) for x in arguments]
        pointers = (C.c_void_p * len(values))(*(C.addressof(x) for x in values))
        self.check(
            self.lib.cuLaunchKernel(
                function,
                *map(C.c_uint, grid),
                *map(C.c_uint, block),
                C.c_uint(0),
                C.c_void_p(torch.cuda.current_stream().cuda_stream),
                pointers,
                None,
            )
        )
        torch.cuda.synchronize()

    def close(self):
        self.check(self.lib.cuModuleUnload(self.module))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class PagedAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        major, minor = torch.cuda.get_device_capability()
        cls.cuda = Cuda(ROOT / f"build/sm_{major}{minor}/qwen3.cubin")

    @classmethod
    def tearDownClass(cls):
        cls.cuda.close()

    def test_ragged_pages_and_causal_prefix(self):
        torch.manual_seed(814)
        cases = [(64, 1, 1), (128, 4, 3), (128, 8, 17), (256, 2, 32)]
        for dim, ratio, groups in cases:
            with self.subTest(dim=dim, ratio=ratio, groups=groups):
                kvheads, heads = 2, 2 * ratio
                lengths = torch.randint(17, 73, (groups,)).tolist()
                queries = torch.randint(1, 5, (groups,)).tolist()
                page_width = (max(lengths) + 15) // 16
                count = groups * page_width
                pages = (torch.randperm(count, device="cuda") + 1).to(torch.int32).view(groups, page_width)
                cache = torch.full(
                    ((count + 1) * 16, 2, kvheads, dim), float("nan"), dtype=torch.bfloat16, device="cuda"
                )
                offsets = [0]
                allq, expected = [], []
                for seq, (length, nquery) in enumerate(zip(lengths, queries)):
                    k = torch.randn(length, kvheads, dim, dtype=torch.bfloat16, device="cuda")
                    v = torch.randn_like(k)
                    q = torch.randn(nquery, heads, dim, dtype=torch.bfloat16, device="cuda")
                    positions = torch.arange(length, device="cuda")
                    slots = (pages[seq, positions // 16].long() * 16 + positions % 16).contiguous()
                    self.cuda.launch(
                        "cache_write", (length, 1, 1), (256, 1, 1), k, v, cache, slots, kvheads * dim
                    )
                    mask = (
                        torch.arange(length, device="cuda")[None, :]
                        <= (length - nquery + torch.arange(nquery, device="cuda"))[:, None]
                    )
                    reference = F.scaled_dot_product_attention(
                        q.transpose(0, 1).float(),
                        k.repeat_interleave(ratio, dim=1).transpose(0, 1).float(),
                        v.repeat_interleave(ratio, dim=1).transpose(0, 1).float(),
                        attn_mask=mask,
                    )
                    expected.append(reference.transpose(0, 1))
                    allq.append(q)
                    offsets.append(offsets[-1] + nquery)
                query = torch.cat(allq)
                output = torch.empty_like(query)
                lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
                offs = torch.tensor(offsets, dtype=torch.int32, device="cuda")
                self.cuda.launch(
                    "attention",
                    (offsets[-1], heads, 1),
                    (32, 1, 1),
                    output,
                    query,
                    cache,
                    pages,
                    lens,
                    offs,
                    groups,
                    page_width,
                    heads,
                    kvheads,
                    dim,
                )
                torch.testing.assert_close(output.float(), torch.cat(expected), atol=0.008, rtol=0.008)
                self.assertTrue(cache[:16].isnan().all().item(), "the reserved page was overwritten")


if __name__ == "__main__":
    unittest.main()
