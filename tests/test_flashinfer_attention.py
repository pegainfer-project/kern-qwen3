"""Run the actual manifest launches against SDPA with shuffled, poisoned KV pages."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import kern
import torch
import torch.nn.functional as F
from kern.host import region

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import attention


class Attention:
    def __init__(self, dim, ratio, groups, rows, context):
        kvheads, heads = 2, 2 * ratio
        width = context // 16
        count = groups * width + 1
        spec = {
            "q": ("bf16", [rows, heads, dim]),
            "output": ("bf16", [rows, heads, dim]),
            "cache": ("bf16", [count * 16, 2, kvheads, dim]),
            "pages": ("i32", [groups, width]),
            "lengths": ("i32", [groups]),
            "offsets": ("i32", [groups + 1]),
        }
        buffers = {
            n: {"dtype": dtype, "shape": shape, "kind": "output" if n == "output" else "input"}
            for n, (dtype, shape) in spec.items()
        }
        major, minor = torch.cuda.get_device_capability()
        directory = ROOT / f"build/sm_{major}{minor}"
        build = json.loads((directory / "build.json").read_text())
        ops = {}
        plan, dec, pre = attention.configure(
            buffers,
            ops,
            build["attention"],
            dim=dim,
            heads=heads,
            kvheads=kvheads,
            context=context,
            tokens=rows,
            seqs=groups,
        )
        calls = {}
        for name, args in [("decode", dec), ("prefill", pre)]:
            op = ops[f"attention_{name}"]
            op["params"][2:4] = ["in buffer<bf16>"] * 2
            call = {
                "op": f"attention_{name}",
                "args": [
                    {"buf": "output"},
                    {"buf": "q"},
                    {"buf": "cache"},
                    {"buf": "cache", "offset": kvheads * dim * 2},
                    *args,
                ],
            }
            calls[name] = {"calls": [plan, call]}
        m = {
            "schema_version": 7,
            "model": "attention-test",
            "vars": {"tokens": {"max": rows}, "seqs": {"max": groups}},
            "buffers": buffers,
            "states": {},
            "ops": ops,
            "modules": {
                k: v for k, v in build["modules"].items() if k in ("attention_plan", f"attention_{dim}")
            },
            "programs": calls,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(m))
            self.rt = kern.Runtime(str(path), str(directory), [], 0)
        self.b = {name: region(self.rt, m, name) for name in buffers}
        self.dim, self.ratio, self.groups, self.rows = dim, ratio, groups, rows
        self.context, self.width, self.count = context, width, count

    def inputs(self, lengths, queries):
        b, dim, ratio = self.b, self.dim, self.ratio
        b["cache"].fill_(float("nan"))
        b["attn.partial"].fill_(float("nan"))
        b["attn.lse"].fill_(float("nan"))
        pages = (
            (torch.randperm(self.count - 1, device="cuda") + 1).to(torch.int32).view(self.groups, self.width)
        )
        b["pages"].copy_(pages)
        b["lengths"].copy_(torch.tensor(lengths, device="cuda"))
        offsets, expected = [0], []
        for seq, (length, rows) in enumerate(zip(lengths, queries)):
            k = torch.randn(length, 2, dim, dtype=torch.bfloat16, device="cuda")
            v = torch.randn_like(k)
            q = torch.randn(rows, 2 * ratio, dim, dtype=torch.bfloat16, device="cuda")
            pos = torch.arange(length, device="cuda")
            slots = pages[seq, pos // 16].long() * 16 + pos % 16
            b["cache"][slots] = torch.stack((k, v), dim=1)
            b["q"][offsets[-1] : offsets[-1] + rows].copy_(q)
            mask = pos[None, :] <= (length - rows + torch.arange(rows, device="cuda"))[:, None]
            ref = F.scaled_dot_product_attention(
                q.transpose(0, 1).float(),
                k.repeat_interleave(ratio, dim=1).transpose(0, 1).float(),
                v.repeat_interleave(ratio, dim=1).transpose(0, 1).float(),
                attn_mask=mask,
            )
            expected.append(ref.transpose(0, 1))
            offsets.append(offsets[-1] + rows)
        b["offsets"].copy_(torch.tensor(offsets, device="cuda"))
        return torch.cat(expected)

    def run(self, program, rows):
        torch.cuda.synchronize()
        self.rt.run(program, {"tokens": rows, "seqs": self.groups})
        return self.b["output"][:rows].clone()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class FlashInferAttention(unittest.TestCase):
    def test_prefill_and_split_decode(self):
        torch.manual_seed(814)
        for dim, ratio in [(64, 1), (128, 2), (128, 4), (256, 8)]:
            with self.subTest(dim=dim, ratio=ratio):
                rt = Attention(dim, ratio, 1, 256, 8192)
                for length, rows in [(1, 1), (17, 16), (65, 33), (511, 128), (8192, 256)]:
                    expected = rt.inputs([length], [rows])
                    torch.testing.assert_close(
                        rt.run("prefill", rows).float(), expected, atol=0.015, rtol=0.015
                    )
                del rt
        for groups in (1, 3, 17, 32):
            with self.subTest(groups=groups):
                rt = Attention(128, 4, groups, groups, 8192)
                lengths = [([1, 15, 16, 17, 63, 64, 65, 511, 4096, 8192])[r % 10] for r in range(groups)]
                expected = rt.inputs(lengths, [1] * groups)
                torch.testing.assert_close(rt.run("decode", groups).float(), expected, atol=0.015, rtol=0.015)
                stream = torch.cuda.Stream()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    rt.rt.enqueue_after("decode", {"tokens": groups, "seqs": groups}, stream.cuda_stream)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    rt.rt.enqueue_after("decode", {"tokens": groups, "seqs": groups}, stream.cuda_stream)
                # Replay changes both the page mapping and the active split count.
                expected = rt.inputs([8193 - n for n in lengths], [1] * groups)
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(rt.b["output"].float(), expected, atol=0.015, rtol=0.015)
                self.assertTrue(rt.b["cache"][:16].isnan().all().item())
                del graph, rt
