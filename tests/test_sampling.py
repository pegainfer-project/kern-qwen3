"""The generated sample op through kern: filtering, replay, graphs and distribution."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import kern
import torch
from kern.host import region

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sampling


def probabilities(logits, temperature, top_k, top_p, min_p):
    """Sorted CPU oracle, including all tokens tied at a filter boundary."""
    x = logits.double()
    if temperature == 0:
        p = torch.zeros_like(x)
        p[x.argmax()] = 1
        return p
    p = (x / temperature).softmax(-1)
    if 0 < top_k < len(p):
        p[p < p.topk(top_k).values[-1]] = 0
    p /= p.sum()
    if top_p < 1:
        ordered = p.sort(descending=True).values
        crossing = min(int(torch.searchsorted(ordered.cumsum(0), top_p)), len(p) - 1)
        p[p < ordered[crossing]] = 0
    p[p < min_p * p.max()] = 0
    return p / p.sum()


class Sampler:
    def __init__(self, vocab, rows):
        major, minor = torch.cuda.get_device_capability()
        directory = ROOT / f"build/sm_{major}{minor}"
        build = json.loads((directory / "build.json").read_text())
        target = sampling.target(build["architecture"])
        buffers = sampling.fills()
        buffers.update(
            {
                "logits": {"dtype": "bf16", "shape": ["seqs", vocab], "kind": "input"},
                "lengths": {"dtype": "i32", "shape": ["seqs"], "kind": "input"},
                "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output"},
            }
        )
        m = {
            "schema_version": 7,
            "model": "sampling-test",
            "vars": {"seqs": {"max": rows}},
            "states": {},
            "buffers": buffers,
            "modules": {name: value for name, value in build["modules"].items() if name != "elementwise"},
            "ops": {"sample": sampling.op(vocab, rows, build["sampling"], target)},
            "programs": {"sample": {"calls": [{"op": "sample", "args": sampling.args(seq_lens="lengths")}]}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            path.write_text(json.dumps(m))
            self.rt = kern.Runtime(str(path), str(directory), [], 0)
        self.buffers = {name: region(self.rt, m, name) for name in buffers}
        self.rows = rows
        for name, value in (("temperature", 1), ("top_k", 0), ("top_p", 1), ("min_p", 0), ("lengths", 1)):
            self.buffers[name].fill_(value)
        self.buffers["seed"].copy_(torch.arange(rows, device="cuda") + 814)

    def run(self):
        torch.cuda.synchronize()
        self.rt.run("sample", {"seqs": self.rows})
        return self.buffers["next_token"].cpu().clone()


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class Sampling(unittest.TestCase):
    def test_filters_replay_and_graph(self):
        torch.manual_seed(71)
        settings = [
            (0, 0, 1, 0),
            (1, 1, 1, 0),
            (0.7, 5, 1, 0),
            (1.3, 0, 0.7, 0),
            (0.8, 0, 1, 0.3),
            (1, 9, 0.8, 0.2),
            (1, -1, 1, 1),
            (1, 999999, 0, 0),
        ]
        for vocab in (32, 1024, 151936):
            with self.subTest(vocab=vocab):
                rows = 65
                sampler = Sampler(vocab, rows)
                logits = torch.full((rows, vocab), -30.0, dtype=torch.bfloat16)
                logits[:, :16] = torch.randn(rows, 16).to(torch.bfloat16) * 2
                sampler.buffers["logits"].copy_(logits)
                for j, name in enumerate(("temperature", "top_k", "top_p", "min_p")):
                    sampler.buffers[name].copy_(
                        torch.tensor([settings[r % len(settings)][j] for r in range(rows)], device="cuda")
                    )
                expected = [probabilities(logits[r], *settings[r % len(settings)]) for r in range(rows)]
                initial = sampler.run()
                torch.testing.assert_close(sampler.run(), initial, rtol=0, atol=0)
                changed = False
                for position in range(2, 10):
                    sampler.buffers["lengths"].fill_(position)
                    tokens = sampler.run()
                    changed |= bool((tokens != initial).any())
                    for r, token in enumerate(tokens.tolist()):
                        self.assertGreater(expected[r][token].item(), 0, (vocab, r, token))
                self.assertTrue(changed, "position must advance the random stream")
                # Capture the exact generated op, including scratch resets and all filters.
                stream = torch.cuda.Stream()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    sampler.rt.enqueue_after("sample", {"seqs": rows}, stream.cuda_stream)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    sampler.rt.enqueue_after("sample", {"seqs": rows}, stream.cuda_stream)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(sampler.buffers["next_token"].cpu(), tokens, rtol=0, atol=0)
                sampler.buffers["seed"].add_(101)
                graph.replay()
                torch.cuda.synchronize()
                graph_tokens = sampler.buffers["next_token"].cpu().clone()
                torch.testing.assert_close(sampler.run(), graph_tokens, rtol=0, atol=0)
                self.assertTrue(bool((graph_tokens != tokens).any()))

    def test_distribution_after_filters(self):
        sampler = Sampler(32, 128)
        logits = torch.full((32,), -30.0, dtype=torch.bfloat16)
        logits[:8] = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05]).log().to(torch.bfloat16)
        sampler.buffers["logits"].copy_(logits.expand(128, -1))
        sampler.buffers["top_k"].fill_(5)
        sampler.buffers["top_p"].fill_(0.85)
        sampler.buffers["min_p"].fill_(0.4)
        expected = probabilities(logits, 1, 5, 0.85, 0.4)
        samples = []
        for position in range(32):
            sampler.buffers["lengths"].fill_(position + 1)
            samples.append(sampler.run())
        counts = torch.cat(samples).bincount(minlength=32).double()
        self.assertEqual(counts[expected == 0].sum().item(), 0)
        predicted = expected[expected > 0] * counts.sum()
        chi_square = ((counts[expected > 0] - predicted).square() / predicted).sum().item()
        self.assertLess(chi_square, 40, f"sampling distribution chi-square={chi_square}")
        print(f"sampling distribution: 4096 draws, chi-square={chi_square:.4f}")


if __name__ == "__main__":
    unittest.main()
