"""CPU-only smoke tests: no model weights or dataset needed.

    Made with Claude.
"""
import pytest

torch = pytest.importorskip("torch")

from asrvuln import config, prepare
from asrvuln.analysis import attention

@pytest.fixture
def snippet():
    return torch.rand(1, 100) * 0.01 - 0.005

@pytest.fixture
def batch():
    return torch.rand(3, 1_000) * 2 - 1

def test_prepare_front_shape(snippet, batch):
    out = prepare.PrepareFront((1, 100))(snippet, batch)
    assert out.shape == (3, 1_100)
    assert torch.equal(out[:, :100], snippet.repeat(3, 1))

def test_prepare_at_position(snippet, batch):
    out = prepare.PrepareAtPosition((1, 100), position=10)(snippet, batch)
    assert out.shape == (3, 1_100)
    assert torch.equal(out[:, 10:110], snippet.repeat(3, 1))
    assert torch.equal(prepare.PrepareAtPosition((1, 100), 0)(snippet, batch),
                       prepare.PrepareFront((1, 100))(snippet, batch))

def test_prepare_overlay_shapes(batch):
    long_snippet = torch.zeros(1, 2_000)
    assert prepare.PrepareOverlay((1, 2_000))(long_snippet, batch).shape == (3, 2_000)
    assert prepare.PrepareOverlayFront((1, 100))(torch.zeros(1, 100), batch).shape == (3, 1_000)

def test_prepare_rejects_1d(snippet):
    with pytest.raises(ValueError):
        prepare.PrepareFront((1, 100))(snippet, torch.rand(1_000))

def test_mu_law_round_trip(batch):
    defences = pytest.importorskip("asrvuln.defences")
    assert torch.allclose(defences.mu_comp_decomp(batch), batch, atol=1e-4)

@pytest.mark.parametrize("name, kwargs", [
    ("PrepareFrontMu", {}),
    ("PrepareFrontMuComp", {}),
    ("PrepareFrontLowpass", {"cutoff": 6000}),
    ("PrepareFrontHighpass", {"cutoff": 120}),
    ("PrepareFrontBandpass", {"cutoffs": (120, 6000)}),
])
def test_defences_preserve_shape(name, kwargs, snippet, batch):
    defences = pytest.importorskip("asrvuln.defences")
    method = getattr(defences, name)((1, 100), **kwargs)
    assert method(snippet, batch).shape == (3, 1_100)

def test_encoder_limit():
    assert attention.encoder_limit(config.SAMPLE_RATE) == config.ENCODER_FPS + 1

def test_stack_cross_attns_shape():
    blocks, heads, enc = 4, 6, 1500
    # generate() returns one tuple per step, each with one (1, heads, 1, enc) tensor per block;
    # the first (prompt) step is skipped.
    steps = [tuple(torch.rand(1, heads, 3, enc) for _ in range(blocks))]
    steps += [tuple(torch.rand(1, heads, 1, enc) for _ in range(blocks)) for _ in range(5)]
    out = attention.stack_cross_attns(steps, limit=100)
    assert out.shape == (blocks, heads, 5, 100)

def test_get_spikes_returns_samples():
    attn = torch.zeros(2, 10)
    attn[0, 4] = 1.0
    indices, samples = attention.get_spikes(attn, 0.5)
    assert indices.tolist() == [4]
    assert samples.tolist() == [4 * config.SAMPLE_RATE // config.ENCODER_FPS]

def test_smoothen_length():
    frames = torch.rand(4, 3, 3)
    out = attention.smoothen(frames, factor=5)
    assert out.shape == (16, 3, 3)
    assert torch.equal(out[-1], frames[-1])
