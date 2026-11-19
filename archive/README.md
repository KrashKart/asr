# Archived exploratory notebooks

These are the original notebooks from the internship, moved here unchanged. They're kept for their saved outputs (figures and results); **they won't run as-is**. They import the old `utils` package and use machine-specific paths like `/home/jovyan/libs` and `../tedlium`.

To re-run something, use the demo notebooks in `notebooks/`, or port the cells using this mapping:

| Old | New |
|---|---|
| `from utils import attacks` | `from asrvuln import prepare, defences` |
| `from utils import gradient` | `from asrvuln import attack, evaluate` (`gradient.evaluate` → `evaluate.evaluate`) |
| `from utils import audio` | `from asrvuln import audio, defences, viz` (`audio.view_mel` → `viz.view_mel`) |
| `attention.init()` | `models.load_hf_whisper()` |
| `attention.inference / get_spikes / smoothen` | `asrvuln.analysis.attention` |
| `attention.plot_*` | `asrvuln.viz` |
| `from utils import causal` | `from asrvuln.analysis import causal` |
| `load_from_disk("../tedlium/test.hf")` | `data.load_tedlium("test")` |
| `tensors/snippets.pt` | `outputs/snippets/<mode>_snippets.pt` |

Known-broken notebooks: `whisper-dsmifgsm` (syntax error, unfinished), `test`, `playground` (scratch), `encoder_attn_attacked` (last cell uses an undefined variable), and `whisper_attention` (out of date with `inference`'s return values).
