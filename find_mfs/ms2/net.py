"""
Torch-free inference for MistNet, the MS2 formula reranker

A numpy reimplementation of `mist_cf.mist_cf_score.mist_cf_model.MistNet`
(a `FormulaTransformer` over the precursor formula as CLS token plus the
assigned MS2 subpeak subformulae).

Weights come from the `.npz` artifact exported by mist-fmfs' `run_scripts/export_mistnet.py`

The model is small -- 457k parameters, ~1.7 MB

Everything runs in float32 to match the torch reference bit-for-bit-ish
(parity is asserted to < 1e-4 by the mist-fmfs test suite).

TWO DELIBERATE BUG REPRODUCTIONS
-----------------------------------
# TODO: Fix this!!

The reference implementation has two latent bugs.

Every existing checkpoint was *trained* with them, so they are part of the learned function:
 fixing them here would change scores i.e. performance/accuracy.

Both bugs are reproduced on purpose and marked BUG-FOR-BUG below.

See mist-fmfs `docs/PROJECT_STATE.md` Sec 5b.ii / Sec 5b.iii
for the write-ups and the ablations to run before changing either.

1. The pairwise "subset fragment" mask never took effect (boolean advanced
   indexing returns a copy, so the in-place fill was discarded).
2. Padded key positions are not masked out of attention -- the raw boolean mask
   is added to the logits instead of a -inf mask, so pads get a *+1 bonus*.

Because of (2), scores depend on how far the input is padded. This module always
pads to a fixed `max_subpeak + 1`, which makes output deterministic and independent of batching.
At mist-fmfs' default ``batch_size=256`` that is byte-identical to the reference
(measured: `max|delta score| = 0.0`), because a full batch's longest candidate already saturates
 the pad length.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import numpy as np

# Peak-type codes, mirroring mist_cf.mist_cf_score.mist_cf_data.
CLS_TYPE = 1
FRAG_TYPE = 0

# The shipped reranker weights, bundled as package data (see pyproject
# [tool.setuptools.package-data]). This is the NPLIB1 checkpoint exported by
# mist-fmfs; `MistNetNumpy.default()` / `with_ms2()` load it so consumers get
# MS2 reranking with no extra files.
_BUNDLED_NPZ = "mistnet_shipped.npz"


def bundled_npz_path() -> Path:
    """Filesystem path to the shipped MistNet ``.npz`` bundled with find-mfs."""
    return Path(str(resources.files(__package__) / "data" / _BUNDLED_NPZ))

_LAYER_NORM_EPS = 1e-5
_NUM_HEADS = 8


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``F.linear``: ``x @ w.T + b`` over the trailing axis."""
    return x @ w.T + b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``nn.LayerNorm`` over the trailing axis (population variance, as torch)."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + _LAYER_NORM_EPS) * w + b


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class MistNetNumpy:
    """
    Torch-free MistNet. Load with :meth:`from_npz`, score with :meth:`forward`.

    Attributes:
        max_subpeak: how many MS2 subpeaks the featurizer may keep.
        collapse_ions: whether the ion one-hot is the narrowed twin-blind head.
            **Must** match how the checkpoint was trained; read from the artifact.
        cls_mass_diff: whether the CLS token carries precursor mass error.
    """

    def __init__(self, weights: dict[str, np.ndarray], meta: dict, dtype=np.float32):
        self.meta = meta
        # float32 matches the torch reference. float64 is available for
        # numerical diagnostics (it costs ~2x and changes results by ~1e-4).
        self.dtype = np.dtype(dtype)
        self.hidden_size = int(meta["hidden_size"])
        self.n_layers = int(meta["layers"])
        self.max_subpeak = int(meta["max_subpeak"])
        self.ion_info = bool(meta["ion_info"])
        self.instrument_info = bool(meta["instrument_info"])
        self.cls_mass_diff = bool(meta["cls_mass_diff"])
        self.collapse_ions = bool(meta["collapse_ions"])
        self.num_heads = _NUM_HEADS
        self.head_dim = self.hidden_size // self.num_heads

        w = {k: v.astype(self.dtype) for k, v in weights.items()}
        self._w = w

        # Formula embedder: a plain lookup table, already baked into the
        # checkpoint (the 'abs-sines' math was folded in at training time).
        self.int_to_feat = w["xformer.form_encoder.int_to_feat_matrix"]
        self.extra_embed = w["xformer.form_encoder._extra_embeddings"]
        self.max_count_int = self.int_to_feat.shape[0]  # 255

        self.layers = [
            {
                "in_proj_w": w[f"xformer.peak_attn_layers.{i}.self_attn.in_proj_weight"],
                "in_proj_b": w[f"xformer.peak_attn_layers.{i}.self_attn.in_proj_bias"],
                "out_proj_w": w[f"xformer.peak_attn_layers.{i}.self_attn.out_proj.weight"],
                "out_proj_b": w[f"xformer.peak_attn_layers.{i}.self_attn.out_proj.bias"],
                "bias_u": w[f"xformer.peak_attn_layers.{i}.self_attn.bias_u"],
                "bias_v": w[f"xformer.peak_attn_layers.{i}.self_attn.bias_v"],
                "lin1_w": w[f"xformer.peak_attn_layers.{i}.linear1.weight"],
                "lin1_b": w[f"xformer.peak_attn_layers.{i}.linear1.bias"],
                "lin2_w": w[f"xformer.peak_attn_layers.{i}.linear2.weight"],
                "lin2_b": w[f"xformer.peak_attn_layers.{i}.linear2.bias"],
                "norm1_w": w[f"xformer.peak_attn_layers.{i}.norm1.weight"],
                "norm1_b": w[f"xformer.peak_attn_layers.{i}.norm1.bias"],
                "norm2_w": w[f"xformer.peak_attn_layers.{i}.norm2.weight"],
                "norm2_b": w[f"xformer.peak_attn_layers.{i}.norm2.bias"],
            }
            for i in range(self.n_layers)
        ]

    @classmethod
    def from_npz(cls, path: str | Path, dtype=np.float32) -> "MistNetNumpy":
        """Load the artifact written by mist-fmfs' ``export_mistnet.py``."""
        npz = np.load(Path(path), allow_pickle=False)
        meta = json.loads(str(npz["__meta__"]))
        weights = {k: npz[k] for k in npz.files if k != "__meta__"}
        return cls(weights, meta, dtype=dtype)

    @classmethod
    def default(cls, dtype=np.float32) -> "MistNetNumpy":
        """Load the reranker weights bundled with find-mfs (see :func:`bundled_npz_path`)."""
        return cls.from_npz(bundled_npz_path(), dtype=dtype)

    # -- embedding ---------------------------------------------------------- #

    def _embed_formula(self, counts: np.ndarray) -> np.ndarray:
        """Element counts ``(..., n_elem)`` -> embedding ``(..., n_elem * d)``.

        Mirrors ``nn_utils.form_embedder.IntFeaturizer.forward``: a row lookup per
        element count, with counts >= ``MAX_COUNT_INT`` falling back to the shared
        learned "extra" embedding. Negative counts index from the end of the table,
        exactly as torch does -- they only arise for pairwise diffs, which are
        abs()'d before they get here.
        """
        idx = counts.astype(np.int64)  # torch .long() truncates toward zero
        extra = idx >= self.max_count_int

        out = self.int_to_feat[np.where(extra, 0, idx)]
        if extra.any():
            out = np.where(extra[..., None], self.extra_embed[0], out)
        return out.reshape(*counts.shape[:-1], -1)

    # -- attention ---------------------------------------------------------- #

    def _attention(self, x, pairwise, key_padding_mask, p) -> np.ndarray:
        """Transformer-XL style attention with additive pairwise features.

        ``x`` is ``(L, B, H)``; ``pairwise`` is ``(B, L, L, H)``; ``key_padding_mask``
        is ``(B, L)`` with True marking padding.
        """
        L, B, H = x.shape
        nh, hd = self.num_heads, self.head_dim

        qkv = _linear(x, p["in_proj_w"], p["in_proj_b"])           # (L, B, 3H)
        q, k, v = np.split(qkv, 3, axis=-1)                        # each (L, B, H)

        # (L, B, H) -> (B*nh, L, hd)
        to_heads = lambda t: t.reshape(L, B * nh, hd).transpose(1, 0, 2)
        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # (B, L, L, H) -> (B*nh, L, L, hd)
        pw = pairwise.transpose(1, 2, 0, 3).reshape(L, L, B * nh, hd).transpose(2, 0, 1, 3)

        q = q / np.sqrt(hd, dtype=self.dtype)
        q4 = q.reshape(B, nh, L, hd)
        q1 = (q4 + p["bias_u"][None, :, None, :]).reshape(B * nh, L, hd)
        q2 = (q4 + p["bias_v"][None, :, None, :]).reshape(B * nh, L, hd)

        a_c = np.einsum("ble,bwe->blw", q1, k, optimize=True)
        b_d = np.einsum("ble,blwe->blw", q2, pw, optimize=True)
        attn = a_c + b_d

        if key_padding_mask is not None:
            mask = np.broadcast_to(
                key_padding_mask[:, None, None, :], (B, nh, 1, L)
            ).reshape(B * nh, 1, L)
            # BUG-FOR-BUG (2): the reference builds a -inf mask and then adds the
            # *boolean* mask instead, so padded keys get +1.0 rather than -inf and
            # are never actually masked out. Reproduced deliberately -- every
            # checkpoint was trained this way. See module docstring.
            attn = attn + mask.astype(self.dtype)

        attn = _softmax(attn)
        out = attn @ v                                             # (B*nh, L, hd)

        out = out.transpose(1, 0, 2).reshape(L * B, H)
        out = _linear(out, p["out_proj_w"], p["out_proj_b"])
        return out.reshape(L, B, H)

    def _encoder_layer(self, x, pairwise, key_padding_mask, p) -> np.ndarray:
        """Post-norm encoder layer (``norm_first=False``)."""
        x = _layer_norm(
            x + self._attention(x, pairwise, key_padding_mask, p),
            p["norm1_w"], p["norm1_b"],
        )
        ff = _linear(_relu(_linear(x, p["lin1_w"], p["lin1_b"])), p["lin2_w"], p["lin2_b"])
        return _layer_norm(x + ff, p["norm2_w"], p["norm2_b"])

    # -- forward ------------------------------------------------------------ #

    def forward(
        self,
        num_peaks: np.ndarray,
        peak_types: np.ndarray,
        form_vec: np.ndarray,
        ion_vec: np.ndarray,
        instrument_vec: np.ndarray,
        intens: np.ndarray,
        rel_mass_diffs: np.ndarray,
    ) -> np.ndarray:
        """Score a padded batch of candidates. Returns ``(B,)`` logits.

        Argument order and shapes match ``MistNet.forward``:
        ``form_vec`` ``(B, L, n_elem)``, ``ion_vec`` ``(B, L, n_ion)``,
        ``instrument_vec`` ``(B, L, n_instr)``, ``intens``/``rel_mass_diffs``
        ``(B, L)``, ``peak_types``/``num_peaks`` integer.

        Row 0 of each candidate is the CLS token (the precursor formula); rows
        ``1..n`` are matched subpeak subformulae.
        """
        form_vec = np.asarray(form_vec, dtype=self.dtype)
        B, L, _ = form_vec.shape

        is_cls = np.asarray(peak_types) == CLS_TYPE                # (B, L)
        cls_tokens = form_vec[is_cls]                              # (B, n_elem)
        diff_vec = cls_tokens[:, None, :] - form_vec

        cat = [self._embed_formula(form_vec), self._embed_formula(diff_vec),
               is_cls[:, :, None].astype(self.dtype)]
        if self.ion_info:
            cat.append(np.asarray(ion_vec, dtype=self.dtype))
        if self.instrument_info:
            cat.append(np.asarray(instrument_vec, dtype=self.dtype))

        num_peak_feat = np.broadcast_to(
            np.asarray(num_peaks, dtype=self.dtype)[:, None, None] / 10.0, (B, L, 1)
        )
        cat.extend([
            np.asarray(intens, dtype=self.dtype)[:, :, None],
            num_peak_feat,
            np.asarray(rel_mass_diffs, dtype=self.dtype)[:, :, None],
        ])
        input_vec = np.concatenate(cat, axis=-1)

        peak_tensor = _relu(_linear(
            input_vec,
            self._w["xformer.formula_encoder.input_layer.weight"],
            self._w["xformer.formula_encoder.input_layer.bias"],
        ))
        peak_tensor = peak_tensor.transpose(1, 0, 2)               # (L, B, H)

        # True marks padding.
        key_padding_mask = ~(np.arange(L)[None, :] < np.asarray(num_peaks)[:, None])

        # Pairwise features. diffs[b, i, j] = form_vec[b, j] - form_vec[b, i].
        form_diffs = form_vec[:, None, :, :] - form_vec[:, :, None, :]
        # BUG-FOR-BUG (1): the reference zeroes non-subset (cross-branch) pairs
        # here via `form_diffs[~same_sign].fill_(0)`, which is a no-op because
        # boolean advanced indexing returns a copy. So *every* pair contributes
        # abs(diff), including chemically meaningless ones. Not implementing the
        # intended mask is the faithful choice. See module docstring.
        form_diffs = np.abs(form_diffs)
        pairwise = _relu(_linear(
            self._embed_formula(form_diffs),
            self._w["xformer.pairwise_featurizer.input_layer.weight"],
            self._w["xformer.pairwise_featurizer.input_layer.bias"],
        ))

        for p in self.layers:
            peak_tensor = self._encoder_layer(peak_tensor, pairwise, key_padding_mask, p)

        # 'cls' pooling: select the CLS row of each candidate.
        pooled = np.einsum("nbd,bn->bd", peak_tensor, is_cls.astype(self.dtype))
        out = _linear(pooled, self._w["output_layer.weight"], self._w["output_layer.bias"])
        return out[:, 0]
