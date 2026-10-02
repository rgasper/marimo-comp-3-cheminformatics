# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.4",
#     "anywidget>=0.11.0",
#     "chemprop>=2.2.0",
#     "loguru>=0.7",
#     "marimo>=0.24.0",
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "pillow>=11",
#     "polars>=1.20",
#     "rdkit>=2024.9",
#     "requests>=2.32",
#     "scikit-learn>=1.5",
#     "scipy>=1.13",
#     "torch>=2.6",
#     "tqdm>=4.66",
#     "typeguard>=4.3",
# ]
# ///
import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # --- Bootstrap: make src/ and data/ available -------------------------
    # Platforms like molab's single-file "Add from GitHub" flow (and the
    # WebAssembly playground) only fetch *this* .py file, not the repo it
    # lives in -- so "import fingerprints" and the precomputed data/ caches
    # it reads would otherwise be missing. If a real checkout (local dev, or
    # a properly repo-synced molab notebook) is already present next to this
    # file, this is a no-op; otherwise it downloads the repo's src/ and
    # data/ directories straight from GitHub (no git, no API calls that hit
    # rate limits -- just one zip download) and puts src/ on sys.path, so
    # the rest of the notebook behaves identically either way.
    _GITHUB_REPO = "rgasper/marimo-comp-3-cheminformatics"
    _GITHUB_BRANCH = "main"

    import importlib.util
    import sys
    from pathlib import Path

    import marimo as mo

    def _bootstrap() -> None:
        if importlib.util.find_spec("fingerprints") is not None:
            return  # already installed (e.g. local `uv run`/editable install)

        here = mo.notebook_dir() or Path.cwd()
        local_src = here / "src"
        if (local_src / "fingerprints").is_dir():
            sys.path.insert(0, str(local_src))
            return  # repo was checked out alongside the notebook already

        import io
        import shutil
        import zipfile

        import requests

        cache_root = Path.home() / ".cache" / "fingerprints-notebook-bootstrap"
        extract_dir = cache_root / _GITHUB_BRANCH
        marker = extract_dir / ".complete"
        if not marker.exists():
            url = (
                f"https://github.com/{_GITHUB_REPO}/archive/refs/heads/"
                f"{_GITHUB_BRANCH}.zip"
            )
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                zf.extractall(cache_root)
            # the zip's sole top-level entry is "<repo>-<branch>/"
            (top,) = (p for p in cache_root.iterdir() if p.is_dir())
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            top.rename(extract_dir)
            marker.touch()
        sys.path.insert(0, str(extract_dir / "src"))

    _bootstrap()
    return (mo,)


@app.cell
def _(mo):
    import altair as alt
    import pandas as pd

    from fingerprints import cliff_view as cv

    return alt, cv, mo, pd


@app.cell
def _(mo):
    mo.md(r"""
    # Exploring Molecular Fingerprints and Activity Cliffs

    A **molecular fingerprint** turns a molecule into a fixed row of numbers so a computer can compare two
    molecules by comparing their rows. It's the workhorse representation behind
    a lot of cheminformatics operations and models.
    This notebook takes fingerprints apart to see what they encode and where that encoding commonly fails.

    To begin, we'll highlight one of the well-known failings of molecular fingerprints: **activity cliffs**. An activity cliff is the term for when a very small change in the structure of a molecule can cause radical changes in some of it's measurable properties. We'll spend most of the rest of the notebook trying to build an intuitive understanding for how fingerprints are related to activity cliffs, and also how they're not!

    *AI was used in the creation of this notebook. For the full disclaimer, head to the very bottom*
    """)
    return


@app.cell
def _(mo):
    # --- Setup -----------------------------------------------------------
    # Everything the plots need ships PRECOMPUTED under data/ (the cliff
    # censuses, kNN analysis, feature-importances, the accumulation module's
    # cross-validation, and the 3D Boltz poses), so the notebook loads
    # instantly. The only thing fetched on first run is the 33 MB CheMeleon
    # neural-fingerprint weight, which the live interactive cells need.
    from fingerprints import chemeleon_fp as _chf

    with mo.status.spinner(title="Fetching CheMeleon weights (first run only)…"):
        _chf.ensure_weights()
    setup_ready = True

    recompute_button = mo.ui.run_button(
        label="Recompute everything but Boltz poses",
        kind="danger",
        tooltip=(
            "Deletes every precomputed cache except the 3D Boltz poses, then "
            "re-downloads every source dataset and re-runs every analysis from "
            "scratch, in this notebook — roughly 15 minutes, dominated by the "
            "accumulation module's cross-validation and CheMeleon "
            "featurisation."
        ),
    )

    mo.vstack(
        [
            mo.md(
                "This notebook ships with its analyses **precomputed** so the "
                "plots load instantly. The button below reruns every analysis "
                "from scratch, right here in the notebook (there's no terminal "
                "on molab, so this replaces the `uv run python -m "
                "fingerprints.recompute` CLI command when you aren't running "
                "locally). The 3D binding poses are the one thing it doesn't "
                "rebuild — folding them needs a GPU and a paid Boltz API key, so "
                "they ship as data; to regenerate those specifically: "
                "`pip install '.[poses]'`, set `BOLTZ_API_KEY`, then run "
                "`python -m fingerprints.rebuild_poses` from a terminal."
            ).callout(kind="neutral"),
            recompute_button,
        ]
    )
    return recompute_button, setup_ready


@app.cell
def _(mo, recompute_button, setup_ready):
    # Runs only when the button above is clicked (mo.stop short-circuits
    # otherwise), and only once per click: a run_button's value resets to
    # False right after the cells that read it finish running.
    mo.stop(not recompute_button.value)

    from fingerprints import recompute as _recompute

    assert setup_ready
    _recompute.clear_outputs()
    _steps = _recompute.steps()
    with mo.status.progress_bar(
        total=len(_steps), title="Rebuilding analyses", remove_on_exit=False
    ) as _bar:
        for _step in _steps:
            _bar.update(increment=0, subtitle=_step.title)
            _step.run(
                lambda msg, _t=_step.title: _bar.update(
                    increment=0, subtitle=f"{_t}: {msg}"
                )
            )
            _bar.update(increment=1)
    # Drop every in-process cache that read the old (shipped) files, so the
    # cells below pick up the freshly-written ones without a kernel restart.
    _recompute.clear_caches()

    mo.md(
        "**Rebuilt everything from scratch** (except the 3D poses). Every plot "
        "below now reads the freshly-computed files — same code path, no "
        "shipped caches."
    ).callout(kind="success")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Here are two real molecules that differ by only a few atoms - we've got a few options you can choose from, to help demonstrate that this is not a phenomenon specific to this exact choice of chemicals or modifications. Whichever case you pick, the two molecules are nearly identical in structure in their fingerprint. Another confounding feature of activity cliffs is that the same molecular change can cause an activity cliff on for one property, but not the other!

    To see this in action, pick both a target pair and a molecule pair below. These are
    from intentionally chosen targets out of the MoleculeACE dataset where the same small change
    to chemical structure is a cliff on one target and only causes a correspondingly small change on the other, and the two targets are closely related.
    """)
    return


@app.cell
def _(mo):
    from fingerprints.data import context_cliffs as ctx

    # One canonical target-pair dropdown, bound to a GLOBAL so marimo tracks it
    # reactively. The molecule-pair dropdown is derived from it in the next cell
    # (its options depend on the chosen pair). Displaying the SAME element object
    # in several places keeps every copy in sync automatically - no mo.state,
    # no on_change handler recreating elements (which silently broke updates).
    pair_dd = mo.ui.dropdown(
        options=ctx.pair_options(),
        value=next(iter(ctx.pair_options())),
        label="Target pair",
    )
    return ctx, pair_dd


@app.cell
def _(ctx, mo, pair_dd):
    # Molecule-pair dropdown, rebuilt whenever the target pair changes so its
    # options match. Bound to a global (cliff_dd) for reactive tracking.
    _pk = pair_dd.value
    _cliff_opts = ctx.cliff_options(_pk)
    cliff_dd = mo.ui.dropdown(
        options=_cliff_opts,
        value=next(iter(_cliff_opts)),
        label="Molecule pair",
    )
    return (cliff_dd,)


@app.cell
def _(cliff_dd, ctx, mo, pair_dd):
    # Accessors used throughout the notebook. get_pair_key() / get_cliff_idx()
    # read the current dropdown selections; selectors() renders the synced row.
    def get_pair_key():
        return pair_dd.value

    def get_cliff_idx():
        _pk = pair_dd.value
        return ctx.clamp_cliff_idx(_pk, cliff_dd.value or 0)

    def selectors():
        """The synced (target-pair, molecule-pair) dropdown row. Renders the
        same global elements, so every placement stays in lock-step."""
        return mo.hstack([pair_dd, cliff_dd], justify="start", gap=2)

    return get_cliff_idx, get_pair_key, selectors


@app.cell
def _(ctx, get_pair_key, mo, selectors):
    _tp = ctx.by_key()[get_pair_key()]
    mo.vstack([mo.md(f"*{_tp.blurb}*"), selectors()])
    return


@app.cell
def _(ctx, cv, get_cliff_idx, get_pair_key, mo):
    _tp = ctx.by_key()[get_pair_key()]
    _pair = _tp.cliffs[get_cliff_idx()]
    _svg1, _svg2 = cv.pair_svgs(_pair, width=320, height=240)

    # Structures with the changed atoms highlighted in orange.
    _structures = mo.hstack(
        [
            mo.vstack([mo.Html(_svg1)], align="center"),
            mo.md("## →"),
            mo.vstack([mo.Html(_svg2)], align="center"),
        ],
        justify="center",
        gap=1,
    )
    _change = mo.md(
        f"The change: {_pair.change}.  \nThe orange atoms are all that differ "
        "between these two molecules."
    ).callout(kind="neutral")

    # Dual-endpoint activity readout: cliff on one, flat on the other.
    def _endpoint_card(target, pki1, pki2):
        delta = abs(pki1 - pki2)
        is_cliff = target == _pair.cliff_on
        fold = cv.fold_change(delta)
        kind = "danger" if is_cliff else "success"
        verdict = f"**{fold} potency change** — a cliff!" if is_cliff else (
            f"**{fold} — essentially unchanged** (flat)"
        )
        return mo.md(
            f"#### {target}\n\n"
            f"pKi: **{pki1}** → **{pki2}**  \n{verdict}"
        ).callout(kind=kind)

    _endpoints = mo.hstack(
        [
            _endpoint_card(_pair.target_a, _pair.pki_1_a, _pair.pki_2_a),
            _endpoint_card(_pair.target_b, _pair.pki_1_b, _pair.pki_2_b),
        ],
        widths=[1, 1],
        gap=2,
    )
    mo.vstack([_structures, _change, _endpoints, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### A quick look at the prediction target - protein binding

    These examples are real molecules that have to
    bind into a real protein pocket, and the strength of that complex interaction is what experimental binding strength actually measures. Below is the more-potent molecule of the pair you
    picked, folded into its binding pocket. The
    orange atoms are the handful that differ across the cliff, and the dashed
    lines indicate interactions between ligand and protein that match common heuristics as being important.
    """)
    return


@app.cell
def _(ctx, get_cliff_idx, get_pair_key, mo):
    from fingerprints import pose_view as _pv_teaser
    from fingerprints.complex_viewer import ComplexViewer as _ComplexViewerTeaser

    # First appearance of the custom 3D anywidget - deliberately a *hook*, not
    # an analysis. Show the single most-potent binder of the current cliff in
    # its cliff target's pocket so the reader can rotate a real complex before
    # we abstract everything into fingerprints. The full four-pose interaction
    # analysis stays at the end as the payoff. All curated pairs ship poses;
    # the has_poses guard is kept so the cell degrades gracefully regardless.
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    _cl = _tp.cliffs[_idx]

    if not _pv_teaser.has_poses(_tp.key, _idx):
        _teaser = mo.md(
            "*3D pose preview isn't available for this cliff — the full analysis "
            "at the end of the notebook covers the pairs that ship poses.*"
        ).callout(kind="info")
    else:
        _poses = _pv_teaser.load_all(_tp.key, _idx)
        _cliff_ep = _cl.cliff_on
        # the more-potent molecule of the pair ON the cliff target
        _pki1 = _cl.actual_pki(1, _cliff_ep)
        _pki2 = _cl.actual_pki(2, _cliff_ep)
        if _pki1 >= _pki2:
            _mol_id, _pki, _other = "mol1", _pki1, "mol2"
        else:
            _mol_id, _pki, _other = "mol2", _pki2, "mol1"
        _pose = _poses[f"{_mol_id}_{_cliff_ep}"]

        _changed = _pv_teaser.changed_atom_names(
            _pose, _poses[f"{_other}_{_cliff_ep}"]
        )
        _viewer = _ComplexViewerTeaser(
            structure=_pose.cif_text,
            format="cif",
            highlight_resi="",  # teaser is a clean first look - no pocket-residue landmark
            highlight_atoms=_changed,
            interactions=_pv_teaser.load_interactions(_pose),
            height=360,
        )
        _mol_name = "molecule 1" if _mol_id == "mol1" else "molecule 2"
        _caption = mo.md(
            f"*{_mol_name} — the more-potent binder here (pKi **{_pki}** on "
            f"**{_cliff_ep}**) — in its predicted pocket. Orange = the atoms that "
            f"change across the cliff; dashed lines = protein contacts (PLIP). "
            f"Predicted pose from Boltz not an experimental structure.*"
        )
        _teaser = mo.vstack([mo.ui.anywidget(_viewer), _caption])
    mo.vstack([_teaser, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    We will return to the 3D pocket at the end. First, though, we'll work with the far cheaper & more common representation that we're trying to understand better: the
    molecule's 2D structure, encoded as a fingerprint. So the question that
    we're digging into is whether that 2D encoding carries enough
    information to explain the cliffs for these targets.

    A fair place to start is with modeling the data. We
    have two nearly-identical molecules whose potency differs by orders of
    magnitude on one target but not the other. Can a statistical (machine-learned)
    model trained on fingerprints the measured binding data reproduce that gap? So, let's train some models on
    each target and check how well it predicts pKi — with special attention paid to these cliff-pairs.

    A little context for what follows: we
    encode each molecule two ways — **ECFP** which is a "classical" fingerprint, built by a relatively simple algorithm and is probably the most-used fingerprint in cheminformatics work; and **CheMeleon** a machine-learned fingerprint. More detail on how these fingerprints work will follow! For
    each target we train a **random forest** to predict pKi from that encoding.
    Each dataset is split into train and test sets by **Bemis–Murcko scaffolds**. All the cliff pairs we've picked happened to land in the training sets for their targets, so the model was trained on both molecules and their
    true potencies. Not to spoil the ending, but even with this advantage the models all fail to accurately learn the activity cliffs.
    """)
    return


@app.cell
def _(mo, selectors):
    # Synced selector right above the scatter, so the reader can switch the
    # target pair / molecule pair without scrolling back up.
    mo.vstack(
        [
            mo.md("**Pick the target pair / molecule pair for these plots:**"),
            selectors(),
        ]
    )
    return


@app.cell
def _(alt, ctx, get_cliff_idx, get_pair_key, mo, pd, setup_ready):
    from rdkit import Chem as _Chem2

    from fingerprints import importance_view as _iv_sc

    assert setup_ready

    # How good are these models, really? A measured-vs-predicted pKi scatter for
    # each of the two targets, ECFP fingerprint. Train points are faint, the
    # held-out (scaffold) test points solid, the diagonal is a perfect model,
    # and the currently-selected cliff pair is drawn as two big stars joined by
    # a line so you can watch the true gap (vertical span) collapse into a
    # near-vanishing predicted gap (horizontal span).
    _tp_sc = ctx.by_key()[get_pair_key()]
    _cl_sc = _tp_sc.cliffs[get_cliff_idx()]
    _eps_sc = _iv_sc.endpoints() if _iv_sc.has_data() else []
    _have_sc = _tp_sc.target_a in _eps_sc and _tp_sc.target_b in _eps_sc

    if not _have_sc:
        _scatter_view = mo.md(
            "*Predicted-vs-measured plots need the feature-importance cache — run "
            "`python -m fingerprints.analyses.importance`.*"
        ).callout(kind="info")
    else:
        _m1_sc = _Chem2.MolFromSmiles(_cl_sc.smiles_1)
        _m2_sc = _Chem2.MolFromSmiles(_cl_sc.smiles_2)

        # One shared, square domain across all four panels (so the y=x diagonal
        # is a true 45deg and panels are visually comparable). Base it on
        # PERCENTILES of the pooled values rather than raw min/max, so a lone
        # outlier prediction can't stretch the axis and leave whitespace. The
        # scales use clamp=True (below) so any point outside the domain is pinned
        # to the edge instead of dropped.
        import numpy as _np_sc

        _all_vals = []
        for _ep in (_cl_sc.cliff_on, _cl_sc.flat_on):
            for _fp in ("ecfp", "chemeleon"):
                for _fold in ("train", "test"):
                    _mv, _pv = _iv_sc.fold_scatter(_ep, _fp, _fold)
                    _all_vals.extend(_mv)
                    _all_vals.extend(_pv)
        _arr_sc = _np_sc.asarray(_all_vals, dtype=float)
        _g_lo = float(_np_sc.percentile(_arr_sc, 0.5)) - 0.2
        _g_hi = float(_np_sc.percentile(_arr_sc, 99.5)) + 0.2
        _dom = [_g_lo, _g_hi]

        def _target_chart(_ep, _is_cliff, _fp, _fp_label):
            _rows = []
            for _fold in ("train", "test"):
                _meas, _pred = _iv_sc.fold_scatter(_ep, _fp, _fold)
                for _mv, _pv in zip(_meas, _pred):
                    _rows.append({"measured": _mv, "predicted": _pv, "fold": _fold})
            _df = pd.DataFrame(_rows)
            _xsc = alt.Scale(domain=_dom, clamp=True)
            _ysc = alt.Scale(domain=_dom, clamp=True)
            _pts = (
                alt.Chart(_df)
                .mark_circle()
                .encode(
                    x=alt.X("measured:Q", title="measured pKi", scale=_xsc),
                    y=alt.Y("predicted:Q", title="predicted pKi", scale=_ysc),
                    color=alt.Color(
                        "fold:N",
                        scale=alt.Scale(domain=["train", "test"],
                                        range=["#748ffc", "#0ca678"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    opacity=alt.Opacity(
                        "fold:N",
                        scale=alt.Scale(domain=["train", "test"], range=[0.45, 0.55]),
                        legend=None,
                    ),
                    size=alt.Size(
                        "fold:N",
                        scale=alt.Scale(domain=["train", "test"], range=[16, 22]),
                        legend=None,
                    ),
                    tooltip=["fold:N", alt.Tooltip("measured:Q", format=".2f"),
                             alt.Tooltip("predicted:Q", format=".2f")],
                )
            )
            _diag = (
                alt.Chart(pd.DataFrame({"x": _dom, "y": _dom}))
                .mark_line(color="#adb5bd", strokeDash=[4, 3])
                .encode(x=alt.X("x:Q", scale=_xsc), y=alt.Y("y:Q", scale=_ysc))
            )
            # the selected cliff pair: measured (true) + predicted (cached RF)
            _cliff_rows = []
            for _mol, _idx in ((_m1_sc, 1), (_m2_sc, 2)):
                _p = _iv_sc.predict(_ep, _fp, _mol)
                if _p is not None:
                    _cliff_rows.append(
                        {"measured": _cl_sc.actual_pki(_idx, _ep),
                         "predicted": _p, "mol": f"molecule {_idx}"}
                    )
            _layers = [_diag, _pts]
            if len(_cliff_rows) == 2:
                _cdf = pd.DataFrame(_cliff_rows)
                _link = (
                    alt.Chart(_cdf)
                    .mark_line(color="#e8590c", strokeWidth=2)
                    .encode(x=alt.X("measured:Q", scale=_xsc),
                            y=alt.Y("predicted:Q", scale=_ysc))
                )
                _stars = (
                    alt.Chart(_cdf)
                    .mark_point(shape="diamond", size=260, filled=True,
                                color="#e8590c", stroke="black", strokeWidth=0.6)
                    .encode(
                        x=alt.X("measured:Q", scale=_xsc),
                        y=alt.Y("predicted:Q", scale=_ysc),
                        tooltip=["mol:N", alt.Tooltip("measured:Q", format=".2f"),
                                 alt.Tooltip("predicted:Q", format=".2f")],
                    )
                )
                _layers += [_link, _stars]
            _mt = _iv_sc.metrics(_ep, _fp)
            _tag = "cliff target" if _is_cliff else "flat target"
            return (
                alt.layer(*_layers)
                .resolve_scale(x="shared", y="shared")
                .properties(
                    width=230, height=230,
                    title=f"{_fp_label} · {_ep} ({_tag}) — test R² {_mt['r2']:.2f}",
                )
            )

        # 2x2: rows = fingerprint (ECFP, CheMeleon), cols = target (cliff, flat)
        _ecfp_row = mo.hstack(
            [
                mo.as_html(_target_chart(_cl_sc.cliff_on, True, "ecfp", "ECFP")),
                mo.as_html(_target_chart(_cl_sc.flat_on, False, "ecfp", "ECFP")),
            ],
            justify="center", gap=2,
        )
        _chem_row = mo.hstack(
            [
                mo.as_html(_target_chart(_cl_sc.cliff_on, True, "chemeleon", "CheMeleon")),
                mo.as_html(_target_chart(_cl_sc.flat_on, False, "chemeleon", "CheMeleon")),
            ],
            justify="center", gap=2,
        )
        _scatter_view = mo.vstack([
            mo.md(
                "**How good are the models?** Each dot is one molecule — "
                "<span style='color:#748ffc'>blue = training</span>, "
                "<span style='color:#0ca678'>green = held-out (scaffold) test</span>. "
                "The dashed line is a perfect prediction. The "
                "<span style='color:#e8590c'>orange diamonds</span> are the cliff "
                "pair you selected, joined by a line: their x-axis distance "
                "is the real potency gap, their y-axis distance is what the "
                "model predicts. On the cliff target you'll see the orange line "
                "tilt away from the diagonal: the predicted gap is real but "
                "**shrunken** — the model moves the two molecules apart, just not "
                "far enough. The top row uses the **ECFP** fingerprint, the bottom "
                "row the **CheMeleon** one; cycle through the pairs and you'll find "
                "the two fingerprints trade small wins, but the same partial-collapse "
                "shows up on every cliff."
            ),
            _ecfp_row,
            _chem_row,
            mo.md("---"),
        ])
    _scatter_view
    return


@app.cell
def _(alt, ctx, mo, pd, setup_ready):
    from rdkit import Chem as _Chem_sum

    from fingerprints import importance_view as _iv_sum

    assert setup_ready

    # Overall view across ALL curated pairs: for each pair and each fingerprint,
    # the TRUE potency gap between the two molecules (x) vs the gap the trained
    # model PREDICTS (y), on both the cliff target and the flat target. Flat
    # points (small true gap) sit on the y=x line; cliff points (large true gap)
    # fall well below it and plateau - the model reproduces small differences
    # but systematically under-calls large ones, on every pair.
    if not _iv_sum.has_data():
        _gap_summary_view = mo.md("")
    else:
        _eps_sum = _iv_sum.endpoints()
        _rows = []
        for _tp in ctx.by_key().values():
            for _i, _c in enumerate(_tp.cliffs):
                _m1 = _Chem_sum.MolFromSmiles(_c.smiles_1)
                _m2 = _Chem_sum.MolFromSmiles(_c.smiles_2)
                for _kind, _ep in (("cliff target", _c.cliff_on), ("flat target", _c.flat_on)):
                    if _ep not in _eps_sum:
                        continue
                    _tg = abs(_c.actual_pki(1, _ep) - _c.actual_pki(2, _ep))
                    for _fp, _fpl in (("ecfp", "ECFP"), ("chemeleon", "CheMeleon")):
                        _p1 = _iv_sum.predict(_ep, _fp, _m1)
                        _p2 = _iv_sum.predict(_ep, _fp, _m2)
                        if _p1 is None or _p2 is None:
                            continue
                        _rows.append({
                            "true_gap": round(_tg, 3),
                            "pred_gap": round(abs(_p1 - _p2), 3),
                            "kind": _kind,
                            "fingerprint": _fpl,
                            "pair": f"{_ep} #{_i}",
                        })
        _sdf = pd.DataFrame(_rows)
        _hi = float(max(_sdf["true_gap"].max(), _sdf["pred_gap"].max())) + 0.3
        _dom_sum = [0.0, _hi]
        _diag = (
            alt.Chart(pd.DataFrame({"x": _dom_sum, "y": _dom_sum}))
            .mark_line(color="#adb5bd", strokeDash=[4, 3])
            .encode(x=alt.X("x:Q", scale=alt.Scale(domain=_dom_sum)),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=_dom_sum)))
        )
        _pts = (
            alt.Chart(_sdf)
            .mark_point(size=90, filled=True, opacity=0.85, stroke="black", strokeWidth=0.4)
            .encode(
                x=alt.X("true_gap:Q", title="true potency gap (|Δ pKi|)",
                        scale=alt.Scale(domain=_dom_sum)),
                y=alt.Y("pred_gap:Q", title="predicted gap (|Δ pred pKi|)",
                        scale=alt.Scale(domain=_dom_sum)),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(domain=["flat target", "cliff target"],
                                    range=["#0ca678", "#e8590c"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                shape=alt.Shape(
                    "fingerprint:N",
                    scale=alt.Scale(domain=["ECFP", "CheMeleon"],
                                    range=["circle", "triangle"]),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["pair:N", "kind:N", "fingerprint:N",
                         alt.Tooltip("true_gap:Q", format=".2f"),
                         alt.Tooltip("pred_gap:Q", format=".2f")],
            )
        )
        _chart = (
            alt.layer(_diag, _pts)
            .resolve_scale(x="shared", y="shared")
            .properties(width=380, height=340,
                        title="true vs predicted potency gap — every curated pair")
        )
        _gap_summary_view = mo.vstack([
            mo.md(
                "### Prediction Accuracy of all the cliff-pairs\n\n"
                "One point per curated pair × fingerprint. The x-axis is the real "
                "potency gap between the two molecules and the y-axis is the gap the "
                "trained model predicted. 1:1 on the dashed line. "
                "<span style='color:#0ca678'>Green = the flat target</span> (where the "
                "two molecules really are close in measurement), "
                "<span style='color:#e8590c'>orange = the cliff target</span> (where "
                "they're far apart); circles are ECFP, triangles CheMeleon."
            ),
            mo.as_html(_chart),
            mo.md(
                "On the flat targets the models don't perfectly capture the property change caused by the molecular change, but the error is typically quite small in practical terms. On the cliff targets however, every prediction consistently under-estimates the actual propery change. This is happening with both fingerprints and even though the models trained on all of this data! As well, one particularly bad outlier for the Dopamine targets actually predicted the cliff in the opposite direction - reducing potency instead of increasing it."
            ),
            mo.md("---"),
        ])
    _gap_summary_view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    # What does a fingerprint actually encode?

    To understand why the fingerprints are failing to notice this huge difference in activity, we need to see in detail what a
    fingerprint records in the first place. Fingerprints come in two families:

    - **Classical** — MACCS, Morgan/ECFP, and the other
      fingerprints available for computation by RDKit. In each of these, a person or a fixed algorithm decided in advance which chemical substructures cause which fingerprint bits to activate. A common trait shared by all these specific "classical" fingerprints is that each dimension is binary - a bit that is either on or off; this can be changed to enable these fingerprints to count up repeat copies of the substructure driving a dimension, but this comes with tradeoffs in interpretability and use cases, and is not the default usage.
    - **Learned** — Fingerprints that come out of pre-trained neural networks. In this notebook, we'll be focusing on the CheMeleon fingerprint, but there are many others. CheMeleon is a neural network *pre-trained* on millions of unique molecules to read the molecular graph and predict a wide swath of physicochemical properties; we then extract out the final embedding vector before the MLP decision head to use as a fingerprint. In contrast to the "classical" fingerprints, most if not all of the learned fingerprints have continuously varying dimensions. 

    These two fingerprint types differ significantly, but share one important aspect - they're static. They can't reactively change to new contexts in chemical or target variable space. Below we look at exactly what chemical features the fingerprints encode. There's another picker for the same sets of molecules that were studied above, or you can input a custom SMILEs for this section. Then you can scrub through bits/dimensions to see what parts of the molecule each fingerprint records.
    """)
    return


@app.cell
def _(ctx, get_cliff_idx, get_pair_key, mo, selectors):
    # Reuse the SAME two molecules from the cliff the reader picked above,
    # instead of an unrelated gallery — it keeps this section tied to the
    # story. The menu is rebuilt reactively from the current (pair, cliff)
    # selection; a synced copy of the selectors is shown here so the reader
    # can switch cliffs without scrolling back up.
    _mol_opts = ctx.cliff_molecule_options(get_pair_key(), get_cliff_idx())
    mol_choice = mo.ui.dropdown(
        options=_mol_opts,
        value=next(iter(_mol_opts)),
        label="Molecule from this cliff",
    )
    custom_smiles = mo.ui.text(
        value="",
        label="…or type your own SMILES (overrides the dropdown)",
        full_width=True,
        placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O",
    )
    _picker = mo.vstack(
        [
            mo.md("**Switch the cliff pair (synced with the sections above):**"),
            selectors(),
        ]
    )
    _picker
    return custom_smiles, mol_choice


@app.cell
def _(custom_smiles, mo, mol_choice):
    from fingerprints import maccs_explorer as mx

    # Resolve the single upstream molecule: custom SMILES wins if provided,
    # otherwise use the molecule picked from the current cliff pair. The
    # dropdown's VALUE is already the molecule's SMILES (see
    # ctx.cliff_molecule_options), so everything downstream depends only on
    # `current_mol` / `current_label`.
    _typed = custom_smiles.value.strip()
    if _typed:
        _mol = mx.mol_from_smiles(_typed)
        _source_label = "custom SMILES"
        _picked_note = ""
    else:
        _smiles = mol_choice.value
        _mol = mx.mol_from_smiles(_smiles)
        # the dropdown key (e.g. "Molecule 1 — before (…)") is the nice label
        _source_label = next(
            (k for k, v in mol_choice.options.items() if v == _smiles),
            "cliff molecule",
        )
        _picked_note = "one of the two molecules from the cliff above"

    current_mol = _mol
    mol_valid = current_mol is not None
    current_label = _source_label

    if _typed and not mol_valid:
        _feedback = mo.md(
            f"⚠️ Could not parse SMILES `{_typed}` — showing nothing downstream "
            f"until it's valid."
        ).callout(kind="warn")
    elif mol_valid:
        from rdkit import Chem

        _canon = Chem.MolToSmiles(current_mol)
        _note = f" — *{_picked_note}*" if _picked_note else ""
        _feedback = mo.md(
            f"**Active molecule: {current_label}**{_note}  \n`{_canon}`"
        ).callout(kind="success")
    else:
        _feedback = mo.md("")

    mo.vstack([mo.hstack([mol_choice, custom_smiles], widths=[1, 2]), _feedback])
    return current_mol, mol_valid, mx


@app.cell
def _(current_mol, mo, mol_valid, mx):
    if mol_valid:
        _svg = mx.highlight_svg(
            current_mol,
            mx.blank_hit(),
            width=360,
            height=260,
        )
        _view = mo.Html(_svg)
    else:
        _view = mo.md("*No valid molecule selected.*")
    _view
    return


@app.cell
def _(mo, mol_valid, mx):
    # Always scrub all 166 MACCS keys, in order, so OFF bits are explorable too.
    scrub_bits = mx.all_bits() if mol_valid else []
    if scrub_bits:
        bit_slider = mo.ui.slider(
            start=0,
            stop=len(scrub_bits) - 1,
            value=0,
            label="Scrub all 166 MACCS keys",
            full_width=True,
            show_value=False,
        )
    else:
        bit_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")
    return bit_slider, scrub_bits


@app.cell
def _(mo):
    mo.md("""
    ## MACCS Fingerprint
    """)
    return


@app.cell
def _(bit_slider, current_mol, mo, mol_valid, mx, scrub_bits):
    if mol_valid and scrub_bits:
        _bit = scrub_bits[bit_slider.value]
        _hit = mx.bit_hit(current_mol, _bit)
        _query = mx.query_svg(_bit, width=240, height=190)

        # Show where it matches (ON) or the plain molecule (OFF). The match/OFF
        # state is already spelled out in the match line below, so no separate
        # "present" badge is needed — it would be redundant.
        if _hit.is_on:
            _mol_svg = mx.highlight_svg(current_mol, _hit, width=460, height=340)
            _mol_panel = mo.vstack([mo.md("**Where it matches:**"), mo.Html(_mol_svg)])
        else:
            _mol_svg = mx.highlight_svg(current_mol, mx.blank_hit(), width=460, height=340)
            _mol_panel = mo.vstack(
                [
                    mo.md("**Not present** — the molecule is shown plain:"),
                    mo.Html(_mol_svg),
                ]
            )

        # "What the bit looks for": the SMARTS query depiction, or — for the
        # three procedurally-computed keys — a plain-language explanation.
        if _query is not None:
            _query_panel = mo.vstack(
                [mo.md("**What this bit looks for:**"), mo.Html(_query)]
            )
        else:
            _query_panel = mo.vstack(
                [
                    mo.md("**What this bit looks for:**"),
                    mo.md(mx.describe_special(_bit) or "*No drawable pattern.*").callout(
                        kind="info"
                    ),
                ]
            )

        # Headline the plain-English description; tuck the terse SMARTS and the
        # official MDL key definition into an expandable "technical details" pane
        # so nobody has to parse a SMARTS string to understand the bit.
        if _hit.is_special:
            _match_line = mo.md(
                "*Special key — RDKit computes this one directly instead of by "
                "matching a SMARTS pattern (see explanation at right).*"
            )
            _details = mo.accordion(
                {
                    "Technical details": mo.md(
                        f"**Official MACCS key:** `{_hit.official}`  \n"
                        "**SMARTS:** *(none — computed procedurally)*"
                    )
                }
            )
        else:
            if _hit.threshold > 0:
                _match_line = mo.md(
                    f"This key needs **more than {_hit.threshold}** matches to turn "
                    f"on. Found **{_hit.match_count}** → "
                    f"{'**ON**' if _hit.is_on else '**OFF**'}."
                )
            else:
                _match_line = mo.md(f"Matches in this molecule: **{_hit.match_count}**")
            _details = mo.accordion(
                {
                    "Technical details": mo.md(
                        f"**Official MACCS key:** `{_hit.official}`  \n"
                        f"**SMARTS:** `{_hit.smarts}`"
                    )
                }
            )

        _header = mo.vstack(
            [
                mo.md(f"### Bit {_hit.bit} · {_hit.name}"),
                _match_line,
                _details,
            ]
        )
        # The strip goes below the molecule image (same layout as Morgan): it's
        # easier to parse the full-fingerprint context after seeing the match.
        _strip = mx.fingerprint_strip_svg(current_mol, _bit, width=920, height=44)
        _strip_legend = mo.md(
            '<span style="color:#2f9e44">█ on</span> &nbsp; '
            '<span style="color:#adb5bd">░ off</span> &nbsp; '
            '<span style="color:#1c7ed6">█ current bit</span>'
        )
        _view = mo.vstack(
            [
                mo.md("---"),
                _header,
                mo.hstack([_query_panel, _mol_panel], justify="start", gap=2, widths=[1, 2]),
                mo.md("**Where this bit sits in the whole 166-bit fingerprint:**"),
                mo.Html(_strip),
                _strip_legend,
                bit_slider,
                mo.md("---"),
            ]
        )
    else:
        _view = mo.vstack(
            [mo.md("*Select a valid molecule to explore its MACCS bits.*"), mo.md("---")]
        )
    _view
    return


@app.cell
def _(mo):
    mo.md("""
    ## Extended Connectivity Fingerprint (ECFP) AKA Morgan Fingerprint
    """)
    return


@app.cell
def _(current_mol, mo, mol_valid):
    from fingerprints import morgan_explorer as me

    if mol_valid:
        _on = me.on_bits(current_mol)
    else:
        _on = []
    morgan_on_bits = _on

    if _on:
        morgan_slider = mo.ui.slider(
            start=0,
            stop=len(_on) - 1,
            value=0,
            label=f"Scrub the {len(_on)} Morgan bits that are ON (radius 2, 2048 bits)",
            full_width=True,
            show_value=False,
        )
    else:
        morgan_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")
    return me, morgan_on_bits, morgan_slider


@app.cell
def _(current_mol, me, mo, mol_valid, morgan_on_bits, morgan_slider):
    if mol_valid and morgan_on_bits:
        _bit = morgan_on_bits[morgan_slider.value]
        _hit = me.bit_hit(current_mol, _bit)
        _svg = me.highlight_svg(current_mol, _hit, width=460, height=340)
        _radii = ", ".join(str(r) for r in _hit.radii)
        _center_word = "center" if _hit.n_instances == 1 else "centers"
        _card = mo.vstack(
            [
                mo.md(f"### Bit {_hit.bit}"),
                mo.md(
                    f"Set by **{_hit.n_instances}** atom {_center_word} "
                    f"(radius {_radii})."
                ),
                mo.md(
                    f"Environment spans **{len(_hit.atoms)} atoms** "
                    f"and **{len(_hit.bonds)} bonds**."
                ),
                mo.md(
                    "*Radius 0 bits are single atoms; larger radii capture more "
                    "of the surrounding structure.*"
                ),
            ]
        )
        _view = mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[1, 1])
    else:
        _view = mo.md("*Select a valid molecule to explore its Morgan bits.*")
    _view
    return


@app.cell
def _(current_mol, me, mo, mol_valid, morgan_on_bits, morgan_slider):
    # Same strip idea as MACCS, but Morgan is long (2048) and sparse: an OFF bit
    # means "no environment happened to hash here" — it has no specific meaning,
    # so the scrubber only visits ON bits and the strip is mostly blank.
    if mol_valid and morgan_on_bits:
        _bit = morgan_on_bits[morgan_slider.value]
        _strip = me.fingerprint_strip_svg(current_mol, _bit, width=920, height=36)
        _legend = mo.md(
            '<span style="color:#2f9e44">█ on</span> &nbsp; '
            '<span style="color:#1c7ed6">█ current bit</span> &nbsp; '
            "the rest is off"
        )
        _note = mo.md(
            f"Only **{len(morgan_on_bits)} of 2048** bits are on — Morgan vectors "
            "are sparse. Unlike MACCS, an off bit here carries no meaning of its "
            "own - it just means no atom environment hashed to that bit, so the "
            "scrubber skips straight between the on bits."
        )
        _view = mo.vstack(
            [mo.md("**The whole 2048-bit fingerprint:**"), mo.Html(_strip), _legend, _note,
             morgan_slider, mo.md("---")]
        )
    else:
        _view = mo.md("---")
    _view
    return


@app.cell
def _(current_mol, me, mo, mol_valid):
    _SHORT = 8
    _collisions = me.find_collisions(current_mol, n_bits=_SHORT) if mol_valid else []
    if _collisions:
        collision_slider = mo.ui.slider(
            start=0,
            stop=len(_collisions) - 1,
            value=0,
            label=f"Scrub the {len(_collisions)} colliding bits at {_SHORT} bits",
            full_width=True,
            show_value=False,
        )
    else:
        collision_slider = mo.ui.slider(start=0, stop=0, value=0, label="(no collisions)")
    short_collisions = _collisions
    return collision_slider, short_collisions


@app.cell
def _(collision_slider, current_mol, me, mo, mol_valid, short_collisions):
    _PALETTE_HEX = ["#e64d3d", "#338cf2", "#33a659", "#d98c1a", "#9959cc"]
    if not mol_valid:
        collision_card = mo.md("*Select a valid molecule.*")
    elif not short_collisions:
        collision_card = mo.md(
            "This molecule has so few atom environments that **none of them "
            "collide** even at 8 bits — try a bigger drug-like molecule from "
            "the selector (e.g. Gefitinib or Imatinib)."
        ).callout(kind="info")
    else:
        _col = short_collisions[collision_slider.value]
        _svg = me.collision_svg(current_mol, _col, width=520, height=380)
        # One legend row per colliding environment, colored to match the drawing.
        _rows = []
        for _i, _sig in enumerate(_col.signatures):
            _c = _PALETTE_HEX[_i % len(_PALETTE_HEX)]
            _label = _sig.replace("atom:", "single atom ")
            _rows.append(f'<span style="color:{_c}">█</span> `{_label}`')
        _legend = mo.md("  \n".join(_rows))
        _card = mo.vstack(
            [
                mo.md(f"### Bit {_col.bit} at 8 bits"),
                mo.md(
                    f"**{_col.n_distinct} different substructures** all hash to this "
                    "one bit. To the fingerprint they are indistinguishable:"
                ),
                _legend,
                mo.md(
                    "*In a 2048-bit fingerprint these substructures would almost always land "
                    "on separate bits — that extra length allows greater specificity."
                ),
            ]
        )
        collision_card = mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[3, 2])
    return (collision_card,)


@app.cell
def _(alt, current_mol, me, mo, mol_valid, pd):
    # Collision rate as the vector lengthens: the payoff of a longer fingerprint.
    if mol_valid:
        _curve = me.collision_curve(current_mol)
        _distinct = _curve[0].distinct_envs
        _df = pd.DataFrame(
            {
                "length": [cp.n_bits for cp in _curve],
                "rate": [
                    round(100 * cp.collisions / cp.distinct_envs, 1)
                    if cp.distinct_envs
                    else 0.0
                    for cp in _curve
                ],
            }
        )
        _chart = (
            alt.Chart(_df)
            .mark_line(point=True, color="#e8590c")
            .encode(
                x=alt.X(
                    "length:O",
                    title="fingerprint length (bits)",
                    sort=[str(cp.n_bits) for cp in _curve],
                ),
                y=alt.Y(
                    "rate:Q",
                    title="collision rate (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("length:O", title="bits"),
                    alt.Tooltip("rate:Q", title="collision rate %"),
                ],
            )
            .properties(height=200, title="Collision rate vs. fingerprint length")
        )
        collision_curve_view = mo.vstack(
            [
                mo.as_html(_chart),
                mo.md(
                    f"This molecule has {_distinct} distinct atom environments. "
                    "The collision rate falls off fast — which is why **2048 bits** is a common "
                    "default: long enough that collisions are rare, short enough to "
                    "stay cheap."
                ),
            ]
        )
    else:
        collision_curve_view = mo.md("")
    return (collision_curve_view,)


@app.cell
def _(collision_card, collision_curve_view, collision_slider, mo):
    mo.vstack(
        [
            mo.accordion(
                {
                    "🔍 Aside: why is a Morgan fingerprint 2048 bits long? (hash collisions)": mo.vstack(
                        [
                            mo.md(
                                "Morgan has *no* fixed vocabulary, so it can't reserve a slot "
                                "per feature the way MACCS does — it **hashes** each atom "
                                "environment into one of *N* bits. Make *N* too small and "
                                "different substructures collide onto the same bit. Squeeze it "
                                "down to just **8 bits** and watch distinct environments pile "
                                "up on one slot:"
                            ),
                            collision_card,
                            collision_curve_view,
                            collision_slider,
                        ]
                    )
                }
            ),
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(current_mol, mo, mol_valid):
    from fingerprints import classical_explorer as ce

    # Each fingerprint gets its OWN top-level slider variable. marimo only tracks
    # reactivity for mo.ui elements bound directly to a global name - sliders
    # hidden inside a dict/list do NOT trigger downstream re-runs.
    def _slider(key):
        on = ce.on_bits(current_mol, key) if mol_valid else []
        if on:
            return mo.ui.slider(
                start=0, stop=len(on) - 1, value=0,
                label=f"Scrub the {len(on)} bits that are ON",
                full_width=True, show_value=False,
            )
        return mo.ui.slider(start=0, stop=0, value=0, label="(no molecule)")

    topo_slider = _slider("rdkit_topo")
    ap_slider = _slider("atom_pair")
    tt_slider = _slider("top_torsion")
    return ap_slider, ce, topo_slider, tt_slider


@app.cell
def _(ap_slider, ce, current_mol, mo, mol_valid, topo_slider, tt_slider):
    def _fp_tab(key, slider):
        info = ce.FP_INFO[key]
        on = ce.on_bits(current_mol, key) if mol_valid else []
        if not (mol_valid and on):
            body = mo.md("*Select a valid molecule.*")
        else:
            bit = on[min(slider.value, len(on) - 1)]
            hit = ce.bit_hit(current_mol, key, bit)
            svg = ce.highlight_svg(current_mol, hit, width=440, height=320)
            strip = ce.fingerprint_strip_svg(current_mol, key, bit, width=900, height=34)
            card = mo.vstack(
                [
                    mo.md(f"### Bit {hit.bit}"),
                    mo.md(
                        f"Set by **{hit.n_instances}** substructure"
                        f"{'s' if hit.n_instances != 1 else ''} — highlighting "
                        f"**{len(hit.atoms)} atoms**."
                    ),
                ]
            )
            body = mo.vstack(
                [
                    mo.hstack([mo.Html(svg), card], justify="start", gap=2, widths=[3, 2]),
                    mo.md("**Where this bit sits in the full 2048-bit vector:**"),
                    mo.Html(strip),
                    slider,
                ]
            )
        return mo.vstack([mo.md(f"*{info.blurb}*"), body])

    tabbed_fps = mo.ui.tabs(
        {
            ce.FP_INFO["rdkit_topo"].label: _fp_tab("rdkit_topo", topo_slider),
            ce.FP_INFO["atom_pair"].label: _fp_tab("atom_pair", ap_slider),
            ce.FP_INFO["top_torsion"].label: _fp_tab("top_torsion", tt_slider),
        }
    )
    mo.vstack(
        [
            mo.md("### The rest of the RDKit toolbox (topological, atom-pair, torsion)"),
            mo.md(
                "Beyond MACCS's checklist and Morgan's circular environments, "
                "RDKit ships several more \"classical\" fingerprints. They each "
                "encode a different notion of structure — paths, atom pairs at "
                "a distance, torsions — but share Morgan's hashing machinery. "
                "These differences matter: later we'll demonstrate that "
                "certain fingerprints can properly capture activity cliffs for one ADMET endpoint "
                "but can fail on another, and there's no single 'best' fingerprint across all endpoints."
            ),
            tabbed_fps,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### CheMeleon Fingerprint

    Since the **CheMeleon** fingerprint is continuously varying, and the result of a deep neural network, there's no chemical "vocabulary" to scrub through — instead each dimension has a complex relationship to the input molecule. Below we show one way of visualizing what it's doing: pick a dimension and see which atoms affect that dimension for the currently selected molecule. In contrast to the "classical" fingerprints, we have to use partial credit instead of clear attribution, and you'll see that some of the dimensions encode multiple seemingly unrelated parts of the molecule at once.

    To try to make this more digestible, we're limiting to visualizing just the dimensions most strongly sensitive to changes in this molecule's structure; any change to the molecule input will likely have some impact on all 2048 dimensions however. You can adjust where you want to put the "is sensitive" cutoff below to get a sense for what fraction of the fingerprint's dimensions are truly sensitive to this particular molecule.
    """)
    return


@app.cell
def _(mo):
    chemeleon_floor = mo.ui.slider(
        start=0.1,
        stop=0.9,
        step=0.05,
        value=0.5,
        label="Structure-sensitivity floor (fraction of this molecule's most structure-sensitive dimension)",
        show_value=True,
        full_width=True,
    )
    return (chemeleon_floor,)


@app.cell
def _(chemeleon_floor, current_mol, mo, mol_valid, setup_ready):
    from fingerprints import chemeleon_fp as chf

    assert setup_ready
    if mol_valid and current_mol.GetNumAtoms() >= 2:
        _dims = chf.most_active_dims(current_mol, floor_frac=chemeleon_floor.value)
    else:
        _dims = [0]
    chemeleon_dim = mo.ui.slider(
        start=0,
        stop=max(len(_dims) - 1, 0),
        value=0,
        label=f"Scrub {len(_dims)} structure-sensitive dimensions (by index)",
        full_width=True,
        show_value=False,
    )
    chemeleon_dims = _dims
    return chemeleon_dim, chemeleon_dims, chf


@app.cell
def _(
    chemeleon_dim,
    chemeleon_dims,
    chemeleon_floor,
    chf,
    current_mol,
    mo,
    mol_valid,
):
    # Render the current molecule as a heatmap of one learned dimension's
    # per-atom contributions. Exact decomposition: the graph fingerprint is a
    # mean over atoms, so atom i's share of dimension k is H[i,k]/n_atoms.
    if not mol_valid:
        _view = mo.md("*Select a valid molecule.*")
    else:
        _dim = chemeleon_dims[min(chemeleon_dim.value, len(chemeleon_dims) - 1)]
        _svg = chf.heatmap_svg(current_mol, _dim, width=460, height=340)
        _strip = chf.strip_svg(
            current_mol, _dim, active_dims=chemeleon_dims, width=920, height=40
        )
        _strip_legend = mo.md(
            '<span style="color:#7048e8">█ dimension magnitude |fₖ|</span> &nbsp; '
            '<span style="color:#2f9e44">█ structure-sensitive (scrubbable)</span> &nbsp; '
            '<span style="color:#1c7ed6">█ selected dimension</span>'
        )
        _card = mo.md(
            f"### CheMeleon dimension {_dim}\n\n"
            f"<span style='color:#2b8a3e'>● green</span> atoms push this dimension "
            f"up, <span style='color:#c2255c'>● pink</span> push it down."
        )
        _view = mo.vstack(
            [
                mo.hstack([mo.Html(_svg), _card], justify="start", gap=2, widths=[3, 2]),
                mo.md("**Where this dimension sits in the full 2048-long vector:**"),
                mo.Html(_strip),
                _strip_legend,
                chemeleon_floor,
                chemeleon_dim,
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.accordion(
        {
            "📐 What does “structure-sensitivity” mean? (the math)": mo.md(
                r"""
    CheMeleon reads the molecular graph and, after message passing, produces a
    **per-atom hidden vector** $h_i \in \mathbb{R}^{2048}$ for every atom $i$. The
    molecule's fingerprint is just the **mean over atoms**:

    $$ f_k \;=\; \frac{1}{N}\sum_{i=1}^{N} h_{i,k}, \qquad k = 1,\dots,2048 $$

    Because the pooling is a plain mean, each atom's contribution to dimension $k$
    is **exact** (no approximation):

    $$ c_{i,k} \;=\; \frac{h_{i,k}}{N}, \qquad \sum_{i=1}^{N} c_{i,k} = f_k $$

    That is what the molecule heatmap draws for a chosen $k$.

    **Structure-sensitivity** of dimension $k$ is how much that contribution
    *varies across the atoms* of this molecule — we use the spread

    $$ s_k \;=\; \max_i h_{i,k} \;-\; \min_i h_{i,k} $$

    - **Large $s_k$:** different atoms push the dimension very differently, so the
      dimension is reading a **local** structural feature — its heatmap is
      informative (some atoms light up, others don't).
    - **Small $s_k$:** every atom contributes about the same, so the dimension
      encodes something **diffuse/global** and its heatmap would be flat.

    The scrubber offers only the **structure-sensitive** dimensions: those whose
    spread clears a relative floor you set with the knob,

    $$ s_k \;\ge\; \phi \cdot \max_j s_j, \qquad \phi \in [0.1,\,0.9] $$

    i.e. at least a fraction $\phi$ as sensitive as this molecule's most-sensitive
    dimension (default $\phi = 0.5$). Raise $\phi$ for a stricter, smaller set;
    lower it to include more. Either way the *count* varies by molecule (like the
    on-bit count of ECFP/MACCS), and the scrubber steps through them **in index
    order**. The purple strip shows each dimension's magnitude $|f_k|$; green ticks
    mark the structure-sensitive dimensions; blue marks the one you're viewing.

    This is an exact decomposition of the **mean-pool**, but a
    single learned dimension rarely maps to one human-named substructure the way a
    Morgan bit does — read it as “where this dimension looks,” not “what it is.”
    """
            )
        }
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## From fingerprints to similarity

    Once every molecule is a row of numbers, "how alike are these two molecules?" becomes "how alike are these two rows?". That single number, the **fingerprint similarity**, is a regular metric used in cheminformatics work and will come up a lot in the next sections of this notebook.

    - Classical (binary) fingerprints use the **Tanimoto** similarity: of all the bits that are on in *either* molecule, what fraction are on in *both*?

        $$ T(A, B) \;=\; \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert} $$

        1.0 means identical bit patterns and 0.0 means no bits in common. It's rare to see values of this close to zero, since as we saw above some of these bits end up encoding for very common substructures like "a carbon connected to another carbon".
    - Learned (continuous) fingerprints like CheMeleon have no on/off bits, so we use the **cosine similarity** of the two vectors instead (the angle between them). It's a different metric on a different scale, so CheMeleon numbers can't be compared one-to-one with Tanimoto numbers.

    For both types of similarity metrics, more choices are available - these are just two common options.

    Below, molecule 1 of the selected cliff pair is the reference. We compare it with its cliff partner and with an unrelated drug, **celecoxib** (a COX-2 anti-inflammatory). Every fingerprint gives the partner a high score and celecoxib a low one, but the exact numbers differ quite a bit between fingerprints, since each one defines "alike" in its own way. Also pay attention to the first molecules shown in the next section covering activity cliffs in ADMET datasets - in some cases there the fingerprints rate molecules which are clearly different as identical!
    """)
    return


@app.cell
def _(
    alt,
    ctx,
    cv,
    get_cliff_idx,
    get_pair_key,
    mo,
    pd,
    selectors,
    setup_ready,
):
    from rdkit import Chem as _ChemSim
    from rdkit.Chem.Draw import rdMolDraw2D as _d2dSim

    assert setup_ready  # CheMeleon weights present
    _CELECOXIB = "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"
    _tp = ctx.by_key()[get_pair_key()]
    _pair = _tp.cliffs[get_cliff_idx()]

    _svg_ref, _svg_partner = cv.pair_svgs(_pair, width=280, height=210)
    _drawer = _d2dSim.MolDraw2DSVG(280, 210)
    _drawer.drawOptions().addStereoAnnotation = False
    _d2dSim.PrepareAndDrawMolecule(_drawer, _ChemSim.MolFromSmiles(_CELECOXIB))
    _drawer.FinishDrawing()
    _svg_cel = _drawer.GetDrawingText()

    _names = {1: "cliff partner (molecule 2)", 2: "celecoxib (unrelated)"}
    _rows = cv.similarity_to_reference([_pair.smiles_1, _pair.smiles_2, _CELECOXIB])
    _df = pd.DataFrame(
        [
            {
                "fingerprint": f"{r['fingerprint']} ({r['metric']})",
                "compared with": _names[r["other_index"]],
                "similarity": round(r["similarity"], 2),
            }
            for r in _rows
        ]
    )
    _fp_order = list(dict.fromkeys(_df["fingerprint"]))
    _chart = (
        alt.Chart(_df)
        .mark_bar(cornerRadius=2)
        .encode(
            y=alt.Y("fingerprint:N", sort=_fp_order, title=None,
                    axis=alt.Axis(labelLimit=250)),
            yOffset=alt.YOffset("compared with:N", sort=list(_names.values())),
            x=alt.X("similarity:Q", title="similarity to molecule 1",
                    scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "compared with:N",
                sort=list(_names.values()),
                scale=alt.Scale(range=["#4c6ef5", "#adb5bd"]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=["fingerprint:N", "compared with:N", "similarity:Q"],
        )
        .properties(height=320)
    )

    def _card(svg, caption):
        return mo.vstack(
            [mo.Html(svg), mo.md(f"<div style='text-align:center'>{caption}</div>")],
            align="center",
        )

    mo.vstack(
        [
            selectors(),
            mo.hstack(
                [
                    _card(_svg_ref, "Molecule 1 (reference)"),
                    _card(_svg_partner, "Molecule 2 (cliff partner)"),
                    _card(_svg_cel, "Celecoxib (unrelated drug)"),
                ],
                justify="center",
                gap=1,
            ),
            mo.as_html(_chart),
            mo.md(
                "*Orange atoms = the change between the cliff pair. Notice how "
                "much the fingerprints disagree about the quantitative similarities."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Activity cliffs are not a protein-binding-specific issue

    The very beginning of this notebook centered on examining pairs of compounds which exhibit an activity cliff against one protein but don't against a second closely related protein. However, the core concept of the activity cliff generalizes across datasets in chemistry and biology. Let's demonstrate this with **ADMET** (Absorption, Distribution, Metabolism, Excretion & Toxicity) properties, where there's no conveniently related second target, just one measurement per molecule.

    Here's analysis on some ADMET endpoints from the Therapeutics Data Commons and previous OpenADMET competitions. We scanned thru each dataset looking for pairs of molecules that were highly similar, and then collated how much the endpoint measurement changed between that pair to detect how many potential activity cliffs are in the dataset. Because different fingerprints encode structures differently, we run this across all the fingerprints mentioned so far to try and see how much difference this makes in the aggregate.
    """)
    return


@app.cell
def _(mo):
    from fingerprints import admet_view as adm

    if adm.has_data():
        _eps = adm.endpoints()
        admet_choice = mo.ui.dropdown(
            options=_eps, value=_eps[0], label="ADMET endpoint"
        )
    else:
        admet_choice = mo.ui.dropdown(options={"(none)": ""}, value="(none)")
    admet_choice
    return adm, admet_choice


@app.cell
def _(adm, admet_choice, alt, mo, pd):
    from rdkit import Chem as _Chem
    from rdkit.Chem.Draw import rdMolDraw2D as _d2d

    # The ADMET census, split per fingerprint: for EACH fingerprint we show its
    # own gap histogram (which pairs it calls "similar", and how cliffy they
    # are) and, directly beneath, the single sharpest cliff THAT fingerprint
    # would wave through as similar. The columns make the point concrete: every
    # fingerprint draws "similar" differently, yet each has its own bad cliff.
    def _pair_svg(smi, w=132, h=100):
        m = _Chem.MolFromSmiles(smi)
        d = _d2d.MolDraw2DSVG(w, h)
        d.drawOptions().addStereoAnnotation = False
        if m is not None:
            _d2d.PrepareAndDrawMolecule(d, m)
        d.FinishDrawing()
        return d.GetDrawingText()

    if not adm.has_data() or admet_choice.value not in adm.endpoints():
        admet_census_view = mo.md(
            "*ADMET census not precomputed — run "
            "`python -m fingerprints.analyses.admet`.*"
        ).callout(kind="info")
    else:
        _ep = admet_choice.value
        _mt = adm.meta(_ep)
        _s = adm.smoothness(_ep)
        _unit = _mt["unit"]

        def _label_gap(h):
            return f"{h['lo']:.1f}\u2013{h['hi']:.1f}" if h["hi"] < 20 else f"{h['lo']:.1f}+"

        def _classify(h):
            if h["hi"] <= _s["flat_gap"]:
                return "flat"
            if h["lo"] >= _s["cliff_gap"]:
                return "cliff"
            return "middle"

        def _hist_chart(fp_entry):
            _rows = [
                {"gap": _label_gap(h), "lo": h["lo"], "count": h["count"],
                 "kind": _classify(h)}
                for h in fp_entry["gap_hist"]
            ]
            return (
                alt.Chart(pd.DataFrame(_rows))
                .mark_bar()
                .encode(
                    x=alt.X("gap:N", sort=alt.SortField("lo"), title=None,
                            axis=alt.Axis(labelAngle=-45, labelFontSize=7)),
                    y=alt.Y("count:Q", title="similar pairs"),
                    color=alt.Color(
                        "kind:N",
                        scale=alt.Scale(
                            domain=["flat", "middle", "cliff"],
                            range=["#2b8a3e", "#adb5bd", "#e03131"],
                        ),
                        legend=None,
                    ),
                    tooltip=["gap:N", "count:Q"],
                )
                .properties(width=150, height=130)
            )

        def _cliff_block(fp_entry):
            c = fp_entry.get("top_cliff")
            if not c:
                return mo.md(
                    "<div style='text-align:center;color:#adb5bd;font-size:11px'>"
                    "no cliff among its similar pairs</div>"
                )
            # Stack the pair vertically (two rows) - side by side they
            # overflow the narrow per-fingerprint column.
            return mo.vstack(
                [
                    mo.Html(_pair_svg(c["smiles_1"])),
                    mo.Html(_pair_svg(c["smiles_2"])),
                    mo.md(
                        f"<div style='text-align:center;font-size:10px'>"
                        f"similarity <b>{c['tanimoto']:.2f}</b> · {_unit} "
                        f"{c['act_1']:.1f} vs {c['act_2']:.1f}<br>"
                        f"<span style='color:#e03131'>gap {c['gap']:.1f}"
                        f"</span></div>"
                    ),
                ],
                align="center",
                gap=0.25,
            )

        _per_fp = _s.get("per_fp", {})
        if _per_fp:
            _cols = []
            for _fv in _per_fp.values():
                _cliff_pct_fp = round(_fv["frac_cliff"] * 100, 1)
                _cols.append(
                    mo.vstack(
                        [
                            mo.md(
                                f"<div style='text-align:center;font-size:12px'>"
                                f"<b>{_fv['label']}</b><br>"
                                f"<span style='color:#868e96'>"
                                f"{_fv['n_similar_pairs']:,} similar · "
                                f"{_cliff_pct_fp}% cliffs</span></div>"
                            ),
                            mo.as_html(_hist_chart(_fv)),
                            _cliff_block(_fv),
                        ]
                    )
                )
            _grid = mo.hstack(_cols, widths=[1] * len(_cols), gap=1)
        else:
            _grid = mo.md("")

        _flat_pct = round(_s["frac_flat"] * 100)
        _cliff_pct = round(_s["frac_cliff"] * 100, 1)
        _msg = mo.md(
            f"**{_ep}** ({_unit}) — **{_mt['n_total']:,}** unique structures "
            f"(salts stripped, deduplicated), scaffold-split "
            f"{_mt['n_train']:,} train / {_mt['n_test']:,} test.\n\n"
            f"Among the **{_s['n_similar_pairs']:,}** pairs **Morgan** calls "
            f"*similar* (Tanimoto ≥ {_s['sim_threshold']:.1f}), **{_flat_pct}%** "
            f"are flat (within {_s['flat_gap']:.1f} log unit) but **{_cliff_pct}%** "
            f"are outright **cliffs** (> {_s['cliff_gap']:.1f} log units — more than "
            f"~30× apart). A single-atom or chain-length "
            f"change can move solubility by orders of magnitude while the "
            f"fingerprint barely changes - or sometimes not at all. **Each column repeats the census under a "
            f"different fingerprint's similarity, with that fingerprint's own "
            f"sharpest cliff drawn beneath** — the exact 'similar' set shifts, "
            f"but every fingerprint has a stubborn red cliff tail and a real pair "
            f"to show for it."
        ).callout(kind="danger" if _s["frac_cliff"] > 0.2 else "warn")
        admet_census_view = mo.vstack(
            [mo.md(f"*{_mt['blurb']}*"), _msg, _grid]
        )
    admet_census_view
    return


@app.cell
def _(adm, alt, mo, pd):
    from fingerprints import knn_view as _knn

    # The payoff comparison across ALL THREE endpoints, split by fingerprint:
    # cliffiness varies wildly by target (you can't know in advance), AND which
    # fingerprint happens to minimise cliffs flips per target - so you can't
    # know in advance which encoding will serve a new endpoint either.
    if not adm.has_data():
        admet_compare_view = mo.md("")
    else:
        _palette = _knn.fp_colors()  # shared with the kNN charts
        _rows = []
        for _e in adm.endpoints():
            _unit = adm.meta(_e)["unit"]
            _elabel = f"{_e}\n({_unit})"
            for _fv in adm.per_fp(_e).values():
                _rows.append(
                    {
                        "endpoint": _elabel,
                        "fingerprint": _fv["label"],
                        "cliff_pct": round(_fv["frac_cliff"] * 100, 1),
                    }
                )
        _df = pd.DataFrame(_rows)
        _ep_order = [f"{_e}\n({adm.meta(_e)['unit']})" for _e in adm.endpoints()]
        _fp_order = [_fv["label"] for _fv in adm.per_fp(adm.endpoints()[0]).values()]
        _chart = (
            alt.Chart(_df)
            .mark_bar()
            .encode(
                x=alt.X("endpoint:N", title=None, sort=_ep_order,
                        axis=alt.Axis(labelAngle=0, labelLimit=200)),
                xOffset=alt.XOffset("fingerprint:N", sort=_fp_order),
                y=alt.Y("cliff_pct:Q",
                        title="% of similar pairs that are cliffs",
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.Color(
                    "fingerprint:N",
                    scale=alt.Scale(
                        domain=_fp_order,
                        range=[_palette.get(lbl, "#868e96") for lbl in _fp_order],
                    ),
                    legend=alt.Legend(title=None, orient="bottom", columns=5),
                ),
                tooltip=["endpoint:N", "fingerprint:N", "cliff_pct:Q"],
            )
            .properties(height=280)
        )
        admet_compare_view = mo.vstack(
            [
                mo.md(
                    "If we run the same census across those ADMET endpoints — three public "
                    "TDC benchmarks **plus the OpenADMET ExpansionRx LogD and "
                    "solubility sets** — and break them out by fingerprint, we can see that the situation is unfortunately not solved by just picking a 'better' fingerprint:"
                ),
                mo.as_html(_chart),
                mo.md(
                    "Three things are only knowable after you have the data:\n\n"
                    "1. **'Cliffiness' changes for each dataset.** Aqueous "
                    "solubility (AqSolDB) has very many activity cliffs, lipophilicity "
                    "far less, and Caco-2 permeability barely any. There is some intuitive sense here - Solubility "
                    "hinges on crystal packing and H-bonding a single atom can "
                    "shatter; logD is a smoother bulk average - but telling ourselves we can understand this after-the-fact does not mean we can reliably predict it for new dataset in the future.\n\n"
                    "2. **No fingerprint is safest everywhere.** On solubility "
                    "**atom-pair** often flags fewer cliffs than Morgan or MACCS "
                    "— its distance-based similarity happens to align with what "
                    "drives solubility — but that lead can disappear on other "
                    "endpoints.\n\n"
                    "3. **Activity cliffs can be a symptom of less standardized data** OpenADMET's "
                    "ExpansionRx LogD and solubility show a *lower* cliff rate "
                    "than the aggregated public AqSolDB — consistent with a "
                    "single-platform, controlled-condition measurement (less "
                    "inter-lab variation in measurements). The presence of cliffs is a "
                    "property of the data as well as the chemistry. A model trained on data from another lab may not play well with data from your lab.\n\n"
                ).callout(kind="info"),
                mo.md("---"),
            ]
        )
    admet_compare_view
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How activity cliffs cause models to lose accuracy

    We've seen so far that the presence activity cliffs can correlate to poor model performance on some specific chemistries and applications, and that fingerprint's notions of similiarities map differently well to different datasets. Next, lets dig into these ideas to try and understand why statistical models cannot reliably capture activity cliffs, regardless of fingerprint.

    To explore that we simplify the model down to a minimal example. Just **k-nearest-neighbours**
    on the fingerprint: a molecule's prediction is just the average value of
    its k most-similar neighbours where "similar" is Tanimoto Similarity of just the fingerprint. Nothing is
    learned on top — the fingerprint's notion of similarity *is* the entire
    model. So whatever kNN structurally cannot do is the fingerprint's own
    limitation, with no fancy decision boundaries to obfuscate the situation.
    """)
    return


@app.cell
def _(mo, selectors):
    # Synced selector mirror: switch target pair / molecule pair right
    # here without scrolling back to the top (same global elements).
    mo.vstack([mo.md('**Pick the target pair / molecule pair for this kNN demo:**'), selectors()])
    return


@app.cell
def _(alt, ctx, get_cliff_idx, get_pair_key, k_slider, knn, mo, pd):
    # kNN cliff analysis, precomputed per endpoint (all curated pairs).
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    _k = k_slider.value

    _eps = knn.endpoints() if knn.has_data() else []
    _cliff_ep = _tp.cliffs[_idx].cliff_on if _tp.cliffs else None
    _pair = knn.cliff_pair(_cliff_ep, _idx) if _cliff_ep in _eps else None

    if _pair is None:
        _view = mo.md(
            "*The kNN cliff analysis wasn't precomputed for this pair yet — "
            "pick another pair above. "
            "(Run `python -m fingerprints.analyses.knn` to add it.)*"
        ).callout(kind="info")
    else:
        _ep = _pair["cliff_on"]
        _m1, _m2 = _pair["mol1"], _pair["mol2"]
        _meta = knn.endpoint_meta(_ep)
        _best = knn.best_k(_ep)

        # (a) The k tradeoff, overlaid for EVERY fingerprint's similarity - the
        # accuracy/k curve (and where it peaks) is itself a choice of
        # fingerprint. Each line is one fingerprint's held-out kNN accuracy.
        # One fixed palette is shared with the per-fingerprint cliff scatter (b).
        _fp_palette = knn.fp_colors()
        _by_fp = knn.k_curves_by_fp(_ep)
        if _by_fp:
            _rows = []
            for _fk, _fv in _by_fp.items():
                for _pt in _fv["k_curve"]:
                    _rows.append(
                        {"k": _pt["k"], "r2": _pt["r2"], "fingerprint": _fv["label"]}
                    )
            _curve = pd.DataFrame(_rows)
            _fp_labels = [_fv["label"] for _fv in _by_fp.values()]
            _color_enc = alt.Color(
                "fingerprint:N",
                scale=alt.Scale(
                    domain=_fp_labels,
                    range=[_fp_palette.get(lbl, "#868e96") for lbl in _fp_labels],
                ),
                legend=alt.Legend(title=None, orient="bottom", columns=2),
            )
            _line = (
                alt.Chart(_curve)
                .mark_line(point=True)
                .encode(
                    x=alt.X("k:Q", title="neighbourhood size k",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("r2:Q", title="held-out R² (accuracy)"),
                    color=_color_enc,
                    tooltip=["fingerprint:N", "k:Q",
                             alt.Tooltip("r2:Q", format=".3f")],
                )
            )
        else:
            _curve = pd.DataFrame(knn.k_curve(_ep))
            _line = (
                alt.Chart(_curve)
                .mark_line(point=True, color="#4c6ef5")
                .encode(
                    x=alt.X("k:Q", title="neighbourhood size k",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("r2:Q", title="held-out R² (accuracy)"),
                    tooltip=["k:Q", alt.Tooltip("r2:Q", format=".3f")],
                )
            )
        _rule = (
            alt.Chart(pd.DataFrame({"k": [_k]}))
            .mark_rule(color="#e8820c", strokeWidth=2)
            .encode(x="k:Q")
        )
        _acc_top = (_line + _rule).properties(height=170, width=300)

        # (a2) The SAME k axis, but RMSE on the *cliff molecules only* (every
        # cliff pair in this endpoint, not just the highlighted one). Stacked
        # directly under the global-accuracy panel so the one orange current-k
        # rule reads across both: as you slide k, watch global accuracy rise to
        # a peak while the cliff error stays stubbornly flat and high - no
        # neighbourhood size rescues the cliffs.
        _cliff_rmse = knn.cliff_rmse_by_fp(_ep)
        _n_cliff = knn.n_cliff_pairs(_ep)
        if _cliff_rmse:
            _crows = []
            for _fv in _cliff_rmse.values():
                for _pt in _fv["rmse_curve"]:
                    _crows.append(
                        {"k": _pt["k"], "rmse": _pt["rmse"],
                         "fingerprint": _fv["label"]}
                    )
            _cdf = pd.DataFrame(_crows)
            _cline = (
                alt.Chart(_cdf)
                .mark_line(point=True, strokeDash=[4, 2])
                .encode(
                    x=alt.X("k:Q", title="neighbourhood size k",
                            scale=alt.Scale(type="log")),
                    y=alt.Y("rmse:Q", title="cliff-pair RMSE (log units)"),
                    color=_color_enc if _by_fp else alt.value("#4c6ef5"),
                    tooltip=["fingerprint:N", "k:Q",
                             alt.Tooltip("rmse:Q", format=".3f")],
                )
            )
            _acc_bottom = (_cline + _rule).properties(height=170, width=300)
            _acc_chart = alt.vconcat(_acc_top, _acc_bottom).resolve_scale(
                color="shared"
            )
        else:
            _acc_chart = _acc_top

        # (b) The cliff itself, "scatter-where-a-box-would-be", now split PER
        # FINGERPRINT: a kNN prediction is the MEAN of the k nearest neighbours,
        # and each fingerprint picks *different* neighbours. So within each
        # molecule we lay out one sub-column per fingerprint (matched to the
        # colours in (a)): faint dots = that fingerprint's neighbour cloud, a
        # bold tick = its prediction. The single black diamond is the molecule's
        # TRUE activity (fingerprint-independent ground truth). Whatever the
        # fingerprint, every prediction sits far from the diamond on at least
        # one molecule - the cliff none of them resolve.
        _p1 = knn.pred_at_k(_m1, _k)
        _p2 = knn.pred_at_k(_m2, _k)
        _pred_gap = abs(_p1 - _p2)
        _true_gap = abs(_m1["true"] - _m2["true"])

        _cliff_fps = knn.cliff_fps(_m1) or {"morgan": "Morgan (ECFP4)"}
        _fp_keys = list(_cliff_fps)
        _fp_lbls = [_cliff_fps[k] for k in _fp_keys]
        _cloud_rows, _pred_rows = [], []
        for _m, _name in [(_m1, "molecule 1"), (_m2, "molecule 2")]:
            for _fk in _fp_keys:
                _lbl = _cliff_fps[_fk]
                for _a in knn.neighbor_acts_at_k(_m, _k, fp=_fk):
                    _cloud_rows.append({"mol": _name, "pKi": _a, "fingerprint": _lbl})
                _pred_rows.append(
                    {"mol": _name, "pKi": knn.pred_at_k(_m, _k, fp=_fk),
                     "fingerprint": _lbl}
                )
        _cloud = pd.DataFrame(_cloud_rows)
        _preds = pd.DataFrame(_pred_rows)
        _truth = pd.DataFrame(
            [
                {"mol": "molecule 1", "pKi": _m1["true"]},
                {"mol": "molecule 2", "pKi": _m2["true"]},
            ]
        )
        _y = alt.Y("pKi:Q", title=f"{_ep} pKi", scale=alt.Scale(zero=False))
        _fp_color = alt.Color(
            "fingerprint:N",
            scale=alt.Scale(
                domain=_fp_lbls,
                range=[_fp_palette.get(lbl, "#868e96") for lbl in _fp_lbls],
            ),
            legend=None,
        )
        _fp_xoffset = alt.XOffset("fingerprint:N", scale=alt.Scale(domain=_fp_lbls))
        # each fingerprint's neighbour cloud (jittered within its sub-column)
        _dots = (
            alt.Chart(_cloud)
            .mark_circle(size=22, opacity=0.28)
            .encode(
                x=alt.X("mol:N", title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=_fp_xoffset,
                y=_y,
                color=_fp_color,
                tooltip=["fingerprint:N", alt.Tooltip("pKi:Q", format=".2f")],
            )
        )
        # each fingerprint's kNN prediction (bold tick where a box median sits)
        _pred_tick = (
            alt.Chart(_preds)
            .mark_tick(thickness=3, size=16)
            .encode(
                x=alt.X("mol:N", title=None),
                xOffset=_fp_xoffset,
                y=_y,
                color=_fp_color,
                tooltip=["fingerprint:N",
                         alt.Tooltip("pKi:Q", format=".2f", title="kNN pred")],
            )
        )
        # the molecule's true activity: one black diamond, fingerprint-agnostic
        _true_pt = (
            alt.Chart(_truth)
            .mark_point(shape="diamond", size=170, filled=True, color="#212529")
            .encode(
                x=alt.X("mol:N", title=None),
                y=_y,
                tooltip=[alt.Tooltip("pKi:Q", format=".2f", title="true pKi")],
            )
        )
        _cliff_chart = (_dots + _pred_tick + _true_pt).properties(
            width=280, height=230
        )

        # Honest, per-pair statement: the gap kNN opens at each k. At tiny k it
        # can occasionally match (or exceed) the true gap by copying a single
        # near-identical neighbour, but that collapses as soon as the
        # neighbourhood grows; at the accuracy-optimal k the gap is small.
        _gaps_by_k = {
            a["k"]: abs(a["pred"] - b["pred"])
            for a, b in zip(_m1["pred_by_k"], _m2["pred_by_k"])
        }
        _max_gap = max(_gaps_by_k.values())
        _argmax_k = max(_gaps_by_k, key=_gaps_by_k.get)
        _gap_at_best = _gaps_by_k.get(_best["k"], _max_gap)
        # Does any k actually resolve the cliff (reach most of the true gap)?
        _resolves = _max_gap >= 0.8 * _true_gap
        if _resolves:
            _range_clause = (
                f"the widest gap it ever opens is **{_max_gap:.2f}** — but only at "
                f"**k={_argmax_k}**, where the prediction is just *copying a single "
                f"near-identical neighbour*; the moment the neighbourhood grows the "
                f"gap collapses (down to **{_gap_at_best:.2f}** at the accuracy-optimal "
                f"**k={_best['k']}**)"
            )
        else:
            _range_clause = (
                f"the widest gap it ever opens for this pair is only "
                f"**{_max_gap:.2f}** — no neighbourhood size comes close to the true "
                f"**{_true_gap:.2f}**"
            )
        _verdict = mo.md(
            f"At **k={_k}**, kNN predicts these two molecules **{_p1:.2f}** and "
            f"**{_p2:.2f}** — a gap of just **{_pred_gap:.2f}**, though the real gap "
            f"is **{_true_gap:.2f}**. "
            f"**Slide k across its whole range:** {_range_clause}. Meanwhile global "
            f"accuracy peaks near **k={_best['k']}** (R² {_best['r2']:.2f}); the k "
            f"that fits the dataset best still can't see this cliff."
        ).callout(kind="warn")

        _view = mo.vstack(
            [
                mo.md(
                    f"**{_ep}** — {_meta['n_total']} molecules. These two panels "
                    f"summarise the *endpoint this cliff sits on* (**{_ep}**), not the "
                    f"single pair — so picking a different molecule pair that happens to "
                    f"be a cliff on the other target will swap which endpoint you're "
                    f"looking at here. The plots share the "
                    "same k axis and the same orange current-k rule. **Top:** "
                    "global held-out accuracy (R²) per fingerprint — it rises to a "
                    "peak at some k. **Bottom:** RMSE on the **cliff pairs only** "
                    f"(all {_n_cliff} in this endpoint) — usually flat or worse with increasing k. "
                    "The same parameter in the model - neighbourhood size - that maximises average accuracy usually makes predictions over cliffs worse, because it washes out the contribution of small structural changes. You can see this very clearly in the plot on the right, where increasing neighborhood size causes the prediction on molecule 1 vs molecule to converge."
                ),
                mo.hstack(
                    [
                        mo.vstack([mo.md(f"**{_ep}: accuracy (top) vs cliff error (bottom) vs k**"), mo.as_html(_acc_chart)]),
                        mo.vstack([
                            mo.md(
                                "**This cliff pair at k** — one coloured "
                                "sub-column per fingerprint (matched to the lines "
                                "at left): <span style='color:#868e96'>faint dots = "
                                "its neighbour cloud, tick = its prediction</span>. "
                                "<span style='color:#212529'>◆ black = true "
                                "activity</span>."
                            ),
                            mo.as_html(_cliff_chart),
                        ]),
                    ],
                    widths=[1, 1], gap=2,
                ),
                k_slider,
                _verdict,
            ]
        )
    _view
    return


@app.cell
def _(mo):
    # Resample button for the flat-pair gallery below (a fresh random draw of
    # 'similar structure, similar activity' pairs - the majority the assumption
    # gets right). value increments on each click -> reactive reseed.
    resample_flat = mo.ui.button(
        label="🎲 Sample different flat pairs", value=0, on_click=lambda v: v + 1
    )
    return (resample_flat,)


@app.cell
def _(alt, ctx, get_cliff_idx, get_pair_key, knn, mo, pd, resample_flat):
    # Why kNN (and any structure-only model) behaves this way: the smoothness
    # census. Among structurally similar pairs the overwhelming majority really
    # are flat, so 'similar fingerprint -> similar prediction' is the RIGHT
    # thing to learn for the dataset as a whole - and the rare cliff is the
    # price. Placed right after the kNN demo so it explains what we just saw.
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    _cliff_ep = _tp.cliffs[_idx].cliff_on if _tp.cliffs else None
    _eps = knn.endpoints() if knn.has_data() else []

    if _cliff_ep not in _eps:
        _view = mo.md(
            "*This census wasn't precomputed for this pair yet — "
            "pick another pair above.*"
        ).callout(kind="info")
    else:
        _s = knn.smoothness(_cliff_ep)
        _hist = pd.DataFrame(
            [
                {
                    "gap": f"{h['lo']:.1f}–{h['hi']:.1f}" if h["hi"] < 10 else f"{h['lo']:.1f}+",
                    "lo": h["lo"],
                    "count": h["count"],
                    "kind": (
                        "flat" if h["hi"] <= _s["flat_gap"]
                        else "cliff" if h["lo"] >= _s["cliff_gap"]
                        else "middle"
                    ),
                }
                for h in _s["gap_hist"]
            ]
        )
        _chart = (
            alt.Chart(_hist)
            .mark_bar()
            .encode(
                x=alt.X("gap:N", sort=alt.SortField("lo"),
                        title="activity gap between the pair (|ΔpKi|, log units)"),
                y=alt.Y("count:Q", title="number of similar pairs"),
                color=alt.Color(
                    "kind:N",
                    scale=alt.Scale(
                        domain=["flat", "middle", "cliff"],
                        range=["#2b8a3e", "#adb5bd", "#e03131"],
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                tooltip=["gap:N", "count:Q"],
            )
            .properties(height=220)
        )
        _flat_pct = round(_s["frac_flat"] * 100)
        _cliff_pct = _s["frac_cliff"] * 100
        _ratio = round(_s["frac_flat"] / max(_s["frac_cliff"], 1e-9))
        _intro = mo.md(
            "### Why kNN does this: it's learning the right rule for the dataset\n\n"
            "The kNN model above isn't broken. Averaging over similar neighbours "
            "means it has learned *similar fingerprint → similar property*, and for "
            f"the {_cliff_ep} dataset as a whole that is the correct rule. It's what "
            "pushes R² up as k grows. But that same rule makes the model blind to "
            "the cliff pair: the two molecules share almost all their neighbours, "
            "so they get almost the same prediction. The histogram below shows how "
            "often the rule holds."
        )
        _msg = mo.md(
            f" Take every pair of {_cliff_ep} "
            f"molecules that a fingerprint calls *similar* (Tanimoto ≥ "
            f"{_s['sim_threshold']:.1f}): **{_s['n_similar_pairs']:,}** pairs. "
            f"**{_flat_pct}%** of them are **flat** (activity within "
            f"{_s['flat_gap']:.0f} log unit) and only **{_cliff_pct:.1f}%** are "
            f"true cliffs — roughly **{_ratio}:1**.\n\n"
            f"So 'similar structure → similar activity' is *right the vast majority "
            f"of the time*. Any model that predicts from structure alone is "
            f"rewarded for learning it — and a model that instead predicted big "
            f"activity jumps for near-identical structures would be wrong on the flat "
            f"{_flat_pct}% to catch the cliffy {_cliff_pct:.1f}%. "
            "Shrinking k to chase the cliff (the bottom panel above) costs global "
            "accuracy, and growing k to win global accuracy smooths the cliff "
            "away. The model can't get one without giving up the other. "
            "This results in an unresolvable tension between specific and global accuracy for whatever fingerprint-based model we pick. **A cliff is where reality "
            f"breaks the very assumption that makes the fingerprint useful.** No "
            f"amount of model cleverness recovers information the structure encoding "
            f"never contained — which is why activity cliffs are a well-documented "
            f"challenge in cheminformatics, not a modelling failure."
        ).callout(kind="danger")

        # A gallery of the flat majority: similar structures whose activity
        # really is similar (resampled on the button click).
        from rdkit import Chem as _Chem
        from rdkit.Chem.Draw import rdMolDraw2D as _d2d

        def _pair_svg(s1, s2, w=150, h=110):
            def one(s):
                m = _Chem.MolFromSmiles(s)
                d = _d2d.MolDraw2DSVG(w, h)
                d.drawOptions().addStereoAnnotation = False
                if m is not None:
                    _d2d.PrepareAndDrawMolecule(d, m)
                d.FinishDrawing()
                return d.GetDrawingText()
            return one(s1), one(s2)

        def _flat_card(fp):
            a, b = _pair_svg(fp["smiles_1"], fp["smiles_2"])
            gap = abs(fp["act_1"] - fp["act_2"])
            return mo.vstack(
                [
                    mo.hstack([mo.Html(a), mo.Html(b)], justify="center", gap=0.5),
                    mo.md(
                        f"<div style='text-align:center;font-size:12px'>"
                        f"Tanimoto <b>{fp['tanimoto']:.2f}</b> · pKi "
                        f"{fp['act_1']:.1f} vs {fp['act_2']:.1f} — "
                        f"<span style='color:#2b8a3e'>gap {gap:.2f} (flat ✓)</span></div>"
                    ),
                ]
            )

        _flats = knn.sample_flat_pairs(_cliff_ep, 3, seed=resample_flat.value)
        _gallery = mo.vstack(
            [
                mo.md(
                    "The flat majority — **similar structure, similar activity** is almost always right. "
                    "These are the pairs the kNN model is getting right:"
                ),
                mo.hstack(
                    [_flat_card(fp) for fp in _flats] or [mo.md("*(no pairs)*")],
                    widths=[1] * max(len(_flats), 1), gap=1,
                ),
                resample_flat,
            ]
        )
        _view = mo.vstack([_intro, mo.as_html(_chart), _msg, _gallery])
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    # The only knob kNN has is neighbourhood size k - the honest analog of
    # "move the decision boundary". Top-level so the section reacts to it.
    from fingerprints import knn_view as knn

    _grid = knn.k_grid() if knn.has_data() else [1, 5, 20]
    k_slider = mo.ui.slider(
        steps=_grid,
        value=_grid[min(3, len(_grid) - 1)],
        label="Neighbourhood size k",
        show_value=True,
        full_width=True,
    )
    return k_slider, knn


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### Binary fingerprints literally can't count

    Before we look at some of the ways that scientist try to overcome activity cliffs, one concrete failure mode is worth isolating because it makes the "the information we need just isn't in the fingerprint" problem quite visible. Many
    ADMET properties are accumulating — solubility, for instance: tack on
    another –CH₂– and the desolvation cost keeps climbing. To predict such a
    property you need to know *how many* hydrophobic units a molecule has. But a
    standard binary fingerprint only records whether a substructure is
    present, not how many times. This makes it quite hard to calculate accumulating properties from this input!

    To demonstrate the effect of switching to count-fingerprints, instead of binary fingerprints, we fit several encodings on AqSolDB and compare them on two targets:

    - a pure accumulating property we control exactly — *heavy-atom count*
    - **experimental solubility**

    The fingerprints span both families from earlier: **binary Morgan** (the usual
    fixed fingerprint), **count Morgan** (same bits, but each slot holds a count
    so a linear head can literally add them up), and
    the **CheMeleon** fingerprint. CheMeleon is interesting here: it's pre-trained for molecular
    property prediction so you might expect it to ace the heavy-atom count task- it does get close but not quite perfect! All heads are fairly tuned (RidgeCV).

    A single train/test split gives one number per model and no sense of how much that number would move on a different split. So instead every model is scored with **5×5 scaffold cross-validation**: the molecules are dealt into 5 folds by Bemis–Murcko scaffold (no scaffold is ever in both train and test), each fold takes a turn as the test set, and the whole thing is repeated 5 times with a different shuffle. That gives **25 scores per model, on the same 25 splits for every model**, which we compare with **Tukey's Honestly Significant Difference (HSD)** test:

    - the model with the best mean (highest R² or lowest RMSE) is shown in **blue**,
    - models statistically equivalent to it are **grey**,
    - models significantly different from it are **red**,
    - the bars are confidence intervals adjusted for comparing many models at once, and the dashed lines mark the best model's interval. Two bars that don't overlap are significantly different.
    """)
    return


@app.cell
def _(mo):
    acc_metric = mo.ui.radio(
        options={"R² (higher is better)": "r2", "RMSE (lower is better)": "rmse"},
        value="R² (higher is better)",
        label="Metric",
        inline=True,
    )
    return (acc_metric,)


@app.cell
def _(acc_metric, alt, mo, pd, setup_ready):
    from fingerprints import accumulation as acc

    assert setup_ready

    # 5x5 scaffold CV results ship precomputed under data/ (instant); only
    # fits live if the cache is missing, e.g. during a from-scratch recompute.
    if acc.has_cv():
        _cv = acc.cv_results()
    else:
        with mo.status.spinner(
            title="Running 5×5 scaffold CV on AqSolDB (live, several minutes)…"
        ):
            _cv = acc.cv_results()

    _metric = acc_metric.value
    _mlabel = "R²" if _metric == "r2" else "RMSE"
    _status_color = {"best": "#1c7ed6", "equivalent": "#868e96", "different": "#e03131"}
    _units = {"heavy-atom count": "atoms", "aqueous solubility": "logS"}

    def _tukey_chart(target):
        res = acc.tukey(target, _metric)
        df = pd.DataFrame(
            [
                {
                    "model": r.label,
                    "mean": round(r.mean, 3),
                    "lo": round(r.lo, 3),
                    "hi": round(r.hi, 3),
                    "status": r.status,
                    "p vs best": round(r.p_vs_best, 4),
                }
                for r in res.rows
            ]
        )
        order = [r.label for r in res.rows]
        best = next(r for r in res.rows if r.status == "best")
        color = alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=list(_status_color), range=list(_status_color.values())
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        )
        y = alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        tip = ["model:N", "mean:Q", "lo:Q", "hi:Q", "status:N", "p vs best:Q"]
        bars = (
            alt.Chart(df)
            .mark_rule(strokeWidth=3)
            .encode(
                y=y,
                x=alt.X("lo:Q", title=f"{_mlabel} ({_units[target]})"
                        if _metric == "rmse" else _mlabel),
                x2="hi:Q",
                color=color,
                tooltip=tip,
            )
        )
        dots = (
            alt.Chart(df)
            .mark_point(filled=True, size=70)
            .encode(y=y, x="mean:Q", color=color, tooltip=tip)
        )
        guides = (
            alt.Chart(pd.DataFrame({"x": [best.lo, best.hi]}))
            .mark_rule(strokeDash=[4, 4], color="#868e96")
            .encode(x="x:Q")
        )
        return (guides + bars + dots).properties(
            width=330,
            height=170,
            title=alt.Title(
                f"{target} — {_mlabel}",
                subtitle=f"ANOVA p = {res.anova_p:.1e} · {res.n_splits} splits per model",
            ),
        )

    # Paired view: binary vs count Morgan, one line per CV split. Because every
    # model saw the same 25 splits, the honest comparison is split-by-split.
    def _paired_chart(target):
        vals = _cv["targets"][target][_metric]
        rows = []
        for key, name in [("binary_linear", "binary"), ("count_linear", "count")]:
            for i, v in enumerate(vals[key]):
                rows.append({"split": i, "encoding": name, _mlabel: round(v, 3)})
        df = pd.DataFrame(rows)
        p = acc.paired_p(target, "binary_linear", "count_linear", _metric)
        base = alt.Chart(df).encode(
            x=alt.X("encoding:N", title=None, sort=["binary", "count"],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{_mlabel}:Q", scale=alt.Scale(zero=False)),
        )
        lines = base.mark_line(color="#adb5bd", opacity=0.6).encode(detail="split:N")
        pts = base.mark_point(filled=True, size=40).encode(
            color=alt.Color(
                "encoding:N",
                scale=alt.Scale(domain=["binary", "count"], range=["#e03131", "#2b8a3e"]),
                legend=None,
            ),
            tooltip=["split:N", "encoding:N", f"{_mlabel}:Q"],
        )
        return (lines + pts).properties(
            width=150,
            height=170,
            title=alt.Title(
                "binary vs count Morgan",
                subtitle=f"paired t-test p = {p:.1e}",
            ),
        )

    _charts = alt.vconcat(
        *[
            alt.hconcat(_tukey_chart(t), _paired_chart(t))
            for t in acc.target_labels()
        ]
    ).resolve_scale(color="independent")

    # Numbers for the prose, always R2 means so the text reads the same
    # whichever metric is plotted.
    def _mean(target, key, metric="r2"):
        v = _cv["targets"][target][metric][key]
        return sum(v) / len(v)

    def _status(target, key, metric):
        return next(r.status for r in acc.tukey(target, metric).rows if r.key == key)

    _h, _s = "heavy-atom count", "aqueous solubility"
    _che_h_r2 = _status(_h, "chemeleon_linear", "r2")
    _che_h_rmse = _status(_h, "chemeleon_linear", "rmse")
    _che_note = (
        "On R² alone Tukey can't separate it from the count fingerprint, because "
        "a single badly-behaved fold drags its R² far down and widens its interval; "
        "on RMSE the gap to the count fingerprint is clearly significant."
        if _che_h_r2 != "different" and _che_h_rmse == "different"
        else ""
    )
    _p_sol = acc.paired_p(_s, "binary_linear", "count_linear")

    _verdict = mo.md(
        f"On the heavy-atom count dataset, the story is clear. A **count** "
        f"fingerprint + a plain linear model is essentially perfect (mean **R² "
        f"{_mean(_h, 'count_linear'):.3f}**) on every one of the 25 splits; it just "
        f"sums the bits, which is the target. The binary fingerprint + the same "
        f"linear model averages **R² {_mean(_h, 'binary_linear'):.2f}**: once you "
        f"binarise, you can't tell one –CH₂– from six.\n\n"
        f"The CheMeleon fingerprint does markedly better than binary (mean **R² "
        f"{_mean(_h, 'chemeleon_linear'):.2f}**). Even though it mean-pools over "
        f"atoms (which divides out molecule size), its pretrained fingerprint "
        f"carries enough size-correlated signal to reconstruct much of the count. "
        f"Given that CheMeleon was trained for calculated property prediction, it "
        f"is not surprising that it's good at this task, but it still lost some "
        f"information that the count fingerprint does contain. {_che_note}\n\n"
        f"On real solubility the picture is less clean. Counting does help here "
        f"too: count Morgan averages **R² {_mean(_s, 'count_linear'):.2f}** vs "
        f"**{_mean(_s, 'binary_linear'):.2f}** for binary (paired p = "
        f"{_p_sol:.0e}), but it's nowhere near the near-perfect fit it gets on a "
        f"pure accumulator, because solubility is only partly an accumulated "
        f"quantity. CheMeleon's pre-trained fingerprint is the best of the three "
        f"on solubility (mean **R² {_mean(_s, 'chemeleon_linear'):.2f}**)."
    ).callout(kind="info")

    _nr, _nf = _cv["n_repeats"], _cv["n_folds"]
    mo.vstack(
        [
            mo.md(
                f"**Encodings vs. two targets**, {_nr}×{_nf} scaffold "
                f"cross-validation on **{acc.n_molecules():,}** AqSolDB molecules. "
                f"<span style='color:#1c7ed6'>■ best</span> · "
                f"<span style='color:#868e96'>■ equivalent to best</span> · "
                f"<span style='color:#e03131'>■ significantly different from best</span> "
                f"(Tukey HSD, α = 0.05). At right, each grey line is one CV split."
            ),
            acc_metric,
            mo.as_html(_charts),
            _verdict,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Do count Morgan and CheMeleon actually handle the solubility *cliffs* better?

    Better average R² doesn't automatically mean better on cliffs. As we saw on the K-nearest-neigbor example, a model can win on average by fitting the smooth bulk of the data more closely while still predicting near-identical values for both halves of a cliff pair. So let's look at the cliffs in the AqSolDB dataset.

    We reuse the same 5×5 scaffold CV, but keep the out-of-fold prediction for every molecule (in each repeat, every molecule is predicted once, by a model that never saw its scaffold). A cliff pair is defined exactly as in the ADMET census earlier: binary-Morgan Tanimoto ≥ 0.7 and a solubility gap of more than 1.5 log units. This fixed definition means all the models are being evaluated on the same pairs of molecules. For each pair we ask: how much of the true solubility gap does the model's predicted gap recover? 100% means the cliff is fully reproduced; 0% means both molecules got the same prediction.

    In a good share of these cliff pairs, the two molecules have **identical** binary Morgan fingerprints (Tanimoto = 1.0), usually because they differ only in chain length or in how many times a group repeats. For those pairs, a binary-fingerprint model is forced to predict the same number for both molecules, no matter how it's trained.
    """)
    return


@app.cell
def _(alt, mo, pd, setup_ready):
    from fingerprints import accumulation as acc3

    assert setup_ready

    if acc3.has_cliff_data():
        _summ = acc3.cliff_summary()
        _pairs = pd.DataFrame(acc3.cliff_pairs_table())
    else:
        with mo.status.spinner(
            title="Collecting out-of-fold predictions for the cliff analysis (live, a few minutes)…"
        ):
            _summ = acc3.cliff_summary()
            _pairs = pd.DataFrame(acc3.cliff_pairs_table())

    _labels = {k: lbl for k, lbl, _f in acc3.CV_METHODS}
    _fam = {k: f for k, _l, f in acc3.CV_METHODS}
    _order = [_labels[k] for k, _l, _f in acc3.CV_METHODS if k in _summ["methods"]]
    _fam_color = {"binary": "#e03131", "count": "#2b8a3e", "learned": "#7048e8"}
    _model_color = alt.Color(
        "model:N",
        sort=_order,
        scale=alt.Scale(
            domain=_order,
            range=[_fam_color[_fam[k]] for k, _l, _f in acc3.CV_METHODS
                   if k in _summ["methods"]],
        ),
        legend=None,
    )

    _pairs["model"] = _pairs["key"].map(_labels)
    _pairs["pair type"] = _pairs["tanimoto"].map(
        lambda t: "identical binary FP (T = 1.0)" if t >= 0.999 else "similar (0.7 ≤ T < 1.0)"
    )
    _pairs["recovered"] = _pairs["pred_gap"] / _pairs["true_gap"]
    _n_ident = int((_pairs[_pairs["key"] == "binary_linear"]["pair type"]
                    .str.startswith("identical")).sum())

    # (1) gap recovered by pair type -- the headline cliff metric
    _rec = (
        _pairs.groupby(["model", "pair type"], as_index=False)["recovered"].mean()
    )
    _rec["recovered %"] = (_rec["recovered"] * 100).round(1)
    _rec_chart = (
        alt.Chart(_rec)
        .mark_bar()
        .encode(
            y=alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("recovered %:Q", title="% of the true cliff gap recovered",
                    scale=alt.Scale(domain=[-20, 100])),
            color=_model_color,
            tooltip=["model:N", "pair type:N", "recovered %:Q"],
        )
        .properties(width=300, height=120)
        .facet(row=alt.Row("pair type:N", title=None,
                           header=alt.Header(labelAngle=0, labelAlign="left")))
    )

    # (2) true vs predicted gap, one dot per cliff pair (predictions averaged
    # over the 5 CV repeats). Guide lines are drawn from the same data so the
    # layers can live inside one facet.
    _base = alt.Chart()
    _diag = _base.mark_line(strokeDash=[4, 4], color="#868e96").encode(
        x="true_gap:Q", y="true_gap:Q"
    )
    _zero = _base.mark_rule(color="#ced4da").encode(y=alt.datum(0))
    _scatter = (
        _base
        .mark_circle(size=22, opacity=0.6)
        .encode(
            x=alt.X("true_gap:Q", title="true solubility gap (logS)"),
            y=alt.Y("pred_gap:Q", title="predicted gap (logS)"),
            color=_model_color,
            shape=alt.Shape("pair type:N", legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["model:N", "pair type:N",
                     alt.Tooltip("tanimoto:Q", format=".2f"),
                     alt.Tooltip("true_gap:Q", format=".2f"),
                     alt.Tooltip("pred_gap:Q", format=".2f")],
        )
    )
    _scatter_chart = (
        alt.layer(_zero, _diag, _scatter, data=_pairs)
        .properties(width=190, height=190)
        .facet(facet=alt.Facet("model:N", sort=_order, title=None), columns=4)
    )

    # (3) RMSE on molecules that sit in a cliff vs everything else, per repeat
    _rows = []
    for _k, _m in _summ["methods"].items():
        for _r, (_c, _o) in enumerate(zip(_m["rmse_cliff"], _m["rmse_rest"])):
            _rows.append({"model": _labels[_k], "repeat": _r,
                          "molecules": "in a cliff pair", "RMSE": round(_c, 3)})
            _rows.append({"model": _labels[_k], "repeat": _r,
                          "molecules": "all others", "RMSE": round(_o, 3)})
    _rmse = pd.DataFrame(_rows)
    _rmse_chart = (
        alt.Chart(_rmse)
        .mark_point(filled=True, size=55)
        .encode(
            y=alt.Y("model:N", sort=_order, title=None, axis=alt.Axis(labelLimit=260)),
            yOffset=alt.YOffset("molecules:N", sort=["in a cliff pair", "all others"]),
            x=alt.X("RMSE:Q", title="out-of-fold RMSE (logS), one dot per CV repeat",
                    scale=alt.Scale(zero=False)),
            color=alt.Color("molecules:N", sort=["in a cliff pair", "all others"],
                            scale=alt.Scale(range=["#e8590c", "#adb5bd"]),
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=["model:N", "molecules:N", "repeat:Q", "RMSE:Q"],
        )
        .properties(width=420, height=200)
    )

    def _mean(key, metric):
        v = _summ["methods"][key][metric]
        return sum(v) / len(v)

    def _rec_of(key, ident):
        sub = _pairs[(_pairs["key"] == key)
                     & (_pairs["pair type"].str.startswith("identical") == ident)]
        return 100 * float(sub["recovered"].mean())

    _p_count = acc3.cliff_paired_p("gap_recovered", "binary_linear", "count_linear")
    _p_che = acc3.cliff_paired_p("gap_recovered", "binary_linear", "chemeleon_linear")
    _verdict = mo.md(
        f"Out of **{_summ['n_similar_pairs']:,}** structurally similar pairs in these "
        f"{_summ['n_mols']:,} molecules, **{_summ['n_cliff_pairs']}** are cliffs, and "
        f"**{_n_ident}** of those have identical binary Morgan fingerprints.\n\n"
        f"- **Binary Morgan** recovers only **{100 * _mean('binary_linear', 'gap_recovered'):.0f}%** "
        f"of the average cliff gap with a linear head. On the "
        f"identical-fingerprint pairs it recovers essentially nothing "
        f"({_rec_of('binary_linear', True):.0f}%), as it must.\n"
        f"- **Count Morgan** recovers **{100 * _mean('count_linear', 'gap_recovered'):.0f}%** "
        f"(paired p vs binary = {_p_count:.0e} over the 5 repeats), and "
        f"{_rec_of('count_linear', True):.0f}% even on the identical-fingerprint pairs, "
        f"because counting the extra –CH₂– units is exactly the information the binary "
        f"version threw away.\n"
        f"- **CheMeleon** does best, recovering **{100 * _mean('chemeleon_linear', 'gap_recovered'):.0f}%** "
        f"(p = {_p_che:.0e}), {_rec_of('chemeleon_linear', True):.0f}% on the "
        f"identical-fingerprint pairs.\n\n"
        f"So on solubility, the representations that carry more information really do "
        f"see more of the cliffs, not just the easy bulk. But none of them get close to "
        f"fully reproducing them: even the best model only recovers about half of a typical "
        f"cliff gap, and every model still has a higher error on molecules that sit in a "
        f"cliff pair (CheMeleon RMSE {_mean('chemeleon_linear', 'rmse_cliff'):.2f} vs "
        f"{_mean('chemeleon_linear', 'rmse_rest'):.2f} on the rest). Better encodings shrink "
        f"the cliff problem; they don't remove it. More recent efforts such as the Monroe model have made further incremental improvements but continue to run into the same fundamental issues."
    ).callout(kind="info")

    mo.vstack(
        [
            mo.md("**How much of each cliff's solubility gap does the model reproduce?**"),
            mo.as_html(_rec_chart),
            mo.md(
                "**Every cliff pair, true gap vs predicted gap.** Dots on the dashed "
                "line are perfectly reproduced cliffs; dots on the grey zero line got "
                "the same prediction for both molecules."
            ),
            mo.as_html(_scatter_chart),
            mo.md("**Prediction error on molecules in a cliff vs everything else:**"),
            mo.as_html(_rmse_chart),
            _verdict,
            mo.md("---"),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Leave fingerprints behind and look in the pocket

    If the signal isn't in *any* 2D-structure fingerprint, we need to try computation that's more information-rich, but harder to pull off. In this case, what we're going to look at is the **3D interactions** between molecule and protein in the binding pocket, not the
    2D graph. Below, we fold each ligand from the protein binding datasets we started this notebook off with into the pocket with **Boltz**, detect
    its contacts with **PLIP**, and read an interaction fingerprint off the
    pose.

    Let's be honest about what this is and isn't:

    - It's a **clue in a direction**, not a general fix. For the μ-opioid pair
      it points at a plausible cause (a single extra H-bond); for other datasets can be even less clear if the Boltz prediction can explain the experimental data. There yet more methods which require more investment of compute and expertise and could give much more rich information on what's happening.
    - These are **predicted** poses — binding-mode hypotheses, not
      experimental structures — for a handful of curated pairs. A
      qualitative contrast, not a benchmark or validated hypothesis.

    With those caveats in advance, here's the demonstration.
    """)
    return


@app.cell
def _(mo, selectors):
    # Synced selector mirror: switch target pair / molecule pair right
    # here without scrolling back to the top (same global elements).
    mo.vstack([mo.md('**Pick the target pair / molecule pair for these 3D poses:**'), selectors()])
    return


@app.cell
def _(ctx, cv, get_cliff_idx, get_pair_key, mo):
    from fingerprints import pose_view as pv
    from fingerprints.complex_viewer import ComplexViewer

    _tp = ctx.by_key()[get_pair_key()]
    _cliff = _tp.cliffs[get_cliff_idx()]
    _pair_key, _idx = _tp.key, get_cliff_idx()

    if not pv.has_poses(_pair_key, _idx):
        _view = mo.md(
            f"*No precomputed 3D poses for this cliff yet ({_tp.target_a} vs "
            f"{_tp.target_b}, pair {_idx + 1}). Poses were folded offline with "
            "Boltz-2 for a subset of cliffs — pick one of those, or run "
            "`python -m fingerprints.rebuild_poses` to add this one.*"
        ).callout(kind="info")
    else:
        _poses = pv.load_all(_pair_key, _idx)

        def _anchor_resi(pose):
            for rn, seq, _d in pv.pocket_residues(pose.cif_text):
                if rn == "ASP":
                    return seq
            return ""

        def _potency_word(pki):
            if pki >= 9.0:
                return "very potent", "#2b8a3e"
            if pki >= 7.5:
                return "potent", "#40a060"
            if pki >= 6.0:
                return "moderate", "#e8820c"
            return "weak", "#e03131"

        def _panel(mol_id, target, counterpart_mol, pki):
            p = _poses[f"{mol_id}_{target}"]
            changed = pv.changed_atom_names(p, _poses[f"{counterpart_mol}_{target}"])
            v = ComplexViewer(
                structure=p.cif_text,
                format="cif",
                highlight_resi=_anchor_resi(p),
                highlight_atoms=changed,
                interactions=pv.load_interactions(p),
                height=320,
            )
            _word, _color = _potency_word(pki)
            _mol_name = "molecule 1" if mol_id == "mol1" else "molecule 2"
            badge = mo.md(
                f"<div style='text-align:center'>"
                f"<b>{_mol_name}</b> — pKi <b>{pki}</b> "
                f"<span style='color:{_color}'><b>({_word})</b></span></div>"
            )
            return mo.vstack([badge, mo.ui.anywidget(v)])

        _ta, _tb = _tp.target_a, _tp.target_b

        # Summary table first: molecule x target grid so the cliff is obvious
        # before looking at any 3D. The cliff target's cells are boxed.
        def _cell(pki, is_cliff_target):
            _word, _color = _potency_word(pki)
            _border = "2px solid #e03131" if is_cliff_target else "1px solid #dee2e6"
            return (
                f"<td style='border:{_border};padding:6px 14px;text-align:center'>"
                f"pKi <b>{pki}</b><br>"
                f"<span style='color:{_color};font-size:12px'>{_word}</span></td>"
            )

        _cliff_on = _cliff.cliff_on
        _table = (
            "<table style='border-collapse:collapse;margin:0 auto'>"
            f"<tr><th></th>"
            f"<th style='padding:4px 14px'>{_ta}</th>"
            f"<th style='padding:4px 14px'>{_tb}</th></tr>"
            f"<tr><td style='padding:4px 10px;text-align:right'><b>molecule 1</b></td>"
            + _cell(_cliff.pki_1_a, _cliff_on == _ta)
            + _cell(_cliff.pki_1_b, _cliff_on == _tb)
            + "</tr>"
            f"<tr><td style='padding:4px 10px;text-align:right'><b>molecule 2</b><br>"
            f"<span style='font-size:11px;color:#868e96'>({_cliff.change})</span></td>"
            + _cell(_cliff.pki_2_a, _cliff_on == _ta)
            + _cell(_cliff.pki_2_b, _cliff_on == _tb)
            + "</tr></table>"
        )
        _summary = mo.vstack(
            [
                mo.Html(_table),
                mo.md(
                    f"The red-boxed column is **{_cliff_on}**, where the one-atom "
                    f"change is a **{cv.fold_change(max(_cliff.delta_a, _cliff.delta_b))} "
                    f"cliff**. On the other target it barely moves. Same two "
                    "molecules, both columns — the poses below are grouped by "
                    "target so you can compare the two molecules in the *same* "
                    "pocket side by side."
                ),
            ]
        )

        _view = mo.vstack(
            [
                _summary,
                mo.md(f"#### Both molecules in {_ta}"),
                mo.hstack(
                    [
                        _panel("mol1", _ta, "mol2", _cliff.pki_1_a),
                        _panel("mol2", _ta, "mol1", _cliff.pki_2_a),
                    ],
                    widths=[1, 1],
                    gap=1,
                ),
                mo.md(f"#### Both molecules in {_tb}"),
                mo.hstack(
                    [
                        _panel("mol1", _tb, "mol2", _cliff.pki_1_b),
                        _panel("mol2", _tb, "mol1", _cliff.pki_2_b),
                    ],
                    widths=[1, 1],
                    gap=1,
                ),
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(ctx, get_cliff_idx, get_pair_key, mo):
    from fingerprints import pose_view as pv2

    # The interaction fingerprint: encode each pose by the contacts it makes
    # (residue x interaction-type bits) and lay the four poses side by side.
    # Unlike the 2D fingerprints earlier in the notebook, these bits are read
    # off the binding event itself — the data telling us what to encode.
    _tp = ctx.by_key()[get_pair_key()]
    _idx = get_cliff_idx()
    if not pv2.has_poses(_tp.key, _idx):
        _view = mo.md("")
    else:
        _poses = pv2.load_all(_tp.key, _idx)
        _cliff = _tp.cliffs[_idx]

        def _potency_word(pki):
            if pki >= 9.0:
                return "very potent"
            if pki >= 7.5:
                return "potent"
            if pki >= 6.0:
                return "moderate"
            return "weak"

        _keys = [
            f"mol1_{_tp.target_a}", f"mol2_{_tp.target_a}",
            f"mol1_{_tp.target_b}", f"mol2_{_tp.target_b}",
        ]
        _col_pki = [_cliff.pki_1_a, _cliff.pki_2_a, _cliff.pki_1_b, _cliff.pki_2_b]
        _col_labels = [
            f"mol 1 · {_tp.target_a}", f"mol 2 · {_tp.target_a}",
            f"mol 1 · {_tp.target_b}", f"mol 2 · {_tp.target_b}",
        ]
        _col_labels = [
            f"{_lab}  —  {_potency_word(_p)} (pKi {_p})"
            for _lab, _p in zip(_col_labels, _col_pki)
        ]
        _bits, _fps = pv2.aligned_interaction_fingerprints(_poses, _keys)
        _type_label = dict(pv2.INTERACTION_TYPES)
        _type_color = {
            "saltbridge": "#e0a800", "hbond": "#4dabf7", "pistack": "#20c997",
            "pication": "#e64980", "hydrophobic": "#adb5bd",
        }

        # Hand-built HTML grid: rows = interaction bits, cols = the 4 poses.
        _html = [
            "<table style='border-collapse:collapse;font-size:12px'>",
            "<tr><th style='text-align:left;padding:2px 8px'>interaction bit</th>"
            + "".join(
                f"<th style='padding:2px 6px;writing-mode:vertical-rl;"
                f"transform:rotate(180deg)'>{c}</th>"
                for c in _col_labels
            )
            + "</tr>",
        ]
        for _res, _typ in _bits:
            _dot = _type_color.get(_typ, "#868e96")
            _label = (
                f"<span style='color:{_dot}'>●</span> {_res} "
                f"<span style='color:#868e96'>({_type_label.get(_typ, _typ)})</span>"
            )
            _cells = ""
            for _k in _keys:
                _on = _fps[_k].get((_res, _typ), 0)
                _bg = _dot if _on else "#f1f3f5"
                _cells += (
                    f"<td style='padding:0;border:1px solid #fff;width:70px;"
                    f"height:20px;background:{_bg}'></td>"
                )
            _html.append(
                f"<tr><td style='padding:2px 8px'>{_label}</td>{_cells}</tr>"
            )
        _html.append("</table>")

        _view = mo.vstack(
            [
                mo.md(
                    "**An interaction fingerprint, read off the pose.** Each row is "
                    "a contact the ligand makes; a filled cell means that pose has "
                    "it. Same idea as the 2D fingerprints from earlier — but here "
                    "the *data* (the binding pose) decides the bits, instead of us "
                    "imposing them."
                ),
                mo.Html("".join(_html)),
                mo.md(
                    "*Contacts via PLIP on the predicted poses. The salt bridge to "
                    "the conserved aspartate is the constant anchor; the cliff shows "
                    "up only as a subtle reshuffle of weaker H-bond / hydrophobic "
                    "bits — even this data-derived fingerprint doesn't obviously "
                    "explain the potency gap.*"
                ),
                mo.md("---"),
                mo.md("""
                The PLIP fingerprint is just intended to help document and understand the 3D pose, and is not really an alternative to chemical structure fingerprints. It's highly informative if you know what you're looking at, but to capture this information we've traded generality for specificity - it's possible and sensible to compare structural fingerprints of molecules between completely unrelated chemical origins or for unrelated tasks; however for the 3D interaction, the information is only useful in a particular context. You can't easily compare between different proteins, or even between different chemical series on the same protein - if the molecules don't share enough structural elements in common, they may have similar binding strengths but interact with totally different residues on the protein.
                """)
            ]
        )
    mo.vstack([_view, mo.md("---")])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## So what are fingerprints good for?

    It would be easy to read much of the above investigation into how fingerprints can't capture activity clfifs as an indictment! It isn't.
    Molecular fingerprints are one of the most useful abstractions in all of
    cheminformatics, and the struggle with activity cliffs is a direct consequence of why
    they work so well:

    - **The core assumption is right almost all of the time.** "Similar
      structure → similar property" held for the overwhelming majority of the
      similar pairs in these real datasets, and does in most. That is exactly why fingerprints
      power similarity search, clustering, library design, and property
      prediction across the entire field — they are cheap, fast, and often correct.
    - **They are a great baseline.** A fixed fingerprint plus a simple
      model is trivial to compute, needs no training of the representation, and
      is hard to beat on smooth, well-behaved endpoints. Despite the increasing usefulness of large foundation models, simple 'classic' ML with fingerprints as input can still turn out to be the best model for some datasets.
    - **We can understand their limits.** Because the classic fingerprints work with an unchanging algorithm, it is possible to develop an informed intuition for what they can and cannot do. My hope with this notebook was to help develop that intuition in myself and others.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ### About this notebook

    **AI use:** This notebook was built with AI assistance: it helped scaffold the marimo
    cells, the custom anywidget for visualizing the Boltz poses, and the analysis scripts; work through experiments, and write early drafts of the prose. Every chemical claim, data source, and result was
    reviewed; the majority of prose was written by the author, with a little bit of LLM prose left behind after review.

    All chemical structure handling runs through **RDKit**. Binding data are from **MoleculeACE**
    (curated ChEMBL bioactivities with published activity-cliff labels); the
    **ADMET** data (aqueous solubility from **AqSolDB**, lipophilicity from
    **AstraZeneca**) come from **Therapeutics Data Commons**, where we strip
    salts, keep the largest organic fragment, and de-duplicate by canonical
    parent SMILES before analysis. Every train/test split — for the learned
    model and the ADMET census alike — is a **Bemis–Murcko scaffold split** - I'm aware that this has limitations and may not be the most rigorous way to do a chemical dataset splitting, but didn't want to get overly complex just for these demonstrations. 3D complexes are **Boltz-2** predictions and protein–ligand interactions are detected with **PLIP**.

    **Reproducibility.** Heavy compute (folding, interaction detection, model
    training) runs offline and is cached in `data/`; the notebook only reads
    those caches, so it stays instant and deterministic. Fingerprint code and
    analyses live in this repo — see `README.md`.

    **Credits.** RDKit · chemprop (D-MPNN) · CheMeleon · Boltz-2 · PLIP ·
    MoleculeACE · Therapeutics Data Commons (AqSolDB, AstraZeneca) ·
    3Dmol.js · Altair · marimo. Thanks to OpenADMET and the
    marimo team for the competition.
    """)
    return


if __name__ == "__main__":
    app.run()
