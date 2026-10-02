"""A minimal 3Dmol.js molecule-viewer anywidget for marimo.

Renders a protein-ligand complex (from PDB/CIF text) in an interactive WebGL
canvas: rotate/zoom the receptor, see the ligand in the pocket, optionally
highlight a set of ligand atoms (e.g. the atoms that change across an activity
cliff). 3Dmol.js is loaded from a CDN inside the widget, so nothing heavy runs
in the Python kernel.

State is a handful of traitlets synced to the browser; the JS side (re)builds
the scene whenever they change. This is a custom widget precisely because no
off-the-shelf marimo element can show a 3D structure.
"""

from __future__ import annotations

import anywidget
import traitlets

_ESM = """
function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (window.$3Dmol) { resolve(); return; }
    if (document.querySelector(`script[src="${src}"]`)) {
      const check = () => window.$3Dmol ? resolve() : setTimeout(check, 50);
      check();
      return;
    }
    const s = document.createElement("script");
    s.src = src; s.onload = () => resolve(); s.onerror = reject;
    document.head.appendChild(s);
  });
}

async function render({ model, el }) {
  el.innerHTML = "";
  const container = document.createElement("div");
  container.style.width = "100%";
  container.style.height = (model.get("height") || 420) + "px";
  container.style.position = "relative";
  el.appendChild(container);

  await loadScript("https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.0/3Dmol-min.js");

  const viewer = window.$3Dmol.createViewer(container, { backgroundColor: "white" });

  function build() {
    viewer.clear();
    const data = model.get("structure");
    const fmt = model.get("format") || "pdb";
    if (!data) { viewer.render(); return; }
    viewer.addModel(data, fmt);

    // Protein: cartoon, faint.
    viewer.setStyle({}, { cartoon: { color: "spectrum", opacity: 0.55 } });

    // Ligand (HETATM, excluding water): sticks.
    const ligSel = { hetflag: true, not: { resn: ["HOH", "WAT"] } };
    viewer.setStyle(ligSel, { stick: { colorscheme: "greenCarbon", radius: 0.18 } });

    // Optionally emphasize a named pocket residue (e.g. the anchoring Asp).
    const res = model.get("highlight_resi");
    if (res) {
      viewer.setStyle(
        { resi: res },
        { stick: { colorscheme: "magentaCarbon", radius: 0.15 } }
      );
      viewer.addResLabels(
        { resi: res },
        { fontSize: 11, backgroundColor: "black", backgroundOpacity: 0.6 }
      );
    }

    // Highlighted ligand atoms (by atom name), if any: fat orange spheres.
    // Scoped to HETATM so ligand names can't collide with protein atom names.
    const hi = model.get("highlight_atoms") || [];
    if (hi.length) {
      viewer.setStyle(
        { hetflag: true, atom: hi },
        { stick: { colorscheme: "orangeCarbon", radius: 0.28 },
          sphere: { color: "orange", radius: 0.45 } }
      );
    }

    // Protein-ligand interactions (from PLIP): dashed lines colored by type.
    const COLORS = {
      saltbridge: "#ffd43b", hbond: "#4dabf7", pistack: "#20c997",
      pication: "#e64980", hydrophobic: "#adb5bd",
    };
    const interactions = model.get("interactions") || [];
    for (const it of interactions) {
      const c = COLORS[it.type] || "#868e96";
      viewer.addCylinder({
        start: { x: it.lig_xyz[0], y: it.lig_xyz[1], z: it.lig_xyz[2] },
        end: { x: it.prot_xyz[0], y: it.prot_xyz[1], z: it.prot_xyz[2] },
        radius: 0.06, color: c, dashed: true, fromCap: 1, toCap: 1,
      });
    }

    // Center + orient consistently across panels: zoom to the ligand, then set
    // a fixed camera orientation so every complex starts from the same angle
    // (otherwise 3Dmol picks an arbitrary orientation per structure).
    viewer.zoomTo(ligSel);
    viewer.zoom(0.7);
    // setView([cx, cy, cz, zoom, qx, qy, qz, qw]) - identity-ish quaternion
    // gives a reproducible front-on view; keep the zoomTo-derived center.
    const v = viewer.getView();
    viewer.setView([v[0], v[1], v[2], v[3], 0, 0, 0, 1]);
    viewer.render();
  }

  build();
  model.on("change:structure", build);
  model.on("change:highlight_atoms", build);
  model.on("change:highlight_resi", build);
  model.on("change:interactions", build);
}

export default { render };
"""


class ComplexViewer(anywidget.AnyWidget):
    """Interactive 3D viewer for a protein-ligand complex."""

    _esm = _ESM
    structure = traitlets.Unicode("").tag(sync=True)  # PDB/CIF text
    format = traitlets.Unicode("pdb").tag(sync=True)
    highlight_atoms = traitlets.List(traitlets.Unicode()).tag(sync=True)  # ligand atom names
    highlight_resi = traitlets.Unicode("").tag(sync=True)  # e.g. "149" for Asp149
    interactions = traitlets.List(traitlets.Dict()).tag(sync=True)  # PLIP records
    height = traitlets.Int(420).tag(sync=True)
