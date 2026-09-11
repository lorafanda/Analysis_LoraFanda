"""A 360-degree rotating glass brain, as one sprite sheet, small enough to inline.

The loading animation has to exist before anything is loaded, so it cannot be a fetched
asset and it cannot be Niivue. It is the real fsaverage surface rendered once here, 36
frames around a full turn, packed into one strip and embedded in the page as a data URI.
CSS steps() then plays it - which keeps animating while the main thread is busy parsing
mesh data, which a JS-driven animation would not.

Rendered off-screen with pyvista. Dark ground to match the visualizer's black canvas,
semi-transparent pial surface so it reads as the glass brain the page itself draws.
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from PIL import Image

MESH = Path("S:/HumanNeuronLab/ANALYSIS/FLM/Analysis_LoraFanda/02_FBM_Clustering/outputs/250_recon/fsaverage/meshes")
OUT = Path("C:/Users/fanda/AppData/Local/Temp/claude/S--HumanNeuronLab-ANALYSIS-FLM-Analysis-LoraFanda/"
           "b2b76878-a2dc-444b-8806-1d2b9386c369/scratchpad")
N, PX = 36, 132

import pyvista as pv
pv.OFF_SCREEN = True

def load(h):
    g = nib.load(str(MESH / f"fsaverage_{h}.gii"))
    v = g.darrays[0].data.astype(float)
    f = g.darrays[1].data.astype(np.int64)
    faces = np.hstack([np.full((len(f), 1), 3), f]).ravel()
    return pv.PolyData(v, faces)

lh, rh = load("lh"), load("rh")
brain = lh.merge(rh)
brain = brain.decimate(0.75)          # 4x fewer triangles: it is 132 px wide
print("mesh", brain.n_points, "pts", brain.n_cells, "tris")

pl = pv.Plotter(off_screen=True, window_size=(PX * 2, PX * 2))   # 2x, downsampled later
pl.set_background("black")
pl.add_mesh(brain, color=(0.78, 0.80, 0.86), opacity=0.42, smooth_shading=True,
            specular=0.15, specular_power=20, ambient=0.35, diffuse=0.7)
pl.enable_anti_aliasing("ssaa")
# ROTATE THE MESH, NOT THE CAMERA. Two attempts at moving the camera per frame produced
# 36 identical views: camera.azimuth as an attribute is a no-op, and camera_position set
# in a loop is applied once and then ignored by screenshot() in off-screen mode. Rotating
# the points about the brain's own vertical axis and forcing a render cannot be cached
# wrongly, and it is the same motion: lateral, anterior, lateral, posterior.
c = np.array(brain.center)
pl.camera_position = "xz"              # lateral, view-up superior
pl.camera.zoom(1.5)
pl.render()
frames = []
step = 360.0 / N
for i in range(N):
    if i:
        brain.rotate_z(step, point=c, inplace=True)
    pl.render()
    img = pl.screenshot(return_img=True, transparent_background=False)
    im = Image.fromarray(img).convert("RGB").resize((PX, PX), Image.LANCZOS)
    frames.append(im)
    sys.stdout.write("."); sys.stdout.flush()
pl.close()
print()

sheet = Image.new("RGB", (PX * N, PX), "black")
for i, im in enumerate(frames):
    sheet.paste(im, (i * PX, 0))
sheet.save(OUT / "spin_sheet.png")
buf = io.BytesIO(); sheet.save(buf, "JPEG", quality=82, optimize=True)
b64 = base64.b64encode(buf.getvalue()).decode()
(OUT / "spin_sheet.b64").write_text(b64)
frames[9].save(OUT / "spin_frame9.png")
print(f"sprite {PX*N}x{PX}  jpeg {len(buf.getvalue())/1024:.0f} KB  base64 {len(b64)/1024:.0f} KB")
