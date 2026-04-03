# # functions/lf_recon.py
# # Helpers for per-patient glass-brain rendering from iELVis files (PIAL-first)
# # - Prefers <subj>/elec_recon/<subj>.PIAL + <subj>.electrodeNames
# # - Fallback path is exposed but the main script can forbid it
# # - Depth shafts get tubes; subdurals get spheres
# # - QC: distance-to-pial included in CSV
# # - Debug: rich provenance when PIAL and names mismatch

# import os, re
# import numpy as np
# import nibabel.freesurfer as nfs
# import pyvista as pv
# from scipy.spatial import cKDTree

# # ---------------- Geometry & cameras ----------------

# def faces_to_pv(faces: np.ndarray) -> np.ndarray:
#     """Convert Nx3 triangle indices → PyVista flat face array [3,i,j,k, 3,...]."""
#     return np.c_[np.full(len(faces), 3), faces].ravel()

# def load_fs_surfaces(subj_dir: str):
#     """Load lh.pial & rh.pial as PyVista PolyData meshes."""
#     lh_v, lh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "lh.pial"))
#     rh_v, rh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "rh.pial"))
#     lh = pv.PolyData(lh_v, faces_to_pv(lh_f))
#     rh = pv.PolyData(rh_v, faces_to_pv(rh_f))
#     return lh, rh

# def build_views():
#     """Five canonical FreeSurfer tkrRAS camera presets."""
#     return {
#         "front":  ((0,  +320, 220), (0, 0, 50),  (0, 0, 1)),  # anterior
#         "right":  ((+320,   0, 220), (0, 0, 50),  (0, 0, 1)),  # right ear
#         "left":   ((-320,   0, 220), (0, 0, 50),  (0, 0, 1)),  # left ear
#         "back":   ((0,  -320, 220), (0, 0, 50),  (0, 0, 1)),  # posterior
#         "top":    ((0,     0, 360), (0, 0, 50),  (0, 1, 0)),  # dorsal
#     }

# # ---------------- iELVis file discovery & parsing ----------------

# def find_mgrid_path(subj_dir: str, pid: str) -> str:
#     """
#     Try common naming schemes inside <subject>/elec_recon/:
#       - elec_recon.mgrid
#       - {pid}.mgrid
#       - the only *.mgrid in the folder (if unique)
#     """
#     elec_dir = os.path.join(subj_dir, "elec_recon")
#     candidates = [
#         os.path.join(elec_dir, "elec_recon.mgrid"),
#         os.path.join(elec_dir, f"{pid}.mgrid"),
#     ]
#     for p in candidates:
#         if os.path.isfile(p):
# #             return p
# #     if os.path.isdir(elec_dir):
# #         globbed = [os.path.join(elec_dir, f) for f in os.listdir(elec_dir) if f.lower().endswith(".mgrid")]
# #         if len(globbed) == 1:
# #             return globbed[0]
# #     raise FileNotFoundError(f"Could not find .mgrid in {elec_dir}. Tried: {candidates}")

# # def electrode_names_path(subj_dir: str, pid: str) -> str:
# #     return os.path.join(subj_dir, "elec_recon", f"{pid}.electrodeNames")

# # def pial_coords_path(subj_dir: str, pid: str) -> str:
# #     return os.path.join(subj_dir, "elec_recon", f"{pid}.PIAL")

# # def parse_electrode_names(path: str, skip_first_n: int = 0):
# #     """
# #     Parse <subj>.electrodeNames (space-delimited).
# #     skip_first_n: skip this many non-empty lines at the top (e.g., header/date rows).
# #     Returns: names (list[str]), isLeft (np.uint8), isSubdural (np.uint8)
# #     Rule (per iELVis): col2 == 'D' => depth (0 subdural), else subdural=1
# #                        col3 == 'L' => isLeft=1 else 0
# #     """
# #     if not os.path.isfile(path):
# #         raise FileNotFoundError(f"Missing electrodeNames: {path}")
# #     names, isLeft, isSubd = [], [], []
# #     seen = 0
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         for line in f:
# #             s = line.strip()
# #             if not s:
# #                 continue
# #             if seen < skip_first_n:
# #                 seen += 1
# #                 continue
# #             toks = s.split()
# #             nm  = toks[0]
# #             typ = toks[1] if len(toks) > 1 else ""
# #             hem = toks[2] if len(toks) > 2 else ""
# #             names.append(nm)
# #             isSubd.append(0 if typ.upper() == "D" else 1)
# #             isLeft.append(1 if hem.upper() == "L" else 0)
# #     return names, np.array(isLeft, dtype=np.uint8), np.array(isSubd, dtype=np.uint8)

# # def load_pial_coords(path: str):
# #     """
# #     Load <subj>.PIAL coordinates (skip the first header line if present).
# #     Returns Nx3 float array.
# #     """
# #     if not os.path.isfile(path):
# #         raise FileNotFoundError(f"Missing PIAL: {path}")
# #     rows = []
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         lines = f.readlines()
# #     for i, line in enumerate(lines):
# #         toks = line.strip().replace(",", " ").split()
# #         vals = []
# #         for t in toks[:3]:
# #             try:
# #                 vals.append(float(t))
# #             except Exception:
# #                 vals = []
# #                 break
# #         if len(vals) == 3:
# #             rows.append(vals)
# #     if not rows:
# #         raise ValueError(f"No numeric XYZ rows found in PIAL: {path}")
# #     return np.array(rows, dtype=float)

# # # ---------------- .mgrid parser (unused in PIAL-only, but kept for completeness) ----------------

# # def parse_mgrid_vtkpx(path: str):
# #     """
# #     Minimal parser for vtkpxElectrodeMultiGrid .mgrid files.
# #     Returns list of dicts: [{"shaft": str, "xyz": [(x,y,z), ...]}, ...] in file order.
# #     """
# #     grids = []
# #     shaft, xyz, want_pos = None, [], False
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         for raw in f:
# #             line = raw.strip()
# #             if line == "#Description":
# #                 shaft, xyz, want_pos = None, [], False
# #                 continue
# #             if shaft is None and line and not line.startswith(("#", "-")):
# #                 shaft = line
# #                 continue
# #             if line == "#Position":
# #                 want_pos = True
# #                 continue
# #             if want_pos and line and not line.startswith(("#", "-")):
# #                 toks = line.replace(",", " ").split()
# #                 if len(toks) >= 3:
# #                     try:
# #                         xyz.append(tuple(map(float, toks[:3])))
# #                     except Exception:
# #                         pass
# #                 want_pos = False
# #             if line.startswith("# Electrode Grid"):
# #                 if shaft is not None and xyz:
# #                     grids.append({"shaft": shaft, "xyz": xyz})
# #                     shaft, xyz = None, []
# #         if shaft is not None and xyz:
# #             grids.append({"shaft": shaft, "xyz": xyz})
# #     return grids

# # def make_contacts(grids):
# #     """Flatten grids into list of (name, xyz[np.float64,3], shaft). Names auto: '{shaft}{idx}'."""
# #     out = []
# #     for g in grids:
# #         for i, p in enumerate(g["xyz"], 1):
# #             out.append((f"{g['shaft']}{i}", np.array(p, float), g["shaft"]))
# #     return out

# # # ---------------- Affines & space chooser (kept for completeness) ----------------

# # def _to_hom(x: np.ndarray) -> np.ndarray:
# #     return np.c_[x, np.ones((len(x), 1))]

# # def apply_affine(x: np.ndarray, A4: np.ndarray) -> np.ndarray:
# #     return (_to_hom(x) @ A4.T)[:, :3]

# # def choose_best_space(xyz_raw: np.ndarray,
# #                       brain_pts: np.ndarray,
# #                       vox2ras: np.ndarray,
# #                       vox2ras_tkr: np.ndarray,
# #                       dims_ijk,
# #                       override: str | None = None):
# #     """
# #     Robust space chooser for .mgrid: maps many hypotheses to tkrRAS and scores by
# #     (inside cortex bbox, median distance to surface).
# #     """
# #     A_sras_to_tkr = vox2ras_tkr @ np.linalg.inv(vox2ras)
# #     LPS2RAS = np.diag([-1, -1, 1, 1])

# #     def aff(x, A): return apply_affine(x, A)

# #     candidates = {
# #         "tkrRAS":     xyz_raw.copy(),
# #         "scannerRAS": aff(xyz_raw, A_sras_to_tkr),
# #         "IJK":        aff(xyz_raw, vox2ras_tkr),
# #         "scannerLPS": aff(xyz_raw, A_sras_to_tkr @ LPS2RAS),
# #     }

# #     perms = {
# #         "IJK": (0, 1, 2),
# #         "JIK": (1, 0, 2),
# #         "IKJ": (0, 2, 1),
# #         "JKI": (1, 2, 0),
# #         "KIJ": (2, 0, 1),
# #         "KJI": (2, 1, 0),
# #     }
# #     dims = np.asarray(dims_ijk, dtype=float)
# #     half = (dims - 1.0) / 2.0

# #     def add_variant(lbl, ijk_pts):
# #         candidates[f"IJK[{lbl}]"] = aff(ijk_pts, vox2ras_tkr)
# #         candidates[f"IJK[{lbl}+center]"] = aff(ijk_pts + half, vox2ras_tkr)
# #         candidates[f"IJK[{lbl}-1based]"] = aff(ijk_pts - 1.0, vox2ras_tkr)
# #         candidates[f"IJK[{lbl}+center-1based]"] = aff(ijk_pts + half - 1.0, vox2ras_tkr)

# #     for lbl, o in perms.items():
# #         ijk = xyz_raw[:, list(o)]
# #         add_variant(lbl, ijk)

# #     if override:
# #         if override not in candidates:
# #             raise ValueError(f"INPUT_SPACE_OVERRIDE '{override}' not in candidates.")
# #         return override, candidates[override], candidates, {}

# #     lo = brain_pts.min(0) - 10.0
# #     hi = brain_pts.max(0) + 10.0
# #     tree = cKDTree(brain_pts)
# #     scores = {}
# #     for k, pts in candidates.items():
# #         inside = np.all((pts >= lo) & (pts <= hi), axis=1).mean()
# #         d, _ = tree.query(pts, k=1)
# #         med_d = float(np.median(d))
# #         scores[k] = (inside, med_d)
# #     label = max(scores.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
# #     return label, candidates[label], candidates, scores

# # # ---------------- QC & grouping ----------------

# # def distances_to_pial(points_tkr: np.ndarray, lh: pv.PolyData, rh: pv.PolyData):
# #     """Return per-point nearest-surface Euclidean distance (mm) to either hemisphere pial."""
# #     pts = np.vstack([lh.points, rh.points])
# #     tree = cKDTree(pts)
# #     dists, _ = tree.query(points_tkr, k=1)
# #     return dists

# # def shaft_prefix(name: str) -> str:
# #     """Extract shaft/group prefix (letters/underscores up to first digit)."""
# #     m = re.match(r"^([A-Za-z_]+)", name)
# #     return m.group(1) if m else name

# # def contact_index(name: str) -> int:
# #     """Extract trailing integer index (e.g., LA12 -> 12); returns 0 if none."""
# #     m = re.search(r"(\d+)$", name)
# #     return int(m.group(1)) if m else 0

# # def group_depth_shafts(names, points_tkr, is_subdural):
# #     """
# #     Group only depth contacts into ordered shafts.
# #     Returns dict: {shaft_name: [(name, point), ... in numeric order]}
# #     """
# #     out = {}
# #     for n, p, sd in zip(names, points_tkr, is_subdural):
# #         if sd:  # subdural → skip for tubes
# #             continue
# #         sh = shaft_prefix(n)
# #         out.setdefault(sh, []).append((n, p))
# #     for sh in out:
# #         out[sh].sort(key=lambda npair: contact_index(npair[0]))
# #     return out

# # # ---------------- DEBUG HELPERS FOR PIAL BUNDLE ----------------

# # AUX_NAME_REGEX = re.compile(r'^(REF|GND|EKG|ECG|DC|TRIG|MARK|EMG|EOG|AUX)', re.I)

# # def _read_electrode_names_loose(path: str, skip_first_n: int = 0):
# #     names = []
# #     seen = 0
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         for raw in f:
# #             s = raw.strip()
# #             if not s:
# #                 continue
# #             if seen < skip_first_n:
# #                 seen += 1
# #                 continue
# #             names.append(s.split()[0])
# #     aux_mask = np.array([bool(AUX_NAME_REGEX.match(nm)) for nm in names], dtype=bool)
# #     return names, aux_mask

# # def _read_pial_numeric_rows_loose(path: str):
# #     coords = []
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         lines = f.readlines()
# #     for line in lines:
# #         toks = line.strip().replace(",", " ").split()
# #         row = []
# #         for t in toks[:3]:
# #             try:
# #                 row.append(float(t))
# #             except Exception:
# #                 row = []
# #                 break
# #         if len(row) == 3:
# #             coords.append(row)
# #     return np.array(coords, dtype=float), len(lines), len(coords)

# # def debug_pial_bundle(subj_dir: str, pid: str, skip_first_n: int = 0):
# #     en_path = electrode_names_path(subj_dir, pid)
# #     pial_path = pial_coords_path(subj_dir, pid)
# #     out = {
# #         "electrodeNames_path": en_path,
# #         "PIAL_path": pial_path,
# #         "electrodeNames_exists": os.path.isfile(en_path),
# #         "PIAL_exists": os.path.isfile(pial_path),
# #     }
# #     if not (out["electrodeNames_exists"] and out["PIAL_exists"]):
# #         return out

# #     names, aux_mask = _read_electrode_names_loose(en_path, skip_first_n=skip_first_n)
# #     coords, n_total, n_numeric = _read_pial_numeric_rows_loose(pial_path)

# #     out.update({
# #         "n_names": len(names),
# #         "n_names_aux": int(aux_mask.sum()),
# #         "n_names_nonaux": int((~aux_mask).sum()),
# #         "n_pial_lines": n_total,
# #         "n_pial_numeric": n_numeric,
# #         "first_names": names[:5],
# #         "last_names": names[-5:],
# #         "first_coords": coords[:3].tolist() if n_numeric else [],
# #         "last_coords": coords[-3:].tolist() if n_numeric else [],
# #     })

# #     hints = []
# #     if aux_mask.any():
# #         hints.append(f"aux labels present: {int(aux_mask.sum())}")
# #     if len(names) != n_numeric:
# #         hints.append("names vs PIAL numeric count differ")
# #     if not hints:
# #         hints.append("no obvious mismatch")
# #     out["hint"] = "; ".join(hints)
# #     return out

# # # ---------------- High-level loader ----------------

# # def load_electrodes_for_subject(subj_dir: str,
# #                                 pid: str,
# #                                 vox2ras: np.ndarray,
# #                                 vox2ras_tkr: np.ndarray,
# #                                 lh: pv.PolyData,
# #                                 rh: pv.PolyData,
# #                                 prefer_pial: bool = True,
# #                                 space_override: str | None = None,
# #                                 names_skip_first_n: int = 0):
# #     """
# #     Returns dict with:
# #       - If PIAL used:
# #           names, points_tkr (Nx3), source_space="PIAL",
# #           isLeft (N), isSubdural (N),
# #           dist_to_pial_mm (N),
# #           provenance (paths, debug)
# #       - If fallback required:
# #           needs_mgrid_mapping=True,
# #           mgrid = {names, xyz_raw, shafts},
# #           provenance (paths, debug, reason)
# #     """
# #     prov = {}
# #     en_path  = electrode_names_path(subj_dir, pid)
# #     pial_path= pial_coords_path(subj_dir, pid)
# #     prov["electrodeNames_path"] = en_path
# #     prov["PIAL_path"]           = pial_path
# #     prov["electrodeNames_exists"] = os.path.isfile(en_path)
# #     prov["PIAL_exists"]           = os.path.isfile(pial_path)

# #     # Preferred: PIAL + electrodeNames with 1:1 rows
# #     if prefer_pial and prov["electrodeNames_exists"] and prov["PIAL_exists"]:
# #         dbg = debug_pial_bundle(subj_dir, pid, skip_first_n=names_skip_first_n)
# #         prov["pial_debug"] = dbg

# #         try:
# #             names, isLeft, isSubd = parse_electrode_names(en_path, skip_first_n=names_skip_first_n)
# #             pial_xyz = load_pial_coords(pial_path)

# #             if len(pial_xyz) != len(names):
# #                 prov["reason"] = (f"PIAL length ({len(pial_xyz)}) != names length ({len(names)}) "
# #                                   f"→ fallback to .mgrid")
# #             else:
# #                 d2p = distances_to_pial(pial_xyz, lh, rh)
# #                 return {
# #                     "names": names,
# #                     "points_tkr": pial_xyz,
# #                     "source_space": "PIAL",
# #                     "isLeft": isLeft,
# #                     "isSubdural": isSubd,
# #                     "dist_to_pial_mm": d2p,
# #                     "provenance": prov
# #                 }
# #         except Exception as e:
# #             prov["reason"] = f"PIAL parse failed: {e}"

# #     # Fallback payload (the main script can choose to forbid using it)
# #     mgrid_path = find_mgrid_path(subj_dir, pid)
# #     from .lf_recon import parse_mgrid_vtkpx, make_contacts  # safe self-import for clarity
# #     grids = parse_mgrid_vtkpx(mgrid_path)
# #     contacts = make_contacts(grids)
# #     if not contacts:
# #         raise RuntimeError(f"{pid}: parsed 0 contacts from {mgrid_path}")
# #     m_names, m_xyz, m_shafts = zip(*contacts)
# #     xyz_raw = np.vstack(m_xyz)
# #     prov["mgrid_path"] = mgrid_path
# #     return {
# #         "needs_mgrid_mapping": True,
# #         "mgrid": {
# #             "names": list(m_names),
# #             "xyz_raw": xyz_raw,
# #             "shafts": list(m_shafts),
# #         },
# #         "provenance": prov
# #     }


# # # functions/lf_recon.py
# # import os, re, numpy as np, nibabel.freesurfer as nfs, pyvista as pv
# # from scipy.spatial import cKDTree

# # # ---------- mesh I/O ----------
# # def _faces_to_pv(faces):  # FreeSurfer faces -> PyVista faces
# #     return np.c_[np.full(len(faces), 3), faces].ravel()

# # def load_fs_surfaces(subj_dir):
# #     lh_v, lh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "lh.pial"))
# #     rh_v, rh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "rh.pial"))
# #     lh = pv.PolyData(lh_v, _faces_to_pv(lh_f))
# #     rh = pv.PolyData(rh_v, _faces_to_pv(rh_f))
# #     return lh, rh

# # # ---------- normals ----------
# # def safe_compute_normals(mesh):
# #     try:
# #         return mesh.compute_normals(
# #             consistent_normals=True, auto_orient_normals=True,
# #             feature_angle=180.0, splitting=False
# #         )
# #     except TypeError:
# #         return mesh.compute_normals(
# #             consistent_normals=True, auto_orient_normals=True,
# #             feature_angle=180.0
# #         )

# # # ---------- electrode loading (PIAL-only) ----------
# # def _read_electrode_names(path, skip_first_n=0):
# #     names, isLeft, isSubd = [], [], []
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         lines = [ln.strip() for ln in f if ln.strip()]
# #     lines = lines[skip_first_n:] if skip_first_n else lines
# #     for ln in lines:
# #         toks = ln.split()
# #         if not toks: continue
# #         nm = toks[0]
# #         typ = (toks[1] if len(toks) > 1 else "")
# #         hem = (toks[2] if len(toks) > 2 else "")
# #         names.append(nm)
# #         isSubd.append(0 if typ.upper()=="D" else 1)     # D=depth else subdural
# #         isLeft.append(1 if hem.upper()=="L" else 0)     # L=left, else right
# #     return np.array(names, dtype=object), np.array(isLeft, int), np.array(isSubd, int)

# # def _read_pial_coords(path):
# #     pts = []
# #     with open(path, "r", encoding="utf-8", errors="ignore") as f:
# #         for ln in f:
# #             ln = ln.strip()
# #             if not ln: continue
# #             # skip header lines that aren't 3 floats
# #             try:
# #                 x,y,z = map(float, ln.replace(",", " ").split()[:3])
# #                 pts.append((x,y,z))
# #             except Exception:
# #                 continue
# #     return np.array(pts, float)

# # def distances_to_pial(points_tkr, lh, rh):
# #     # combine both hemispheres
# #     all_pts = np.vstack([lh.points, rh.points])
# #     tree = cKDTree(all_pts)
# #     d, _ = tree.query(points_tkr, k=1)
# #     return d

# # def load_electrodes_for_subject(
# #     subj_dir, pid, vox2ras, vox2ras_tkr, lh, rh,
# #     prefer_pial=True, space_override=None, names_skip_first_n=0
# # ):
# #     en_path = os.path.join(subj_dir, "elec_recon", f"{pid}.electrodeNames")
# #     pi_path = os.path.join(subj_dir, "elec_recon", f"{pid}.PIAL")

# #     if prefer_pial:
# #         prov = {
# #             "pid": pid,
# #             "electrodeNames": en_path,
# #             "PIAL": pi_path,
# #             "mode": "PIAL-only"
# #         }
# #         if not (os.path.isfile(en_path) and os.path.isfile(pi_path)):
# #             return {"needs_mgrid_mapping": True, "provenance": {**prov, "reason": "missing PIAL or electrodeNames"}}

# #         names, isLeft, isSubd = _read_electrode_names(en_path, skip_first_n=names_skip_first_n)
# #         pts = _read_pial_coords(pi_path)

# #         if len(pts) != len(names):
# #             return {"needs_mgrid_mapping": True,
# #                     "provenance": {**prov, "reason": f"count mismatch PIAL({len(pts)}) != names({len(names)})"}}

# #         points_tkr = pts.copy()  # PIAL already in (surface) RAS
# #         d2p = distances_to_pial(points_tkr, lh, rh)

# #         return {
# #             "names": names,
# #             "points_tkr": points_tkr,
# #             "isLeft": isLeft,
# #             "isSubdural": isSubd,
# #             "dist_to_pial_mm": d2p,
# #             "source_space": "PIAL",
# #             "needs_mgrid_mapping": False,
# #             "provenance": prov,
# #         }

# #     # not used in this project (mgrid fallback removed)
# #     return {"needs_mgrid_mapping": True, "provenance": {"pid": pid, "reason": "prefer_pial=False not implemented"}}

# # # ---------- shafts / coloring ----------
# # _GRID_COLOR = "#5C78FF"  # fixed color for grid ("G*") channels

# # def shaft_key_for_coloring(name):
# #     """Return shaft key: 'GRID' if startswith 'G', else first two letters (uppercase) or whole letters if <2."""
# #     m = re.match(r'^([A-Za-z]+)', str(name))
# #     if not m: return "MISC"
# #     letters = m.group(1).upper()
# #     if letters.startswith("G"):
# #         return "GRID"
# #     return letters[:2] if len(letters) >= 2 else letters

# # def group_depth_shafts(names, points_tkr, isSubdural):
# #     """Depth-only shafts for tube rendering (unchanged behavior)."""
# #     out = {}
# #     for n, p, sd in zip(names, points_tkr, isSubdural):
# #         if sd:  # skip subdurals for tubes
# #             continue
# #         key = shaft_key_for_coloring(n)
# #         out.setdefault(key, []).append((n, p))
# #     return out

# # def group_all_shafts_for_coloring(names, points_tkr):
# #     """All electrodes (depth + subdural) grouped by color shaft key."""
# #     out = {}
# #     for n, p in zip(names, points_tkr):
# #         key = shaft_key_for_coloring(n)
# #         out.setdefault(key, []).append((n, p))
# #     return out

# # def build_shaft_colors(keys):
# #     """Assign colors per shaft key with GRID fixed; remaining keys cycle a palette."""
# #     PALETTE = [
# #         "#e41a1c","#377eb8","#4daf4a","#ffd700","#984ea3","#ff7f00",
# #         "#17becf","#a65628","#f781bf","#1f78b4","#2ca02c","#d62728",
# #         "#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#aec7e8",
# #         "#98df8a","#ffbb78",
# #     ]
# #     keys = list(sorted(keys))
# #     colors = {}
# #     # pin GRID first
# #     if "GRID" in keys:
# #         colors["GRID"] = _GRID_COLOR
# #         keys.remove("GRID")
# #     for i, k in enumerate(keys):
# #         colors[k] = PALETTE[i % len(PALETTE)]
# #     return colors

# # # ---------- rendering helpers ----------
# # def add_soft_lights(pl):
# #     try:
# #         pl.add_light(pv.Light(position=(300,300,400), focal_point=(0,0,0), intensity=0.20))
# #         pl.add_light(pv.Light(position=(-400,150,150), focal_point=(0,0,0), intensity=0.10))
# #         pl.add_light(pv.Light(position=(0,-400,300),  focal_point=(0,0,0), intensity=0.15))
# #     except Exception:
# #         pass

# # def render_single_view_image(
# #     lh, rh, *,
# #     names, points_tkr, isSubdural, view, cam,
# #     by_shaft_tubes, shaft_colors, highlights,
# #     brain_color="#ead6db", brain_opacity=0.35,
# #     specular=0.02, specular_power=8, ambient=0.34, diffuse=0.66,
# #     per_pane_size=(1000,950), ss_scale=2, depth_peeling=False, transparent_bg=True,
# #     other_color="#999999", other_radius=1.2, other_opacity=0.9,
# #     hi_color="#D55E00", hi_radius=2.2, hi_halo=True
# # ):
# #     pl = pv.Plotter(off_screen=True, window_size=per_pane_size)
# #     pl.set_background("white")
# #     try: pl.renderer.SetTwoSidedLighting(True)
# #     except Exception: pass
# #     if depth_peeling:
# #         try: pl.enable_depth_peeling(number_of_peels=120, occlusion_ratio=0.0)
# #         except Exception: pass
# #     try: pl.enable_anti_aliasing()
# #     except Exception: pass
# #     add_soft_lights(pl)

# #     for hemi in (lh, rh):
# #         pl.add_mesh(
# #             hemi, color=brain_color, opacity=brain_opacity, smooth_shading=True,
# #             backface_culling=False,
# #             specular=specular, specular_power=specular_power,
# #             ambient=ambient, diffuse=diffuse, show_scalar_bar=False
# #         )

# #     # tubes (depth shafts only)
# #     for shaft, seq in (by_shaft_tubes or {}).items():
# #         if len(seq) < 2: continue
# #         pts = np.vstack([p for (_, p) in seq])
# #         tube = pv.Spline(pts, n_points=max(50, len(seq)*10)).tube(radius=0.45)
# #         pl.add_mesh(tube, color=shaft_colors.get(shaft, "#56B4E9"))

# #     # color by shaft key for all contacts (G* unified)
# #     # build name -> shaftkey map across ALL electrodes for coloring
# #     by_all = group_all_shafts_for_coloring(names, points_tkr)
# #     name2shaft = {n: k for k, seq in by_all.items() for (n, _p) in seq}
# #     hi = set(highlights or [])

# #     for n, p, sd in zip(names, points_tkr, isSubdural):
# #         if n in hi:
# #             pl.add_mesh(pv.Sphere(radius=hi_radius, center=p), color=hi_color, opacity=1.0)
# #             if hi_halo:
# #                 pl.add_mesh(pv.Sphere(radius=hi_radius*1.12, center=p),
# #                             style="wireframe", color="black", line_width=1.0, opacity=0.9)
# #         else:
# #             c = shaft_colors.get(name2shaft.get(n, ""), other_color)
# #             pl.add_mesh(pv.Sphere(radius=other_radius, center=p), color=c, opacity=other_opacity)

# #     pl.camera_position = cam
# #     pl.add_text(view, position="upper_left", font_size=16, color="black")
# #     img = pl.screenshot(return_img=True, scale=ss_scale, transparent_background=transparent_bg)
# #     pl.close()
# #     return img

# # def stitch_three_horiz(imgs):
# #     h = min(im.shape[0] for im in imgs)
# #     out = []
# #     for im in imgs:
# #         if im.shape[0] == h:
# #             out.append(im)
# #         else:
# #             dy = (im.shape[0] - h) // 2
# #             out.append(im[dy:dy+h, :, :])
# #     return np.concatenate(out, axis=1)




# # functions/lf_recon.py
# # Helpers for per-patient mosaics (PIAL-only; ipsi-side panels)

# import os, re, numpy as np, nibabel as nib, pyvista as pv

# # ---------- paths / geometry ----------

# def subj_dir_for(pid: str, root_pat: str, root_micro: str) -> str:
#     if pid.startswith("PAT_"):
#         return os.path.join(root_pat, pid, "anatomy", "prep", "freesurfer")
#     if pid.startswith("MicroEPI"):
#         return os.path.join(root_micro, pid, "anatomy", "prep", "freesurfer")
#     raise ValueError(f"Unknown patient prefix: {pid}")

# def load_fs_surfaces(subj_dir: str):
#     import nibabel.freesurfer as nfs
#     def faces_to_pv(f): return np.c_[np.full(len(f), 3), f].ravel()
#     lh_v, lh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "lh.pial"))
#     rh_v, rh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "rh.pial"))
#     return (pv.PolyData(lh_v, faces_to_pv(lh_f)),
#             pv.PolyData(rh_v, faces_to_pv(rh_f)))

# def safe_compute_normals(mesh: pv.PolyData) -> pv.PolyData:
#     try:
#         return mesh.compute_normals(consistent_normals=True,
#                                     auto_orient_normals=True,
#                                     feature_angle=180.0,
#                                     splitting=False)
#     except TypeError:
#         return mesh.compute_normals(consistent_normals=True,
#                                     auto_orient_normals=True,
#                                     feature_angle=180.0)

# # ---------- electrodes (PIAL-only) ----------

# def electrode_names_path(subj_dir, pid):  # internal
#     return os.path.join(subj_dir, "elec_recon", f"{pid}.electrodeNames")

# def pial_path(subj_dir, pid):              # internal
#     return os.path.join(subj_dir, "elec_recon", f"{pid}.PIAL")

# def parse_electrode_names(path):
#     names, left, subd = [], [], []
#     with open(path, "r", encoding="utf-8", errors="ignore") as f:
#         for raw in f:
#             row = raw.strip().split()
#             if not row: 
#                 continue
#             nm = row[0]
#             typ = (row[1] if len(row) > 1 else "S")   # S=subdural default
#             side= (row[2] if len(row) > 2 else "R")
#             names.append(nm)
#             subd.append(0 if typ.upper().startswith("D") else 1)
#             left.append(1 if side.upper().startswith("L") else 0)
#     return np.array(names), np.array(left, np.uint8), np.array(subd, np.uint8)

# def load_electrodes_for_subject(
#     subj_dir, pid, vox2ras, vox2ras_tkr, lh, rh,
#     prefer_pial=True, space_override=None, names_skip_first_n=0
# ):
#     en = electrode_names_path(subj_dir, pid)
#     pp = pial_path(subj_dir, pid)
#     prov = {"pid": pid, "electrodeNames": en, "PIAL": pp, "reason": ""}

#     if not (os.path.isfile(en) and os.path.isfile(pp)):
#         prov["reason"] = "PIAL bundle missing."
#         return {"needs_mgrid_mapping": True, "provenance": prov}

#     names, isLeft, isSubd = parse_electrode_names(en)
#     if names_skip_first_n:
#         keep = np.arange(len(names)) >= int(names_skip_first_n)
#         names, isLeft, isSubd = names[keep], isLeft[keep], isSubd[keep]

#     xyz = []
#     with open(pp, "r", encoding="utf-8", errors="ignore") as f:
#         for raw in f:
#             t = raw.strip().split()
#             if len(t) >= 3:
#                 try: xyz.append([float(t[0]), float(t[1]), float(t[2])])
#                 except: pass
#     xyz = np.asarray(xyz, float)

#     if len(xyz) != len(names):
#         prov["reason"] = f"PIAL length ({len(xyz)}) != names length ({len(names)})"
#         return {"needs_mgrid_mapping": True, "provenance": prov}

#     points_tkr = xyz.copy()  # PIAL RAS ~ tkrRAS
#     d2p = distances_to_pial(points_tkr, lh, rh)

#     return dict(
#         names=list(names),
#         points_tkr=points_tkr,
#         isLeft=isLeft,
#         isSubdural=isSubd,
#         dist_to_pial_mm=d2p,
#         source_space="PIAL",
#         provenance=prov,
#     )

# def distances_to_pial(pts, lh, rh):
#     from scipy.spatial import cKDTree
#     all_pts = np.vstack([lh.points, rh.points])
#     tree = cKDTree(all_pts)
#     d, _ = tree.query(np.asarray(pts), k=1)
#     return d

# def group_depth_shafts(names, points_tkr, isSubd):
#     """Only depths (isSubd==0), grouped by alphanumeric shaft prefix before digits."""
#     out = {}
#     for n, p, sd in zip(names, points_tkr, isSubd):
#         if int(sd) == 1:
#             continue
#         m = re.match(r"([A-Za-z]+[A-Za-z]?)[0-9]+", n)
#         shaft = m.group(1) if m else n
#         out.setdefault(shaft, []).append((n, np.asarray(p, float)))
#     return out

# def make_name2shaft(by_shaft):
#     """Map contact name -> shaft (for depths)."""
#     return {n: shaft for shaft, seq in by_shaft.items() for (n, _p) in seq}

# # ---------- coloring ----------

# def two_letter_key(name: str) -> str:
#     m = re.match(r"([A-Za-z]+)", name)
#     if not m: return "UNK"
#     s = m.group(1).upper()
#     return "GRID" if s.startswith("G") else (s[:2] if len(s) >= 2 else s)

# def build_strip_prefix_colors(strip_names, palette20, grid_color):
#     """Colors for subdural strips/grids ONLY (by 2-letter key; 'GRID' single color)."""
#     prefixes = {two_letter_key(n) for n in strip_names}
#     keys = sorted(prefixes)
#     lut, i = {}, 0
#     for k in keys:
#         if k == "GRID":
#             lut[k] = grid_color
#         else:
#             lut[k] = palette20[i % len(palette20)]
#             i += 1
#     return lut

# def build_depth_shaft_colors(shafts, palette20):
#     """Unique color per depth shaft."""
#     shafts = sorted(list(shafts))
#     return {s: palette20[i % len(palette20)] for i, s in enumerate(shafts)}

# # ---------- rendering ----------

# def add_soft_lights(pl: pv.Plotter):
#     try:
#         pl.add_light(pv.Light(position=(300, 300, 400), focal_point=(0, 0, 0), intensity=0.20))
#         pl.add_light(pv.Light(position=(-400, 150, 150), focal_point=(0, 0, 0), intensity=0.10))
#         pl.add_light(pv.Light(position=(0, -400, 300),  focal_point=(0, 0, 0), intensity=0.15))
#     except Exception:
#         pass

# def render_single_view_image(
#     lh, rh, names, pts, isLeft, isSubd,
#     view_name, cam, show_hemi,
#     by_shaft, shaft_colors_depth, strip_prefix_colors,
#     style, mode="standard", target_name=None,
#     per_pane_size=(1000, 950), ss_scale=2,
#     use_depth_peeling=False, transparent_bg=True
# ):
#     """
#     Colors:
#       - depths: by 'shaft_colors_depth' (unique per shaft)
#       - strips/grids: by 'strip_prefix_colors' (two-letter; 'GRID' single color)
#     In 'single_electrode_plots' mode, only target is bright; all others gray.
#     """
#     pl = pv.Plotter(off_screen=True, window_size=per_pane_size)
#     pl.set_background("white")
#     if use_depth_peeling:
#         try: pl.enable_depth_peeling(number_of_peels=120, occlusion_ratio=0.0)
#         except Exception: pass
#     try: pl.enable_anti_aliasing()
#     except Exception: pass
#     add_soft_lights(pl)

#     names = np.array(names); pts = np.asarray(pts)
#     L = np.asarray(isLeft).astype(bool); S = np.asarray(isSubd).astype(bool)

#     # Ipsi-only side panels
#     if show_hemi != "both":
#         keep = (L == (show_hemi == "lh"))
#         names, pts, S = names[keep], pts[keep], S[keep]
#         by_shaft = {s: [(n, p) for (n, p) in seq if n in set(names)] for s, seq in by_shaft.items()}

#     # Brain
#     for hemi in {"lh":[lh], "rh":[rh], "both":[lh, rh]}[show_hemi]:
#         pl.add_mesh(
#             hemi, color=style["brain_color"], opacity=style["brain_opacity"], smooth_shading=True,
#             backface_culling=False, specular=style["specular"], specular_power=style["specular_power"],
#             ambient=style["ambient"], diffuse=style["diffuse"], show_scalar_bar=False
#         )

#     # Depth tubes
#     for shaft, seq in by_shaft.items():
#         if len(seq) < 2: 
#             continue
#         c = shaft_colors_depth.get(shaft, style["other_gray"])
#         if mode == "single_electrode_plots":
#             c = "#b0b0b0"
#         tube = pv.Spline(np.vstack([p for _, p in seq]), n_points=max(50, len(seq)*10)).tube(radius=style["shaft_tube_radius"])
#         pl.add_mesh(tube, color=c)

#     name2shaft = make_name2shaft(by_shaft)
#     target = target_name if (mode == "single_electrode_plots") else None

#     # Contacts
#     for n, p, sd in zip(names, pts, S):
#         if mode == "single_electrode_plots":
#             if target is not None and n == target:
#                 pl.add_mesh(pv.Sphere(radius=style["target_radius"], center=p), color=style["target_color"], opacity=1.0)
#             else:
#                 pl.add_mesh(pv.Sphere(radius=style["depth_radius"], center=p), color=style["other_gray"], opacity=0.85)
#         else:
#             if int(sd) == 1:  # strip/grid
#                 key = two_letter_key(n)  # 'GRID' or 2-letter
#                 c = strip_prefix_colors.get(key, style["other_gray"])
#             else:             # depth
#                 c = shaft_colors_depth.get(name2shaft.get(n, ""), style["other_gray"])
#             pl.add_mesh(pv.Sphere(radius=style["depth_radius"], center=p), color=c, opacity=style["depth_opacity"])

#     pl.camera_position = cam
#     pl.add_text(view_name, position="upper_left", font_size=16, color="black")
#     img = pl.screenshot(return_img=True, scale=ss_scale, transparent_background=transparent_bg)
#     pl.close()
#     return img

# def stitch_three_horiz(imgs):
#     h = min(im.shape[0] for im in imgs)
#     out = []
#     for im in imgs:
#         if im.shape[0] == h: out.append(im)
#         else:
#             dy = (im.shape[0] - h) // 2
#             out.append(im[dy:dy+h, :, :])
#     return np.concatenate(out, axis=1)

# def save_mosaic(path, mosaic, transparent_bg=True):
#     try:
#         import imageio.v3 as iio
#         iio.imwrite(path, mosaic)
#     except Exception:
#         pl = pv.Plotter(off_screen=True, window_size=(mosaic.shape[1], mosaic.shape[0]))
#         tex = pv.numpy_to_texture(mosaic); plane = pv.Plane(i_size=mosaic.shape[1], j_size=mosaic.shape[0])
#         pl.add_mesh(plane, texture=tex); pl.show(auto_close=False)
#         pl.screenshot(path, scale=1, transparent_background=transparent_bg); pl.close()

        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# functions/lf_recon.py
import os, re, numpy as np, nibabel as nib, pyvista as pv

# ---------- paths / geometry ----------
def subj_dir_for(pid: str, root_pat: str, root_micro: str) -> str:
    if pid.startswith("PAT_"):
        return os.path.join(root_pat, pid, "anatomy", "prep", "freesurfer")
    if pid.startswith("MicroEPI"):
        return os.path.join(root_micro, pid, "anatomy", "prep", "freesurfer")
    raise ValueError(f"Unknown patient prefix: {pid}")

def load_fs_surfaces(subj_dir: str):
    import nibabel.freesurfer as nfs
    def faces_to_pv(f): return np.c_[np.full(len(f), 3), f].ravel()
    lh_v, lh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "lh.pial"))
    rh_v, rh_f = nfs.read_geometry(os.path.join(subj_dir, "surf", "rh.pial"))
    return (pv.PolyData(lh_v, faces_to_pv(lh_f)),
            pv.PolyData(rh_v, faces_to_pv(rh_f)))

def safe_compute_normals(mesh: pv.PolyData) -> pv.PolyData:
    try:
        return mesh.compute_normals(consistent_normals=True, auto_orient_normals=True,
                                    feature_angle=180.0, splitting=False)
    except TypeError:
        return mesh.compute_normals(consistent_normals=True, auto_orient_normals=True,
                                    feature_angle=180.0)

# ---------- electrodes (PIAL-only) ----------
def electrode_names_path(subj_dir, pid):  return os.path.join(subj_dir, "elec_recon", f"{pid}.electrodeNames")
def pial_path(subj_dir, pid):              return os.path.join(subj_dir, "elec_recon", f"{pid}.PIAL")

def parse_electrode_names(path):
    names, left, subd = [], [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            row = raw.strip().split()
            if not row: continue
            nm = row[0]; typ = (row[1] if len(row)>1 else "S"); side=(row[2] if len(row)>2 else "R")
            names.append(nm); subd.append(0 if typ.upper().startswith("D") else 1); left.append(1 if side.upper().startswith("L") else 0)
    return np.array(names), np.array(left, np.uint8), np.array(subd, np.uint8)

def load_electrodes_for_subject(subj_dir, pid, vox2ras, vox2ras_tkr, lh, rh,
                                prefer_pial=True, space_override=None, names_skip_first_n=0):
    en, pp = electrode_names_path(subj_dir, pid), pial_path(subj_dir, pid)
    prov = {"pid": pid, "electrodeNames": en, "PIAL": pp, "reason": ""}
    if not (os.path.isfile(en) and os.path.isfile(pp)):
        prov["reason"] = "PIAL bundle missing."
        return {"needs_mgrid_mapping": True, "provenance": prov}

    names, isLeft, isSubd = parse_electrode_names(en)
    if names_skip_first_n:
        keep = np.arange(len(names)) >= int(names_skip_first_n)
        names, isLeft, isSubd = names[keep], isLeft[keep], isSubd[keep]

    xyz = []
    with open(pp, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            t = raw.strip().split()
            if len(t) >= 3:
                try: xyz.append([float(t[0]), float(t[1]), float(t[2])])
                except: pass
    xyz = np.asarray(xyz, float)

    if len(xyz) != len(names):
        prov["reason"] = f"PIAL length ({len(xyz)}) != names length ({len(names)})"
        return {"needs_mgrid_mapping": True, "provenance": prov}

    points_tkr = xyz.copy()
    d2p = distances_to_pial(points_tkr, lh, rh)
    return dict(names=list(names), points_tkr=points_tkr, isLeft=isLeft, isSubdural=isSubd,
                dist_to_pial_mm=d2p, source_space="PIAL", provenance=prov)

def distances_to_pial(pts, lh, rh):
    from scipy.spatial import cKDTree
    all_pts = np.vstack([lh.points, rh.points]); tree = cKDTree(all_pts)
    d, _ = tree.query(np.asarray(pts), k=1); return d

def group_depth_shafts(names, points_tkr, isSubd):
    out = {}
    for n, p, sd in zip(names, points_tkr, isSubd):
        if int(sd) == 1: continue
        m = re.match(r"([A-Za-z]+[A-Za-z]?)[0-9]+", n)
        shaft = m.group(1) if m else n
        out.setdefault(shaft, []).append((n, np.asarray(p, float)))
    return out

def make_name2shaft(by_shaft):
    return {n: shaft for shaft, seq in by_shaft.items() for (n, _p) in seq}

# ---------- coloring ----------
def two_letter_key(name: str) -> str:
    m = re.match(r"([A-Za-z]+)", name); 
    if not m: return "UNK"
    s = m.group(1).upper()
    return "GRID" if s.startswith("G") else (s[:2] if len(s) >= 2 else s)

def build_strip_prefix_colors(strip_names, palette20, grid_color):
    prefixes = {two_letter_key(n) for n in strip_names}
    keys = sorted(prefixes); lut = {}; i = 0
    for k in keys:
        if k == "GRID": lut[k] = grid_color
        else: lut[k] = palette20[i % len(palette20)]; i += 1
    return lut

def build_depth_shaft_colors(shafts, palette20):
    shafts = sorted(list(shafts))
    return {s: palette20[i % len(palette20)] for i, s in enumerate(shafts)}

# ---------- rendering (panes) ----------
def add_soft_lights(pl: pv.Plotter):
    try:
        pl.add_light(pv.Light(position=(300, 300, 400), focal_point=(0, 0, 0), intensity=0.20))
        pl.add_light(pv.Light(position=(-400, 150, 150), focal_point=(0, 0, 0), intensity=0.10))
        pl.add_light(pv.Light(position=(0, -400, 300),  focal_point=(0, 0, 0), intensity=0.15))
    except Exception: pass

def render_single_view_image(
    lh, rh, names, pts, isLeft, isSubd,
    view_name, cam, show_hemi,
    by_shaft, shaft_colors_depth, strip_prefix_colors,
    style, mode="standard", target_name=None,
    per_pane_size=(1000, 950), ss_scale=2,
    use_depth_peeling=False, transparent_bg=True
):
    pl = pv.Plotter(off_screen=True, window_size=per_pane_size)
    pl.set_background("white")
    if use_depth_peeling:
        try: pl.enable_depth_peeling(number_of_peels=120, occlusion_ratio=0.0)
        except Exception: pass
    try: pl.enable_anti_aliasing()
    except Exception: pass
    add_soft_lights(pl)

    names = np.array(names); pts = np.asarray(pts)
    L = np.asarray(isLeft).astype(bool); S = np.asarray(isSubd).astype(bool)

    if show_hemi != "both":
        keep = (L == (show_hemi == "lh"))
        names, pts, S = names[keep], pts[keep], S[keep]
        by_shaft = {s: [(n, p) for (n, p) in seq if n in set(names)] for s, seq in by_shaft.items()}

    for hemi in {"lh":[lh], "rh":[rh], "both":[lh, rh]}[show_hemi]:
        pl.add_mesh(hemi, color=style["brain_color"], opacity=style["brain_opacity"], smooth_shading=True,
                    backface_culling=False, specular=style["specular"], specular_power=style["specular_power"],
                    ambient=style["ambient"], diffuse=style["diffuse"], show_scalar_bar=False)

    for shaft, seq in by_shaft.items():
        if len(seq) < 2: continue
        c = shaft_colors_depth.get(shaft, style["other_gray"])
        if mode == "single_electrode_plots": c = "#b0b0b0"
        tube = pv.Spline(np.vstack([p for _, p in seq]), n_points=max(50, len(seq)*10)).tube(radius=style["shaft_tube_radius"])
        pl.add_mesh(tube, color=c)

    name2shaft = make_name2shaft(by_shaft)
    target = target_name if (mode == "single_electrode_plots") else None

    for n, p, sd in zip(names, pts, S):
        if mode == "single_electrode_plots":
            if target is not None and n == target:
                pl.add_mesh(pv.Sphere(radius=style["target_radius"], center=p), color=style["target_color"], opacity=1.0)
            else:
                pl.add_mesh(pv.Sphere(radius=style["depth_radius"], center=p), color=style["other_gray"], opacity=0.85)
        else:
            if int(sd) == 1:  # strip/grid
                key = two_letter_key(n)
                c = strip_prefix_colors.get(key, style["other_gray"])
            else:             # depth
                c = shaft_colors_depth.get(name2shaft.get(n, ""), style["other_gray"])
            pl.add_mesh(pv.Sphere(radius=style["depth_radius"], center=p), color=c, opacity=style["depth_opacity"])

    pl.camera_position = cam
    pl.add_text(view_name, position="upper_left", font_size=16, color="black")
    img = pl.screenshot(return_img=True, scale=ss_scale, transparent_background=transparent_bg)
    pl.close()
    return img

def stitch_three_horiz(imgs):
    h = min(im.shape[0] for im in imgs)
    out = []
    for im in imgs:
        if im.shape[0] == h: out.append(im)
        else:
            dy = (im.shape[0] - h) // 2
            out.append(im[dy:dy+h, :, :])
    return np.concatenate(out, axis=1)

def save_mosaic(path, mosaic, transparent_bg=True):
    try:
        import imageio.v3 as iio
        iio.imwrite(path, mosaic)
    except Exception:
        pl = pv.Plotter(off_screen=True, window_size=(mosaic.shape[1], mosaic.shape[0]))
        tex = pv.numpy_to_texture(mosaic); plane = pv.Plane(i_size=mosaic.shape[1], j_size=mosaic.shape[0])
        pl.add_mesh(plane, texture=tex); pl.show(auto_close=False)
        pl.screenshot(path, scale=1, transparent_background=transparent_bg); pl.close()

# ---------- legend row (new) ----------
def _hex_to_rgba(c, a=255):
    c = c.lstrip("#"); return (int(c[0:2],16), int(c[2:4],16), int(c[4:6],16), a)

def render_color_legend_row(items, width, height=110):
    """
    items: list[(label, color_hex)]
    returns RGBA numpy image of shape (height, width, 4)
    Tries PIL for labels; falls back to colored chips (no text) if PIL missing.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        try: font = ImageFont.load_default()
        except: font = None
        x, y = 16, height//2
        chip = 20; gap = 18; textgap = 8
        for label, color in items:
            # chip
            draw.rounded_rectangle([x, y-chip//2, x+chip, y+chip//2], radius=4, fill=color)
            x += chip + textgap
            # label
            if font:
                try: tw = draw.textlength(label, font=font)
                except Exception:
                    tw = draw.textbbox((0,0), label, font=font)[2]
                draw.text((x, y-chip//2-2), label, fill=(0,0,0,255), font=font)
            else:
                tw = 8*len(label)
                draw.text((x, y-chip//2-2), label, fill=(0,0,0,255))
            x += tw + gap
            if x > width - 40: break
        return np.array(img, dtype=np.uint8)
    except Exception:
        # fallback: chips only
        img = np.zeros((height, width, 4), dtype=np.uint8)
        n = max(1, len(items)); w = max(20, width // n)
        x = 0
        for _label, color in items:
            img[:, x:x+w, :] = _hex_to_rgba(color, 255)
            x += w
            if x >= width: break
        return img
