import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
import os, tempfile, traceback

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MD Simulation Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-title{font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#00b4d8,#0077b6,#023e8a);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:.2rem;}
.subtitle{color:#6b7280;font-size:1.1rem;font-weight:300;}
.feature-card{background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border-left:4px solid #0077b6;border-radius:12px;padding:1.2rem;margin:.4rem 0;}
.feature-card h4{margin:0 0 .4rem 0;color:#0077b6;}
.feature-card p{margin:0;color:#374151;font-size:.9rem;}
.step-card{background:white;border:1px solid #e5e7eb;border-radius:10px;padding:1rem;margin:.4rem 0;box-shadow:0 2px 8px rgba(0,0,0,.06);}
.step-number{display:inline-block;background:#0077b6;color:white;border-radius:50%;width:28px;height:28px;text-align:center;line-height:28px;font-weight:700;font-size:.85rem;margin-right:.6rem;}
.dock-step{background:#f8fafc;border:2px solid #e2e8f0;border-radius:14px;padding:1.4rem;margin:.8rem 0;}
.dock-step-header{font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:.8rem;}
.result-card{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border-left:4px solid #16a34a;border-radius:10px;padding:1rem;margin:.4rem 0;}
.score-best{color:#16a34a;font-weight:700;font-size:1.2rem;}
.score-good{color:#0077b6;font-weight:600;}
.score-weak{color:#ea580c;font-weight:600;}
.tip-box{background:#fefce8;border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-top:.6rem;font-size:.88rem;color:#78350f;}
.info-box{background:#f0f9ff;border-left:3px solid #0ea5e9;border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-top:.6rem;font-size:.88rem;color:#0c4a6e;}
.warn-box{background:#fff7ed;border-left:3px solid #f97316;border-radius:0 8px 8px 0;padding:.6rem 1rem;font-size:.88rem;color:#7c2d12;}
div[data-testid="stMetricValue"]{font-size:1.4rem!important;font-weight:700!important;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# DOCKING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def check_docking_libs():
    # Only rdkit needed — Vina runs as a downloaded binary (no pip package)
    libs = {'vina': False, 'rdkit': False}
    try:
        from rdkit import Chem; libs['rdkit'] = True
    except Exception: pass
    # Vina binary present if already downloaded
    libs['vina'] = os.path.exists("/tmp/vina_bin")
    libs['all']     = libs['rdkit']   # rdkit is the only pip requirement for docking
    libs['partial'] = libs['rdkit']
    return libs


def get_ligand_centroid(content, ext):
    """Parse coordinates from SDF/MOL2/PDB and return centroid [x,y,z]"""
    coords = []
    try:
        if ext in ('sdf', 'mol'):
            lines = content.split('\n')
            if len(lines) < 4: return [0.0,0.0,0.0]
            cl = lines[3]
            try:   n_atoms = int(cl[:3].strip())
            except: return [0.0,0.0,0.0]
            for i in range(4, 4 + n_atoms):
                if i < len(lines):
                    p = lines[i].split()
                    if len(p) >= 3:
                        coords.append([float(p[0]), float(p[1]), float(p[2])])
        elif ext == 'pdb':
            for line in content.split('\n'):
                if line.startswith(('ATOM','HETATM')):
                    try: coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                    except: pass
        elif ext == 'mol2':
            in_atom = False
            for line in content.split('\n'):
                if '@<TRIPOS>ATOM' in line: in_atom = True; continue
                if '@<TRIPOS>' in line and 'ATOM' not in line: in_atom = False
                if in_atom:
                    p = line.split()
                    if len(p) >= 5:
                        try: coords.append([float(p[2]), float(p[3]), float(p[4])])
                        except: pass
    except Exception: pass
    if coords:
        arr = np.array(coords)
        return arr.mean(axis=0).round(3).tolist()
    return [0.0, 0.0, 0.0]


def prepare_receptor_pdbqt(pdb_path, out_path):
    """
    Pure Python PDB → PDBQT for AutoDock Vina.
    Key rules:
      - Remove ALL hydrogen atoms (standard Vina receptor prep)
      - Remove water molecules
      - Add AutoDock atom type + zero charge in correct PDBQT columns
    """
    AD = {
        'C':'C',  'N':'N',  'O':'OA', 'S':'SA',
        'P':'P',  'F':'F',  'CL':'Cl','BR':'Br','I':'I',
        'FE':'Fe','ZN':'Zn','MG':'Mg','CA':'Ca','MN':'Mn',
        'NA':'Na','K':'K',  'CU':'Cu',
    }
    out_lines = []
    with open(pdb_path) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ('ATOM', 'HETATM'):
                if rec == 'TER':
                    out_lines.append('TER\n')
                continue
            if len(line) < 54:
                continue

            resname = line[17:20].strip()
            if resname in ('HOH','WAT','SOL'):
                continue

            # Determine element symbol
            element = line[76:78].strip().upper() if len(line) > 76 else ''
            if not element:
                aname = line[12:16].strip().lstrip('0123456789')
                raw   = ''.join(c for c in aname if c.isalpha())
                element = raw[:2].upper() if raw[:2].upper() in AD else raw[:1].upper()

            # *** Skip all hydrogen atoms — Vina doesn't want them on receptor ***
            if element in ('H', 'D'):
                continue

            ad_type = AD.get(element, 'C')

            # PDBQT format: cols 1-66 (standard PDB) + cols 67-76 (charge 10.3f) + col 77+ (type)
            pdb66 = line.rstrip('\n')[:66].ljust(66)
            out_lines.append(f"{pdb66}{0.0:>10.3f} {ad_type}\n")

    with open(out_path, 'w') as f:
        f.writelines(out_lines)

    n_heavy = sum(1 for l in out_lines if l.startswith(('ATOM','HETATM')))
    return n_heavy > 0


def rdkit_mol_to_pdbqt(mol):
    """
    RDKit mol → valid PDBQT with BRANCH/ENDBRANCH torsion records.
    Pure Python — no meeko or openbabel required.
    """
    import sys
    from rdkit import Chem
    from rdkit.Chem import AllChem
    sys.setrecursionlimit(10000)

    mol = Chem.AddHs(mol)

    # Generate 3D conformer if missing
    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        if AllChem.EmbedMolecule(mol, params) == -1:
            AllChem.EmbedMolecule(mol, randomSeed=42)   # fallback
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    except Exception:
        pass

    if mol.GetNumConformers() == 0:
        raise ValueError("RDKit could not generate 3D coordinates for this ligand.")

    # AutoDock atom types
    AD = {'C':'C','N':'NA','O':'OA','S':'SA','H':'HD',
          'F':'F','Cl':'Cl','Br':'Br','I':'I','P':'P'}
    conf = mol.GetConformer()

    # Rotatable bonds: single, non-ring, both endpoints heavy atoms
    rot_bonds = set()
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if (b.GetBondTypeAsDouble() == 1.0 and not b.IsInRing()
                and a1.GetSymbol() != 'H' and a2.GetSymbol() != 'H'):
            rot_bonds.add(frozenset([a1.GetIdx(), a2.GetIdx()]))

    atom_num, atom_map, visited, lines = [0], {}, set(), []

    def fmt(idx):
        at  = mol.GetAtomWithIdx(idx)
        p   = conf.GetAtomPosition(idx)
        sym = at.GetSymbol()
        atom_num[0] += 1
        atom_map[idx] = atom_num[0]
        n     = atom_num[0]
        aname = f"{sym}{n}"[:4].ljust(4)
        t     = AD.get(sym, 'C')
        # Exact PDBQT HETATM column layout:
        # 1-6 record, 7-11 serial, 12 space, 13-16 name, 17-30 resinfo,
        # 31-38 x, 39-46 y, 47-54 z, 55-60 occ, 61-66 bfac,
        # 67-76 charge (10.3f), 77 space, 78-79 AD type
        return (f"HETATM{n:5d} {aname} LIG     1    "
                f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00"
                f"{0.0:>10.3f} {t}")

    def dfs(idx, parent):
        visited.add(idx)
        lines.append(fmt(idx))
        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
            nidx = nb.GetIdx()
            if nidx in visited:
                continue
            if frozenset([idx, nidx]) in rot_bonds:
                pnum = atom_map[idx]
                bpos = len(lines)
                lines.append("__BRANCH__")          # placeholder
                dfs(nidx, idx)
                lines[bpos] = f"BRANCH {pnum} {atom_map[nidx]}"
                lines.append(f"ENDBRANCH {pnum} {atom_map[nidx]}")
            else:
                dfs(nidx, idx)

    root = next((i for i in range(mol.GetNumAtoms())
                 if mol.GetAtomWithIdx(i).GetSymbol() != 'H'), 0)
    lines.append('ROOT')
    dfs(root, -1)
    lines.append('ENDROOT')
    lines.append(f'TORSDOF {len(rot_bonds)}')
    return '\n'.join(lines)


def prepare_ligand_pdbqt(lig_path, out_path, ext):
    """Convert ligand (SDF/MOL/MOL2/PDB) → PDBQT using rdkit only."""
    from rdkit import Chem

    mol = None
    if ext in ('sdf', 'mol'):
        mol = Chem.MolFromMolFile(lig_path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMolFile(lig_path)
    elif ext == 'mol2':
        mol = Chem.MolFromMol2File(lig_path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromMol2File(lig_path)
    elif ext == 'pdb':
        mol = Chem.MolFromPDBFile(lig_path, removeHs=False)
        if mol is None:
            mol = Chem.MolFromPDBFile(lig_path)

    if mol is None:
        return False

    pdbqt_str = rdkit_mol_to_pdbqt(mol)
    with open(out_path, 'w') as f:
        f.write(pdbqt_str)
    return os.path.exists(out_path) and os.path.getsize(out_path) > 0


@st.cache_resource(show_spinner=False)
def get_vina_binary():
    """
    Download AutoDock Vina 1.2.5 Linux binary from GitHub at runtime.
    Cached so it only downloads once per session.
    Returns (path, error_message).
    """
    import urllib.request, stat

    vina_path = "/tmp/vina_bin"
    url = ("https://github.com/ccsb-scripps/AutoDock-Vina/releases/"
           "download/v1.2.5/vina_1.2.5_linux_x86_64")

    if os.path.exists(vina_path) and os.path.getsize(vina_path) > 1_000_000:
        return vina_path, None

    try:
        urllib.request.urlretrieve(url, vina_path)
        os.chmod(vina_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH)
        return vina_path, None
    except Exception as e:
        return None, str(e)


def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, box_size,
                     exhaustiveness, n_poses, seed):
    """
    Run AutoDock Vina via downloaded binary (subprocess).
    Returns (energies_list, output_pdbqt_path).
    """
    import subprocess

    vina_bin, err = get_vina_binary()
    if not vina_bin:
        raise RuntimeError(f"Could not download Vina binary: {err}")

    out_path = ligand_pdbqt.replace("ligand.pdbqt", "docked.pdbqt")

    cmd = [
        vina_bin,
        "--receptor", receptor_pdbqt,
        "--ligand",   ligand_pdbqt,
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x",   f"{box_size[0]:.1f}",
        "--size_y",   f"{box_size[1]:.1f}",
        "--size_z",   f"{box_size[2]:.1f}",
        "--exhaustiveness", str(int(exhaustiveness)),
        "--num_modes",      str(int(n_poses)),
        "--seed",           str(int(seed)),
        "--out",            out_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=360)
    if result.returncode != 0:
        raise RuntimeError(f"Vina error:\n{result.stderr or result.stdout}")

    # Parse affinity table from stdout
    energies = []
    for line in result.stdout.split('\n'):
        parts = line.split()
        if len(parts) >= 4:
            try:
                int(parts[0])          # mode number
                energies.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                pass

    return energies, out_path




def parse_pdbqt_poses(pdbqt_content):
    """Split PDBQT multi-pose file into individual pose strings"""
    poses, current = [], []
    for line in pdbqt_content.split('\n'):
        current.append(line)
        if line.startswith('ENDMDL') or line.startswith('END'):
            if any(l.startswith(('ATOM','HETATM')) for l in current):
                poses.append('\n'.join(current))
            current = []
    if current and any(l.startswith(('ATOM','HETATM')) for l in current):
        poses.append('\n'.join(current))
    return poses if poses else [pdbqt_content]


# ═══════════════════════════════════════════════════════════════════════════════
# XVG PARSER (reused across analysis pages)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_xvg(uploaded_file):
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        return None, "X", "Y", "Plot", []
    lines = content.split("\n")
    data, legends = [], []
    x_label, y_label, title = "Time (ps)", "Value", "GROMACS Output"
    for line in lines:
        line = line.strip()
        if line.startswith("#"): continue
        elif line.startswith("@"):
            if 'xaxis  label' in line and '"' in line: x_label = line.split('"')[1]
            elif 'yaxis  label' in line and '"' in line: y_label = line.split('"')[1]
            elif line.startswith("@ title") and '"' in line: title = line.split('"')[1]
            elif "legend" in line and '"' in line: legends.append(line.split('"')[1])
        else:
            try:
                vals = list(map(float, line.split()))
                if vals: data.append(vals)
            except: pass
    if not data: return None, x_label, y_label, title, legends
    return pd.DataFrame(data), x_label, y_label, title, legends


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MOLECULAR DOCKING
# ═══════════════════════════════════════════════════════════════════════════════

def page_docking():
    st.markdown('<div class="main-title">🎯 Molecular Docking</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload protein + ligand → define binding site → run AutoDock Vina</div>', unsafe_allow_html=True)
    st.markdown("")

    # ── Library status banner ─────────────────────────────────────────────────
    libs = check_docking_libs()
    if libs['all']:
        st.success("✅ All docking libraries detected — **real AutoDock Vina docking** is active.")
    elif libs['partial']:
        missing = [k for k,v in libs.items() if k not in ('all','partial') and not v]
        st.warning(f"⚠️ Partial install ({', '.join(missing)} missing) — will run in **preparation mode** and show you the commands to run locally.")
    else:
        st.info("ℹ️ Docking libraries not installed — running in **preparation mode**. Add them to `requirements.txt` for real docking.")

    with st.expander("📦 How docking works on Streamlit Cloud"):
        st.markdown("""
- **RDKit** (pip package) — reads your ligand file and generates 3D PDBQT
- **AutoDock Vina binary** — downloaded automatically from GitHub (~5 MB) on first run, then cached
- **Receptor prep** — pure Python, no extra packages needed
- No `openbabel`, `meeko`, or `vina` pip packages required ✅
        """)
        st.code("# requirements.txt — only this needed:\nstreamlit\npandas\nnumpy\nplotly\nrdkit", language="bash")
        icon_r = "✅" if libs['rdkit'] else "❌"
        icon_v = "✅" if libs['vina']  else "⏳ (downloads when you click Run)"
        st.markdown(f"{icon_r} **RDKit** installed  |  {icon_v} **Vina binary**")

    st.markdown("---")

    # ── STEP 1: Upload files ──────────────────────────────────────────────────
    st.markdown('<div class="dock-step"><div class="dock-step-header">📁 Step 1 — Upload Protein & Ligand</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🧬 Protein (Receptor)**")
        st.caption("Crystal structure or homology model — PDB format")
        prot_file = st.file_uploader("Upload Protein", type=['pdb'], label_visibility="collapsed")
        if prot_file:
            prot_content = prot_file.read().decode('utf-8', errors='ignore')
            prot_file.seek(0)
            atom_lines = [l for l in prot_content.split('\n') if l.startswith('ATOM')]
            het_lines  = [l for l in prot_content.split('\n') if l.startswith('HETATM') and 'HOH' not in l]
            water_lines= [l for l in prot_content.split('\n') if 'HOH' in l]
            st.success(f"✅ **{prot_file.name}**")
            c1,c2,c3 = st.columns(3)
            c1.metric("ATOM records", len(atom_lines))
            c2.metric("HETATM", len(het_lines))
            c3.metric("Waters", len(water_lines))
        else:
            prot_content = None
            st.markdown('<div class="info-box">Upload your receptor PDB file. Tip: remove all waters and co-factors before uploading.</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("**💊 Ligand (Drug Molecule)**")
        st.caption("Supports SDF, MOL2, MOL, PDB formats")
        lig_file = st.file_uploader("Upload Ligand", type=['sdf','mol2','mol','pdb'], label_visibility="collapsed")
        if lig_file:
            lig_ext = lig_file.name.rsplit('.',1)[-1].lower()
            lig_content = lig_file.read().decode('utf-8', errors='ignore')
            lig_file.seek(0)
            st.success(f"✅ **{lig_file.name}** (.{lig_ext})")
            centroid = get_ligand_centroid(lig_content, lig_ext)
            st.info(f"📍 Ligand centroid detected: X={centroid[0]:.2f}, Y={centroid[1]:.2f}, Z={centroid[2]:.2f}")
            st.caption("Use these as box center coordinates below ↓")
        else:
            lig_content, lig_ext, centroid = None, None, [0.0, 0.0, 0.0]
            st.markdown('<div class="info-box">Upload your drug molecule. SDF is recommended for best compatibility.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 2: 3D Preview ────────────────────────────────────────────────────
    if prot_content or lig_content:
        st.markdown('<div class="dock-step"><div class="dock-step-header">🔬 Step 2 — 3D Structure Preview</div>', unsafe_allow_html=True)

        preview_cols = st.columns(2)
        if prot_content:
            with preview_cols[0]:
                st.markdown("**Protein**")
                _show_3dmol(prot_content, style="cartoon", color="spectrum", height=320, label="protein")
        if lig_content:
            with preview_cols[1]:
                st.markdown("**Ligand**")
                lig_pdb_for_view = lig_content if lig_ext == 'pdb' else _sdf_to_viewer_html(lig_content, lig_ext)
                if lig_pdb_for_view:
                    _show_3dmol(lig_pdb_for_view, style="stick", color="element", height=320, label="ligand")
                else:
                    st.info("3D preview available for PDB ligands. Molecule loaded for docking.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 3: Binding site box ──────────────────────────────────────────────
    st.markdown('<div class="dock-step"><div class="dock-step-header">📦 Step 3 — Define Binding Site Search Box</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ How to find binding site coordinates"):
        st.markdown("""
**Option A — Use ligand centroid (auto-detected above):** If you uploaded a reference ligand or co-crystallized drug, its centroid is shown above. Use those values.

**Option B — From PyMOL:**
```python
# Select binding site residues and get center
select site, resi 101+203+305
center site
get_position   # shows X Y Z of current view center
```

**Option C — From UCSF Chimera:**
- Select residues → Actions → Inspect → shows coordinates

**Box size tip:** Start with 20×20×20 Å. Increase if ligand is large or binding site is flexible.
        """)

    cx_default = float(centroid[0]) if lig_content else 0.0
    cy_default = float(centroid[1]) if lig_content else 0.0
    cz_default = float(centroid[2]) if lig_content else 0.0

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        st.markdown("**Box Center (Å)**")
        c1,c2,c3 = st.columns(3)
        cx = c1.number_input("X", value=cx_default, step=0.5, key="bx", format="%.2f")
        cy = c2.number_input("Y", value=cy_default, step=0.5, key="by", format="%.2f")
        cz = c3.number_input("Z", value=cz_default, step=0.5, key="bz", format="%.2f")
    with bcol2:
        st.markdown("**Box Size (Å)**")
        s1,s2,s3 = st.columns(3)
        sx = s1.number_input("X", value=20.0, min_value=10.0, max_value=40.0, step=1.0, key="sx")
        sy = s2.number_input("Y", value=20.0, min_value=10.0, max_value=40.0, step=1.0, key="sy")
        sz = s3.number_input("Z", value=20.0, min_value=10.0, max_value=40.0, step=1.0, key="sz")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 4: Settings ──────────────────────────────────────────────────────
    st.markdown('<div class="dock-step"><div class="dock-step-header">⚙️ Step 4 — Docking Settings</div>', unsafe_allow_html=True)
    scol1, scol2, scol3, scol4 = st.columns(4)
    with scol1:
        exhaustiveness = st.slider("Exhaustiveness", 1, 16, 4,
            help="Higher = more accurate but slower. 4 = fast (~2 min), 8 = standard (~5 min)")
    with scol2:
        n_poses = st.slider("Number of poses", 1, 9, 5)
    with scol3:
        seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    with scol4:
        st.markdown("")
        st.markdown("")
        st.markdown(f"⏱️ Est. time: ~**{exhaustiveness*30//60 + 1} min** on free cloud")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 5: RUN ───────────────────────────────────────────────────────────
    st.markdown("---")
    btn_disabled = not (prot_content and lig_content)

    if btn_disabled:
        st.info("👆 Upload both protein and ligand files to enable the docking button.")
    else:
        col_run, col_mode = st.columns([2,1])
        with col_run:
            run_btn = st.button("🚀 Run AutoDock Vina Docking", type="primary", use_container_width=True)
        with col_mode:
            force_prep = st.checkbox("Force preparation mode only", value=False,
                help="Skip real docking — just prepare files and show commands")

        if run_btn:
            if libs['all'] and not force_prep:
                _run_real_docking(
                    prot_file, lig_file, lig_ext,
                    [cx, cy, cz], [sx, sy, sz],
                    exhaustiveness, n_poses, seed,
                    prot_content
                )
            else:
                _run_prep_mode(prot_content, lig_content, lig_ext,
                               [cx,cy,cz], [sx,sy,sz], exhaustiveness, n_poses)


def _show_3dmol(pdb_content, style="cartoon", color="spectrum", height=350, label="mol"):
    escaped = pdb_content.replace("`", "\\`").replace("\\", "\\\\")
    # Build JS style object as a plain string — no % operator (conflicts with CSS %)
    if style == "cartoon":
        style_js = f"{{cartoon:{{color:'{color}'}}}}"
    elif style == "stick":
        style_js = "{stick:{}}"
    elif style == "sphere":
        style_js = "{sphere:{radius:0.4}}"
    else:
        style_js = "{stick:{colorscheme:'elementColors'}}"
    html = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>body{{margin:0;background:#0d1b2a;}}#v{{width:100%;height:{height}px;}}</style>
</head><body><div id="v"></div>
<script>
var v=$3Dmol.createViewer("v",{{backgroundColor:"0x0d1b2a"}});
v.addModel(`{escaped}`,"pdb");
v.setStyle({{}},{style_js});
v.zoomTo();v.render();
</script></body></html>"""
    components.html(html, height=height + 10, scrolling=False)


def _sdf_to_viewer_html(content, ext):
    """For non-PDB ligands, try to show using 3dmol SDF support"""
    if ext in ('sdf','mol'):
        return content  # 3dmol can handle SDF directly with format hint
    return None


def _run_real_docking(prot_file, lig_file, lig_ext, center, box_size, exhaustiveness, n_poses, seed, prot_content):
    """Run actual AutoDock Vina docking via downloaded binary"""
    progress = st.empty()

    try:
        with tempfile.TemporaryDirectory() as tmp:
            prot_pdb   = os.path.join(tmp, "receptor.pdb")
            prot_pdbqt = os.path.join(tmp, "receptor.pdbqt")
            lig_in     = os.path.join(tmp, f"ligand.{lig_ext}")
            lig_pdbqt  = os.path.join(tmp, "ligand.pdbqt")

            prot_file.seek(0)
            with open(prot_pdb, 'wb') as f: f.write(prot_file.read())
            lig_file.seek(0)
            with open(lig_in,  'wb') as f: f.write(lig_file.read())

            # ── Validate box vs receptor bounding box ─────────────────────────
            prot_coords = []
            for line in prot_content.split('\n'):
                if line.startswith('ATOM') and len(line) >= 54:
                    try:
                        prot_coords.append([float(line[30:38]),
                                            float(line[38:46]),
                                            float(line[46:54])])
                    except Exception:
                        pass
            if prot_coords:
                pc = np.array(prot_coords)
                pmin, pmax = pc.min(axis=0), pc.max(axis=0)
                cx, cy, cz = center
                in_box = (pmin[0]-10 < cx < pmax[0]+10 and
                          pmin[1]-10 < cy < pmax[1]+10 and
                          pmin[2]-10 < cz < pmax[2]+10)
                if not in_box:
                    progress.empty()
                    st.error(
                        f"❌ **Box center ({cx:.1f}, {cy:.1f}, {cz:.1f}) is outside the protein!**\n\n"
                        f"Protein bounding box: "
                        f"X {pmin[0]:.1f}→{pmax[0]:.1f}, "
                        f"Y {pmin[1]:.1f}→{pmax[1]:.1f}, "
                        f"Z {pmin[2]:.1f}→{pmax[2]:.1f}\n\n"
                        "💡 **Fix:** Upload your co-crystallized ligand so the app auto-detects the box center. "
                        "Or use the midpoint of the protein bounding box above."
                    )
                    auto_cx = float((pmin[0]+pmax[0])/2)
                    auto_cy = float((pmin[1]+pmax[1])/2)
                    auto_cz = float((pmin[2]+pmax[2])/2)
                    st.info(f"Suggested center: X={auto_cx:.2f}, Y={auto_cy:.2f}, Z={auto_cz:.2f}")
                    return

            # ── Step 1: Download Vina binary ───────────────────────────────────
            progress.info("⬇️ **Step 1/5** — Fetching AutoDock Vina binary (cached after first use)...")
            vina_bin, dl_err = get_vina_binary()
            if not vina_bin:
                progress.error(f"❌ Could not download Vina binary: {dl_err}")
                return

            # ── Step 2: Prepare receptor ───────────────────────────────────────
            progress.info("⚙️ **Step 2/5** — Preparing receptor (removing H + waters → PDBQT)...")
            if not prepare_receptor_pdbqt(prot_pdb, prot_pdbqt):
                progress.error("❌ Receptor has no heavy atoms after preparation. Check your PDB file.")
                return
            rec_n = sum(1 for l in open(prot_pdbqt) if l.startswith(('ATOM','HETATM')))

            # ── Step 3: Prepare ligand ─────────────────────────────────────────
            progress.info("⚙️ **Step 3/5** — Preparing ligand (3D conformer + PDBQT)...")
            if not prepare_ligand_pdbqt(lig_in, lig_pdbqt, lig_ext):
                progress.error("❌ Ligand preparation failed. Try SDF format with valid 3D coordinates.")
                return
            lig_n = sum(1 for l in open(lig_pdbqt) if l.startswith('HETATM'))

            # ── Step 4: Run Vina ───────────────────────────────────────────────
            progress.warning(
                f"🚀 **Step 4/5** — Running AutoDock Vina "
                f"(receptor {rec_n} atoms, ligand {lig_n} atoms, "
                f"exhaustiveness={exhaustiveness})… Please wait ~{exhaustiveness+1} min."
            )
            energies, docked_path = run_vina_docking(
                prot_pdbqt, lig_pdbqt, center, box_size, exhaustiveness, n_poses, seed
            )

            # ── Step 5: Results ────────────────────────────────────────────────
            progress.success("✅ **Step 5/5** — Docking complete!")
            with open(docked_path) as f: docked_content = f.read()
            with open(prot_pdbqt)  as f: receptor_content = f.read()
            with open(lig_pdbqt)   as f: ligand_prep_content = f.read()
            progress.empty()
            _display_docking_results(energies, docked_content, prot_content,
                                     receptor_content, ligand_prep_content, center, box_size)

    except Exception as e:
        progress.empty()
        st.error(f"❌ Docking failed: {str(e)}")
        with st.expander("🔍 Full error details"):
            st.code(traceback.format_exc())
        st.info("💡 Switching to **preparation mode** — you can run locally with the commands below.")
        prot_file.seek(0)
        lig_file.seek(0)
        pc = prot_file.read().decode('utf-8', errors='ignore')
        lc = lig_file.read().decode('utf-8', errors='ignore')
        _run_prep_mode(pc, lc, lig_ext, center, box_size, exhaustiveness, n_poses)


def _display_docking_results(energies, docked_content, prot_content,
                              receptor_content, ligand_prep_content, center, box_size):
    st.markdown("---")
    st.markdown("## 🏆 Docking Results")

    # ── Scores table ──
    st.markdown("### Binding Affinity Scores")
    if energies is not None and len(energies) > 0:
        rows = []
        for i, e in enumerate(energies):
            score = float(e[0]) if hasattr(e,'__len__') else float(e)
            rows.append({
                "Pose": i+1,
                "Binding Affinity (kcal/mol)": round(score, 2),
                "RMSD lb": round(float(e[1]), 2) if hasattr(e,'__len__') and len(e)>1 else "—",
                "RMSD ub": round(float(e[2]), 2) if hasattr(e,'__len__') and len(e)>2 else "—",
                "Rating": "⭐ Best" if i==0 else ("✅ Good" if score < -7 else ("⚠️ Moderate" if score < -5 else "❌ Weak")),
            })
        df_results = pd.DataFrame(rows)
        st.dataframe(df_results, use_container_width=True, hide_index=True)

        best_score = rows[0]["Binding Affinity (kcal/mol)"]
        c1,c2,c3 = st.columns(3)
        c1.metric("Best Binding Affinity", f"{best_score} kcal/mol")
        c2.metric("Total Poses", len(rows))
        c3.metric("Poses < -7 kcal/mol", sum(1 for r in rows if r["Binding Affinity (kcal/mol)"] < -7))

        if best_score < -9:
            st.success(f"🏆 **Excellent binding** ({best_score} kcal/mol) — Strong drug candidate! Proceed to MD simulation.")
        elif best_score < -7:
            st.success(f"✅ **Good binding** ({best_score} kcal/mol) — Promising compound. Recommend MD validation.")
        elif best_score < -5:
            st.warning(f"⚠️ **Moderate binding** ({best_score} kcal/mol) — Consider structural optimization.")
        else:
            st.error(f"❌ **Weak binding** ({best_score} kcal/mol) — Try a different compound or adjust binding site.")

        # Bar chart
        fig = go.Figure(go.Bar(
            x=[f"Pose {r['Pose']}" for r in rows],
            y=[abs(r["Binding Affinity (kcal/mol)"]) for r in rows],
            marker_color=["#16a34a" if i==0 else "#0077b6" for i in range(len(rows))],
            text=[f"{r['Binding Affinity (kcal/mol)']} kcal/mol" for r in rows],
            textposition='outside',
        ))
        fig.update_layout(
            title="Binding Affinity per Pose (larger bar = stronger binding)",
            yaxis_title="|Binding Affinity| (kcal/mol)",
            template="plotly_white", height=320,
            yaxis=dict(autorange=True)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not parse energy scores from output.")

    # ── 3D Viewer: protein + best docked pose ──
    st.markdown("### 🔬 3D View — Best Docked Pose")
    poses = parse_pdbqt_poses(docked_content)
    best_pose_pdbqt = poses[0] if poses else docked_content

    # Build combined PDB for viewing: protein + docked ligand
    combined = prot_content + "\n" + best_pose_pdbqt
    escaped = combined.replace("`","\\`").replace("\\","\\\\")
    box_js = f"""
    viewer.addBox({{
        center:{{x:{center[0]},y:{center[1]},z:{center[2]}}},
        dimensions:{{w:{box_size[0]},h:{box_size[1]},d:{box_size[2]}}},
        color:'yellow', opacity:0.15, wireframe:true
    }});
    """
    html_view = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>body{{margin:0;background:#0d1b2a;}}#v{{width:100%;height:500px;}}</style>
</head><body><div id="v"></div>
<script>
var v=$3Dmol.createViewer("v",{{backgroundColor:"0x0d1b2a"}});
v.addModel(`{escaped}`,"pdb");
v.setStyle({{}},"{{cartoon:{{color:'spectrum'}}}}");
v.setStyle({{hetflag:true}},{{stick:{{}}}});
v.setStyle({{hetflag:true,resn:"HOH"}},{{sphere:{{radius:0.15}}}});
{box_js}
v.zoomTo({{hetflag:true}});
v.render();
</script></body></html>"""
    # Fix the JS object literal (Python f-string escaping issue)
    html_view = html_view.replace('"{{cartoon:{{color:\'spectrum\'}}}}"', "{cartoon:{color:'spectrum'}}")
    components.html(html_view, height=520)
    st.caption("🟡 Yellow box = docking search space | Protein shown as cartoon | Ligand as sticks")

    # ── Downloads ──
    st.markdown("### 📥 Download Results")
    dl1, dl2, dl3 = st.columns(3)
    dl1.download_button("⬇️ Docked Poses (.pdbqt)", docked_content,
                        "docked_poses.pdbqt", "chemical/x-pdbqt")
    dl2.download_button("⬇️ Prepared Receptor (.pdbqt)", receptor_content,
                        "receptor.pdbqt", "chemical/x-pdbqt")
    dl3.download_button("⬇️ Prepared Ligand (.pdbqt)", ligand_prep_content,
                        "ligand_prepared.pdbqt", "chemical/x-pdbqt")

    # ── Next step ──
    st.markdown("---")
    st.markdown("### ➡️ Next Steps: MD Simulation")
    st.info("""
**Your docking is done! Here's what to do next:**

1. Download the **docked poses** above
2. Open in **PyMOL** or **Chimera** — visually inspect the best pose
3. Verify key interactions match literature (H-bonds, hydrophobic contacts)
4. Use the **📚 MD Workflow Guide** in this app to set up GROMACS MD simulation
5. Upload your MD results (.xvg files) to the analysis pages in this app

**Key quality check:** Is the ligand in the expected binding pocket? Does it match known inhibitors?
    """)


def _run_prep_mode(prot_content, lig_content, lig_ext, center, box_size, exhaustiveness, n_poses):
    """Preparation mode — prepare files and show commands"""
    st.markdown("---")
    st.markdown("## 📦 Preparation Mode")
    st.info("⚠️ **Docking ran into an error** — showing preparation mode as fallback. "
            "The files below are ready; scroll up to see the exact error and fix it.")

    cx, cy, cz = center
    sx, sy, sz = box_size

    # Protein info
    atom_count = sum(1 for l in prot_content.split('\n') if l.startswith('ATOM'))
    st.success(f"✅ Protein file valid — {atom_count} ATOM records found")

    # Ligand info
    cent = get_ligand_centroid(lig_content, lig_ext)
    st.success(f"✅ Ligand file valid (.{lig_ext}) — centroid at {cent[0]:.2f}, {cent[1]:.2f}, {cent[2]:.2f}")

    st.markdown("### 🖥️ Commands to Run Locally")
    st.markdown("**Step 1 — Install tools (once):**")
    st.code("""# Install AutoDock Vina + preparation tools
pip install vina meeko rdkit openbabel-wheel

# Or via conda (recommended):
conda install -c conda-forge vina meeko rdkit openbabel""", language="bash")

    st.markdown("**Step 2 — Prepare receptor:**")
    st.code("""# Option A: Python
from openbabel import pybel
mol = next(pybel.readfile("pdb", "protein.pdb"))
mol.write("pdbqt", "receptor.pdbqt", overwrite=True)

# Option B: Command line (if AutoDockTools installed)
python prepare_receptor4.py -r protein.pdb -o receptor.pdbqt -A hydrogens""", language="python")

    st.markdown("**Step 3 — Prepare ligand:**")
    if lig_ext in ('sdf','mol'):
        st.code(f"""from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation

mol = Chem.MolFromMolFile("ligand.{lig_ext}")
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

prep = MoleculePreparation()
prep.prepare(mol)
prep.write_pdbqt_file("ligand.pdbqt")""", language="python")
    elif lig_ext == 'mol2':
        st.code("""from openbabel import pybel
mol = next(pybel.readfile("mol2", "ligand.mol2"))
mol.addh()
mol.make3D()
mol.write("pdbqt", "ligand.pdbqt", overwrite=True)""", language="python")
    else:
        st.code("""acpype -i ligand.pdb -c bcc -n 0
# OR
obabel ligand.pdb -O ligand.pdbqt -xr""", language="bash")

    st.markdown("**Step 4 — Run AutoDock Vina:**")
    st.code(f"""from vina import Vina

v = Vina(sf_name='vina')
v.set_receptor("receptor.pdbqt")
v.set_ligand_from_file("ligand.pdbqt")
v.compute_vina_maps(
    center=[{cx:.2f}, {cy:.2f}, {cz:.2f}],
    box_size=[{sx:.1f}, {sy:.1f}, {sz:.1f}]
)
v.dock(exhaustiveness={exhaustiveness}, n_poses={n_poses})
v.write_poses("docked.pdbqt", n_poses={n_poses}, overwrite=True)

# Print scores
energies = v.energies()
for i, e in enumerate(energies):
    print(f"Pose {{i+1}}: {{e[0]:.2f}} kcal/mol")""", language="python")

    # Config file option
    st.markdown("**Alternative — Vina config file:**")
    config_str = f"""receptor = receptor.pdbqt
ligand = ligand.pdbqt

center_x = {cx:.2f}
center_y = {cy:.2f}
center_z = {cz:.2f}

size_x = {sx:.1f}
size_y = {sy:.1f}
size_z = {sz:.1f}

exhaustiveness = {exhaustiveness}
num_modes = {n_poses}
energy_range = 3
out = docked.pdbqt
log = docking_log.txt
"""
    st.code(f"""# Run from terminal:
vina --config config.txt""", language="bash")

    col1, col2 = st.columns(2)
    col1.download_button("⬇️ Download config.txt", config_str, "config.txt", "text/plain")
    col2.download_button("⬇️ Download Protein PDB", prot_content, "protein_clean.pdb", "chemical/x-pdb")

    st.markdown("### 📊 Interpreting Results")
    interp = pd.DataFrame({
        "Score (kcal/mol)": ["< −9", "−7 to −9", "−5 to −7", "> −5"],
        "Interpretation": ["🏆 Excellent", "✅ Good", "⚠️ Moderate", "❌ Weak"],
        "Action": ["Proceed to MD", "Proceed to MD", "Optimize ligand", "Redesign compound"],
    })
    st.dataframe(interp, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.markdown('<div class="main-title">🧬 MD Simulation Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">From molecular docking to MD simulation analysis — all in one place</div>', unsafe_allow_html=True)
    st.markdown("")

    c1,c2,c3,c4 = st.columns(4)
    for col, (title, desc) in zip([c1,c2,c3,c4],[
        ("🎯 Dock","Upload protein + drug and run AutoDock Vina directly in the browser"),
        ("📚 Learn","Step-by-step GROMACS workflow with commands for every stage"),
        ("📊 Analyze","Upload .xvg files for RMSD, RMSF, H-bond and MM-PBSA plots"),
        ("🔬 Visualize","Interactive 3D viewer for protein-ligand PDB structures"),
    ]):
        with col:
            st.markdown(f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Full Pipeline")
    steps = [
        ("1","Molecular Docking","This app — AutoDock Vina"),
        ("2","Best Pose Selection","PyMOL / Chimera"),
        ("3","Ligand Topology","ACPYPE / CGenFF"),
        ("4","Protein Topology","pdb2gmx"),
        ("5","Solvation","editconf + genion"),
        ("6","Energy Minimization","gmx mdrun"),
        ("7","NVT Equilibration","300K, 200 ps"),
        ("8","NPT Equilibration","1 bar, 200 ps"),
        ("9","Production MD","50-200 ns"),
        ("10","RMSD/RMSF","This app"),
        ("11","H-Bond Analysis","This app"),
        ("12","MM-PBSA","This app + gmx_MMPBSA"),
    ]
    cols = st.columns(4)
    for i,(num,name,detail) in enumerate(steps):
        with cols[i%4]:
            highlight = "background:#eff6ff;border:2px solid #0077b6;" if num in ("1","10","11","12") else ""
            st.markdown(f'<div class="step-card" style="{highlight}"><span class="step-number">{num}</span><b>{name}</b><br><small style="color:#6b7280;">{detail}</small></div>', unsafe_allow_html=True)
    st.caption("🔵 Blue highlighted steps = handled by this app")
    st.markdown("---")
    st.info("👈 **Start with '🎯 Molecular Docking'** in the sidebar — upload your protein and drug molecule!")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: WORKFLOW GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

def page_workflow():
    st.title("📚 Step-by-Step MD Simulation Guide")
    st.markdown("Complete GROMACS workflow for post-docking molecular dynamics simulation.")

    steps = [
        ("1 — Prepare Docking Output","🎯",
         "Extract best pose, separate protein + ligand.",
         "- Export best docked pose (lowest ΔG binding energy)\n- Separate ATOM (protein) and HETATM (ligand) records\n- Remove waters, co-factors unless essential\n- Verify protonation at pH 7.4",
         "grep '^ATOM' complex.pdb > protein.pdb\ngrep '^HETATM' complex.pdb > ligand.pdb",
         "⚠️ Verify ligand net charge before topology generation."),
        ("2 — Ligand Topology","⚙️",
         "Generate GROMACS-compatible force field for ligand.",
         "- ACPYPE → GAFF2 (AMBER-compatible, recommended)\n- CGenFF / CHARMM-GUI → CHARMM36\n- LigParGen → OPLS-AA",
         "pip install acpype\nacpype -i ligand.pdb -c bcc -n 0\n# Output: ligand_GMX.itp, ligand_GMX.gro, ligand_GMX.top",
         "💡 Specify net charge: acpype -i lig.pdb -c bcc -n -1 for charged ligands."),
        ("3 — Protein Topology","🔬",
         "Assign force field to protein using pdb2gmx.",
         "- Force fields: AMBER99SB-ILDN, CHARMM36m, GROMOS96\n- Water: TIP3P (AMBER), TIP4P/2005\n- Handle HIS protonation: HIE/HID/HIP",
         "gmx pdb2gmx -f protein.pdb -o protein.gro -water tip3p -ff amber99sb-ildn\n# Creates: topol.top, protein.gro, posre.itp",
         "⚠️ Fix missing residues with MODELLER before this step."),
        ("4 — Combine System","🔗",
         "Merge protein + ligand GRO files and update topology.",
         "- Manually merge .gro coordinate files\n- Update atom count in header\n- Add ligand #include to topol.top",
         '# In topol.top, before [ system ]:\n#include "ligand_GMX.itp"\n\n# Under [ molecules ]:\nLIG    1',
         "💡 CHARMM-GUI automates steps 2–4 reliably for complex systems."),
        ("5 — Solvation & Ions","💧",
         "Add water box and physiological NaCl (0.15 M).",
         "- Box: dodecahedron, 1.2 nm padding\n- TIP3P water model\n- Na⁺/Cl⁻ to neutralize + 0.15 M ionic strength",
         "gmx editconf -f complex.gro -o box.gro -c -d 1.2 -bt dodecahedron\ngmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top\ngmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr\ngmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15",
         "💡 Select 'SOL' when genion asks which group to replace."),
        ("6 — Energy Minimization","⚡",
         "Relax the system, remove steric clashes.",
         "- Steepest descent algorithm\n- Converge to Fmax < 1000 kJ/mol/nm\n- Typically 5000–50000 steps",
         "gmx grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr\ngmx mdrun -v -deffnm em\ngmx energy -f em.edr -o em_potential.xvg  # Select 'Potential'",
         "⚠️ If EM fails: check for overlapping atoms or incorrect topology."),
        ("7 — NVT Equilibration","🌡️",
         "Temperature equilibration at 300 K with position restraints.",
         "- Duration: 100–200 ps\n- Thermostat: V-rescale (τ = 0.1 ps)\n- Position restraints on heavy atoms\n- Target: 300 ± 5 K",
         "gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr\ngmx mdrun -v -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0\ngmx energy -f nvt.edr -o nvt_temp.xvg  # Select 'Temperature'",
         "💡 Temperature should plateau at 300 K. Upload nvt_temp.xvg to RMSD page to verify."),
        ("8 — NPT Equilibration","🗜️",
         "Pressure equilibration at 1 bar.",
         "- Duration: 100–200 ps\n- Barostat: Berendsen (equil) → Parrinello-Rahman (production)\n- Target: 1 bar, density ~1000 kg/m³",
         "gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr\ngmx mdrun -v -deffnm npt\ngmx energy -f npt.edr -o npt_density.xvg  # Select 'Density'",
         "💡 Pressure fluctuates ±100 bar — focus on the average (~1 bar)."),
        ("9 — Production MD","🚀",
         "Main simulation — 50–200 ns, no restraints.",
         "- Save coordinates every 10–20 ps\n- No position restraints\n- Fix PBC artifacts before any analysis",
         "gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr\ngmx mdrun -v -deffnm md -ntmpi 1 -ntomp 8 -gpu_id 0\n\n# Fix PBC (ALWAYS do this before analysis)\ngmx trjconv -s md.tpr -f md.xtc -o md_noPBC.xtc -pbc mol -center\n# Select: Protein (center), System (output)",
         "💡 Use -cpi md.cpt -append to continue if interrupted."),
        ("10 — RMSD & RMSF","📊",
         "Quantify stability and flexibility.",
         "- RMSD < 0.3 nm = stable\n- RMSF peaks = flexible loops\n- Analyze backbone + ligand separately",
         "gmx rms -s md.tpr -f md_noPBC.xtc -o rmsd_protein.xvg -tu ns  # Select Backbone, Backbone\ngmx rms -s md.tpr -f md_noPBC.xtc -o rmsd_ligand.xvg -tu ns   # Select Backbone, LIG\ngmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res            # Select C-alpha",
         "💡 Upload the .xvg files to the RMSD/RMSF Viewer page of this app!"),
        ("11 — H-Bond Analysis","🔗",
         "Persistence of protein-ligand hydrogen bonds.",
         "- Average ≥2 H-bonds = stable interaction\n- Occupancy >50% = persistent",
         "gmx hbond -s md.tpr -f md_noPBC.xtc -num hbond_num.xvg -tu ns\n# Select: Protein → LIG",
         "💡 Upload hbond_num.xvg to the H-Bond Analysis page!"),
        ("12 — MM-PBSA","⚗️",
         "Calculate binding free energy.",
         "- ΔGbind = ΔEvdW + ΔEelec + ΔGpolar + ΔGnonpolar\n- Use last 30–50% of trajectory\n- Tool: gmx_MMPBSA",
         "conda install -c conda-forge gmx_mmpbsa\n\ngmx_MMPBSA -O -i mmpbsa.in -cs md.tpr -ct md_noPBC.xtc \\\n           -ci index.ndx -cg 1 13 -cp topol.top",
         "💡 Export CSV and upload to the MM-PBSA page!"),
    ]

    for num, icon, desc, details, cmds, tip in steps:
        with st.expander(f"{icon}  Step {num}", expanded=False):
            c1,c2 = st.columns([1,1], gap="large")
            with c1:
                st.markdown(f"**{desc}**")
                st.markdown(details)
                st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
            with c2:
                st.code(cmds, language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RMSD / RMSF
# ═══════════════════════════════════════════════════════════════════════════════

def page_rmsd_rmsf():
    st.title("📊 RMSD / RMSF Viewer")
    tab1, tab2 = st.tabs(["📈 RMSD — Stability", "📉 RMSF — Flexibility"])

    with tab1:
        st.markdown("### Root Mean Square Deviation (RMSD)")
        st.markdown('<div class="info-box">📌 Stable RMSD plateau < 0.3 nm = well-equilibrated complex.</div>', unsafe_allow_html=True)
        up1 = st.file_uploader("Upload RMSD .xvg / .csv", type=["xvg","txt","csv"], key="r1")
        up2 = st.file_uploader("Optional: 2nd RMSD file (e.g. ligand vs protein)", type=["xvg","txt","csv"], key="r2")
        smooth = st.slider("Smoothing window", 1, 100, 20)
        show_raw = st.checkbox("Show raw data", value=True)

        if up1:
            df,xl,yl,title,_ = parse_xvg(up1)
            if df is not None and not df.empty:
                x,y = df.iloc[:,0], df.iloc[:,1]
                ys = pd.Series(y.values).rolling(smooth, center=True, min_periods=1).mean()
                fig = go.Figure()
                if show_raw:
                    fig.add_trace(go.Scatter(x=x,y=y,mode="lines",name="Raw",line=dict(color="#93c5fd",width=1),opacity=0.5))
                fig.add_trace(go.Scatter(x=x,y=ys,mode="lines",name="Smoothed",line=dict(color="#1d4ed8",width=2.5)))
                if up2:
                    df2,_,_,_,_ = parse_xvg(up2)
                    if df2 is not None and not df2.empty:
                        y2 = df2.iloc[:,1]
                        y2s = pd.Series(y2.values).rolling(smooth,center=True,min_periods=1).mean()
                        if show_raw:
                            fig.add_trace(go.Scatter(x=df2.iloc[:,0],y=y2,mode="lines",name="Raw 2",line=dict(color="#fca5a5",width=1),opacity=0.4))
                        fig.add_trace(go.Scatter(x=df2.iloc[:,0],y=y2s,mode="lines",name="Smoothed 2",line=dict(color="#dc2626",width=2.5)))
                fig.update_layout(title=title or "RMSD vs Time",xaxis_title=xl,yaxis_title=yl,height=420,template="plotly_white",hovermode="x unified")
                st.plotly_chart(fig, use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Mean",f"{y.mean():.3f} nm"); c2.metric("Max",f"{y.max():.3f} nm")
                c3.metric("Std Dev",f"{y.std():.3f} nm"); c4.metric("Final",f"{y.iloc[-1]:.3f} nm")
                mv = y.mean()
                if mv<0.25: st.success("✅ Excellent stability — well-equilibrated complex.")
                elif mv<0.40: st.info("ℹ️ Moderate RMSD — generally stable.")
                elif mv<0.60: st.warning("⚠️ High RMSD — extend equilibration or check binding pose.")
                else: st.error("❌ Very high RMSD — ligand may have dissociated. Inspect in VMD.")
            else: st.error("Could not parse file.")
        else:
            st.code("# GROMACS command:\ngmx rms -s md.tpr -f md_noPBC.xtc -o rmsd.xvg -tu ns\n# Select: Backbone (reference), Backbone or LIG (group)", language="bash")

    with tab2:
        st.markdown("### Root Mean Square Fluctuation (RMSF)")
        st.markdown('<div class="info-box">📌 Peaks = flexible regions (loops, hinges). Binding site residues should have low RMSF.</div>', unsafe_allow_html=True)
        up_rmsf = st.file_uploader("Upload RMSF .xvg / .csv", type=["xvg","txt","csv"], key="rmsf")
        thr = st.slider("Highlight threshold (nm)", 0.05, 1.0, 0.20, step=0.05)
        ctype = st.radio("Chart type", ["Bar","Line"], horizontal=True)

        if up_rmsf:
            df,xl,yl,title,_ = parse_xvg(up_rmsf)
            if df is not None and not df.empty:
                x,y = df.iloc[:,0], df.iloc[:,1]
                colors = ["#dc2626" if v>thr else "#0077b6" for v in y]
                fig = go.Figure()
                if ctype == "Bar":
                    fig.add_trace(go.Bar(x=x,y=y,marker_color=colors))
                else:
                    fig.add_trace(go.Scatter(x=x,y=y,mode="lines",fill="tozeroy",line=dict(color="#0077b6",width=1.5)))
                    hm = y>thr
                    fig.add_trace(go.Scatter(x=x[hm],y=y[hm],mode="markers",marker=dict(color="#dc2626",size=5),name="High flex"))
                fig.add_hline(y=thr,line_dash="dash",line_color="#dc2626",annotation_text=f"Threshold {thr} nm")
                fig.update_layout(title=title or "RMSF per Residue",xaxis_title=xl,yaxis_title=yl,height=420,template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                hi = df[df.iloc[:,1]>thr]
                c1,c2,c3 = st.columns(3)
                c1.metric("Mean RMSF",f"{y.mean():.3f} nm"); c2.metric("Max RMSF",f"{y.max():.3f} nm")
                c3.metric("Flexible residues",f"{len(hi)}")
                if not hi.empty:
                    with st.expander(f"⚡ {len(hi)} high-flexibility residues"):
                        st.dataframe(hi.rename(columns={0:"Residue",1:"RMSF (nm)"}).sort_values("RMSF (nm)",ascending=False))
        else:
            st.code("gmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res\n# Select: C-alpha", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MM-PBSA
# ═══════════════════════════════════════════════════════════════════════════════

def page_mmpbsa():
    st.title("⚗️ MM-PBSA Binding Energy Analysis")
    st.markdown("> **ΔG_bind = ΔEvdW + ΔEelec + ΔGpolar + ΔGnonpolar** (more negative = stronger binding)")

    sample = pd.DataFrame({"Component":["ΔEvdW","ΔEelec","ΔGpolar","ΔGnonpolar","ΔGbind"],
                           "Mean (kcal/mol)":[-32.45,-18.72,24.31,-3.82,-30.68],"Std Dev":[2.1,3.4,1.8,0.4,2.9]})
    tab1,tab2 = st.tabs(["📁 Upload","✏️ Manual Entry"])

    with tab1:
        st.dataframe(sample, use_container_width=True, hide_index=True)
        up = st.file_uploader("Upload MM-PBSA .csv", type=["csv","txt"])
        if up:
            df = pd.read_csv(up)
            st.success("Loaded!"); st.dataframe(df, hide_index=True)
            if "Component" in df.columns and "Mean (kcal/mol)" in df.columns:
                _render_mmpbsa(df)
        else:
            if st.button("Use sample data"): _render_mmpbsa(sample)

    with tab2:
        c1,c2 = st.columns(2)
        with c1:
            vdw = st.number_input("ΔEvdW",value=-32.45,step=0.01)
            elec = st.number_input("ΔEelec",value=-18.72,step=0.01)
        with c2:
            polar = st.number_input("ΔGpolar",value=24.31,step=0.01)
            nonpolar = st.number_input("ΔGnonpolar",value=-3.82,step=0.01)
        total = vdw+elec+polar+nonpolar
        st.metric("ΔGbind",f"{total:.2f} kcal/mol",delta="Favorable" if total<0 else "Unfavorable",delta_color="normal" if total<0 else "inverse")
        if st.button("Generate Charts",type="primary"):
            _render_mmpbsa(pd.DataFrame({"Component":["ΔEvdW","ΔEelec","ΔGpolar","ΔGnonpolar","ΔGbind"],
                                          "Mean (kcal/mol)":[vdw,elec,polar,nonpolar,total],"Std Dev":[0,0,0,0,0]}))

def _render_mmpbsa(df):
    comp = df[df["Component"]!="ΔGbind"].copy()
    colors = ["#16a34a" if v<0 else "#dc2626" for v in comp["Mean (kcal/mol)"]]
    fig = make_subplots(rows=1,cols=2,subplot_titles=["Energy Components","Contribution (%)"])
    fig.add_trace(go.Bar(x=comp["Component"],y=comp["Mean (kcal/mol)"],marker_color=colors,
                         error_y=dict(type="data",array=comp["Std Dev"].tolist(),visible=True) if "Std Dev" in comp.columns else None),row=1,col=1)
    fig.add_hline(y=0,line_color="black",line_width=1,row=1,col=1)
    fig.add_trace(go.Pie(labels=comp["Component"],values=comp["Mean (kcal/mol)"].abs(),hole=0.4,
                         marker_colors=["#1d4ed8","#7c3aed","#dc2626","#ea580c"],showlegend=False),row=1,col=2)
    fig.update_layout(height=400,template="plotly_white",showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    tr = df[df["Component"]=="ΔGbind"]
    if not tr.empty:
        g = tr["Mean (kcal/mol)"].values[0]
        if g<-20: st.success(f"✅ Strong binding: {g:.2f} kcal/mol")
        elif g<-10: st.info(f"ℹ️ Moderate binding: {g:.2f} kcal/mol")
        elif g<0: st.warning(f"⚠️ Weak binding: {g:.2f} kcal/mol")
        else: st.error(f"❌ Unfavorable binding: {g:.2f} kcal/mol")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

def page_structure_viewer():
    st.title("🔬 3D Protein-Ligand Viewer")
    col1,col2 = st.columns([2,1])
    with col1:
        up = st.file_uploader("Upload PDB file", type=["pdb"])
    with col2:
        pstyle = st.selectbox("Protein",["cartoon","stick","sphere","surface"])
        lstyle = st.selectbox("Ligand",["stick","sphere","ballAndStick"])
        bg = st.selectbox("Background",["0x1a1a2e","0xffffff","0x000000"])
        cscheme = st.selectbox("Color",["spectrum","chain","ss","residue"])
    if up:
        pdb = up.read().decode("utf-8",errors="ignore")
        esc = pdb.replace("`","\\`").replace("\\","\\\\")
        html = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>body{{margin:0;background:{bg};}}#v{{width:100%;height:520px;}}</style></head><body>
<div id="v"></div><script>
var v=$3Dmol.createViewer("v",{{backgroundColor:"{bg}"}});
v.addModel(`{esc}`,"pdb");
v.setStyle({{}},{{{pstyle}:{{color:"{cscheme}"}}}});
v.setStyle({{hetflag:true}},{{{lstyle}:{{}}}});
v.setStyle({{hetflag:true,resn:"HOH"}},{{sphere:{{radius:0.15}}}});
v.zoomTo();v.render();
</script></body></html>"""
        components.html(html, height=540)
        lines = pdb.split("\n")
        atoms = [l for l in lines if l.startswith("ATOM")]
        hets  = [l for l in lines if l.startswith("HETATM") and "HOH" not in l]
        c1,c2,c3 = st.columns(3)
        c1.metric("Protein Atoms",len(atoms)); c2.metric("Ligand Atoms",len(hets))
        c3.metric("Residues",len({l[17:26] for l in atoms if len(l)>26}))
    else:
        st.info("Upload a PDB file to visualize")
        st.code("gmx trjconv -s md.tpr -f md_noPBC.xtc -o last_frame.pdb -dump 100000", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: H-BOND
# ═══════════════════════════════════════════════════════════════════════════════

def page_hbond():
    st.title("🔗 Hydrogen Bond Analysis")
    tab1,tab2 = st.tabs(["📈 H-Bond Count vs Time","📋 Interaction Occupancy"])

    with tab1:
        st.markdown('<div class="info-box">📌 Average ≥2 H-bonds = strong stable interaction.</div>', unsafe_allow_html=True)
        up = st.file_uploader("Upload H-bond .xvg / .csv", type=["xvg","txt","csv"])
        if up:
            df,xl,yl,title,_ = parse_xvg(up)
            if df is not None and not df.empty:
                x,y = df.iloc[:,0], df.iloc[:,1].astype(float)
                fig = make_subplots(rows=2,cols=1,row_heights=[0.68,0.32],
                                    subplot_titles=["H-Bonds vs Time","Frequency Distribution"],vertical_spacing=0.12)
                mv = y.mean()
                fig.add_trace(go.Scatter(x=x,y=y,mode="lines",fill="tozeroy",line=dict(color="#0891b2",width=1.2),fillcolor="rgba(8,145,178,.15)"),row=1,col=1)
                fig.add_hline(y=mv,line_dash="dash",line_color="#f59e0b",annotation_text=f"Mean:{mv:.1f}",row=1,col=1)
                fig.add_trace(go.Histogram(x=y,nbinsx=max(int(y.max())+1,5),marker_color="#0891b2"),row=2,col=1)
                fig.update_layout(height=540,template="plotly_white",showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Average",f"{mv:.2f}"); c2.metric("Max",f"{int(y.max())}")
                c3.metric("≥1 H-bond",f"{(y>=1).mean()*100:.1f}%"); c4.metric("≥2 H-bonds",f"{(y>=2).mean()*100:.1f}%")
                if mv>=2.5: st.success("✅ Very strong H-bond network.")
                elif mv>=1.5: st.success("✅ Good H-bond interactions.")
                elif mv>=0.5: st.info("ℹ️ Moderate H-bonds — binding likely hydrophobic-driven.")
                else: st.warning("⚠️ Minimal H-bonds — check if ligand stayed bound.")
        else:
            st.code("gmx hbond -s md.tpr -f md_noPBC.xtc -num hbond_num.xvg -tu ns\n# Select: Protein → LIG", language="bash")

    with tab2:
        st.markdown("### Residue Interaction Occupancy")
        df_edit = st.data_editor(pd.DataFrame({
            "Residue":["ASP101","SER203","HIS305","TYR107","LYS209"],
            "Interaction Type":["H-bond","H-bond","π-stacking","H-bond","Electrostatic"],
            "Occupancy (%)": [82.4,67.1,45.3,91.2,38.7],
            "Avg Distance (Å)":[2.8,3.1,3.9,2.7,3.5],
        }), num_rows="dynamic", use_container_width=True)
        if st.button("📊 Generate Chart"):
            fig = px.bar(df_edit.sort_values("Occupancy (%)",ascending=False),x="Residue",y="Occupancy (%)",
                         color="Interaction Type",text="Occupancy (%)",height=400,template="plotly_white")
            fig.add_hline(y=50,line_dash="dash",line_color="gray",annotation_text="50% threshold")
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: CHEATSHEET
# ═══════════════════════════════════════════════════════════════════════════════

def page_cheatsheet():
    st.title("📋 GROMACS Command Reference")
    search = st.text_input("🔍 Search", placeholder="e.g. RMSD, solvate, GPU, energy...")

    SHEET = {
        "🔧 System Preparation": [
            ("pdb2gmx","gmx pdb2gmx -f protein.pdb -o protein.gro -water tip3p -ff amber99sb-ildn","Generate protein topology"),
            ("editconf","gmx editconf -f complex.gro -o box.gro -c -d 1.2 -bt dodecahedron","Create simulation box"),
            ("solvate","gmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top","Add TIP3P water"),
            ("genion","gmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15","Add NaCl ions"),
            ("make_ndx","gmx make_ndx -f md.tpr -o index.ndx","Create index groups"),
        ],
        "⚡ Run Simulation": [
            ("grompp","gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr","Compile TPR"),
            ("mdrun (CPU)","gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 8","Run on CPU"),
            ("mdrun (GPU)","gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 4 -gpu_id 0","Run on GPU"),
            ("mdrun (continue)","gmx mdrun -v -deffnm md -cpi md.cpt -append","Continue from checkpoint"),
        ],
        "📊 Analysis": [
            ("rms","gmx rms -s md.tpr -f md_noPBC.xtc -o rmsd.xvg -tu ns","RMSD vs time"),
            ("rmsf","gmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res","Per-residue RMSF"),
            ("gyrate","gmx gyrate -s md.tpr -f md_noPBC.xtc -o gyration.xvg","Radius of gyration"),
            ("hbond","gmx hbond -s md.tpr -f md_noPBC.xtc -num hbond.xvg -tu ns","H-bond count"),
            ("sasa","gmx sasa -s md.tpr -f md_noPBC.xtc -o sasa.xvg","Solvent accessible surface area"),
            ("trjconv","gmx trjconv -s md.tpr -f md.xtc -o md_noPBC.xtc -pbc mol -center","Fix PBC"),
            ("trjconv (frame)","gmx trjconv -s md.tpr -f md.xtc -o frame.pdb -dump 100000","Extract frame at 100 ns"),
        ],
        "🌡️ Energy": [
            ("energy (potential)","gmx energy -f em.edr -o potential.xvg   # Select: Potential","EM potential energy"),
            ("energy (temp)","gmx energy -f nvt.edr -o temp.xvg       # Select: Temperature","NVT temperature"),
            ("energy (pressure)","gmx energy -f npt.edr -o pressure.xvg  # Select: Pressure","NPT pressure"),
            ("energy (density)","gmx energy -f npt.edr -o density.xvg   # Select: Density","NPT density"),
        ],
        "⚗️ MM-PBSA": [
            ("install","conda install -c conda-forge gmx_mmpbsa","Install gmx_MMPBSA"),
            ("run","gmx_MMPBSA -O -i mmpbsa.in -cs md.tpr -ct md_noPBC.xtc -ci index.ndx -cg 1 13 -cp topol.top","Run MM-PBSA"),
        ],
        "🔗 Ligand Tools": [
            ("acpype","acpype -i ligand.pdb -c bcc -n 0","GAFF2 topology"),
            ("obabel","obabel ligand.mol2 -O ligand.pdb","Format conversion"),
            ("vina (Python)","from vina import Vina; v=Vina(); v.set_receptor('rec.pdbqt')","Python Vina API"),
        ],
    }

    found = False
    for cat, entries in SHEET.items():
        filt = [e for e in entries if not search or any(search.lower() in x.lower() for x in e)] if search else entries
        if not filt: continue
        found = True
        with st.expander(cat, expanded=bool(search)):
            for name,cmd,desc in filt:
                c1,c2 = st.columns([1,3])
                with c1: st.markdown(f"**`{name}`**"); st.caption(desc)
                with c2: st.code(cmd, language="bash")
                st.markdown("---")
    if search and not found:
        st.info(f"No commands found for '{search}'.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

PAGES = {
    "🏠 Home":                  page_home,
    "🎯 Molecular Docking":     page_docking,
    "📚 MD Workflow Guide":     page_workflow,
    "📊 RMSD / RMSF Viewer":    page_rmsd_rmsf,
    "⚗️ MM-PBSA Analysis":      page_mmpbsa,
    "🔬 3D Structure Viewer":   page_structure_viewer,
    "🔗 H-Bond Analysis":       page_hbond,
    "📋 GROMACS Cheatsheet":    page_cheatsheet,
}

with st.sidebar:
    st.markdown("## 🧬 MD Sim Explorer")
    st.caption("Post-Docking Analysis Toolkit")
    st.markdown("---")
    sel = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    libs = check_docking_libs()
    st.markdown("**Docking Status:**")
    for k, label in [("rdkit","RDKit (ligand prep)"),("vina","Vina binary (cached)")]:
        icon = "🟢" if libs[k] else "🔴"
        st.caption(f"{icon} {label}")
    st.markdown("---")
    st.caption("GROMACS · AMBER · AutoDock Vina")

PAGES[sel]()
