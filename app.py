import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MD Simulation Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(135deg, #00b4d8, #0077b6, #023e8a);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.subtitle { color: #6b7280; font-size: 1.1rem; font-weight: 300; }

.feature-card {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-left: 4px solid #0077b6;
    border-radius: 12px; padding: 1.2rem;
    margin: 0.4rem 0; transition: transform 0.2s;
}
.feature-card:hover { transform: translateX(4px); }
.feature-card h4 { margin: 0 0 0.4rem 0; color: #0077b6; }
.feature-card p { margin: 0; color: #374151; font-size: 0.9rem; }

.step-card {
    background: white; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1rem;
    margin: 0.4rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.step-number {
    display: inline-block; background: #0077b6; color: white;
    border-radius: 50%; width: 28px; height: 28px;
    text-align: center; line-height: 28px; font-weight: 700;
    font-size: 0.85rem; margin-right: 0.6rem;
}

.cmd-box {
    background: #0d1117; color: #58a6ff;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; padding: 1rem; border-radius: 8px;
    border-left: 3px solid #58a6ff; margin: 0.5rem 0;
    overflow-x: auto; white-space: pre;
}
.cmd-comment { color: #8b949e; }

.metric-chip {
    display: inline-block; background: #eff6ff;
    border: 1px solid #bfdbfe; border-radius: 20px;
    padding: 0.3rem 0.8rem; margin: 0.2rem;
    font-size: 0.8rem; color: #1d4ed8; font-weight: 600;
}

.tip-box {
    background: #fefce8; border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
    margin-top: 0.6rem; font-size: 0.88rem; color: #78350f;
}
.info-box {
    background: #f0f9ff; border-left: 3px solid #0ea5e9;
    border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
    margin-top: 0.6rem; font-size: 0.88rem; color: #0c4a6e;
}

.sidebar-header {
    font-size: 1.3rem; font-weight: 700;
    color: #0077b6; text-align: center; padding: 0.5rem 0;
}
.sidebar-sub {
    font-size: 0.75rem; color: #9ca3af;
    text-align: center; margin-bottom: 1rem;
}

div[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; }
div[data-testid="stExpander"] { border: 1px solid #e5e7eb !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ─── XVG Parser ─────────────────────────────────────────────────────────────────
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
        if line.startswith("#"):
            continue
        elif line.startswith("@"):
            if 'xaxis  label' in line and '"' in line:
                x_label = line.split('"')[1]
            elif 'yaxis  label' in line and '"' in line:
                y_label = line.split('"')[1]
            elif line.startswith("@ title") and '"' in line:
                title = line.split('"')[1]
            elif "legend" in line and '"' in line:
                legends.append(line.split('"')[1])
        else:
            try:
                vals = list(map(float, line.split()))
                if vals:
                    data.append(vals)
            except Exception:
                pass
    if not data:
        return None, x_label, y_label, title, legends
    return pd.DataFrame(data), x_label, y_label, title, legends


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════
def page_home():
    st.markdown('<div class="main-title">🧬 MD Simulation Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Interactive guide & analysis toolkit for post-docking Molecular Dynamics</div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    cards = [
        ("📚 Learn", "Step-by-step workflow from docking output to production MD run with GROMACS commands."),
        ("📊 Analyze", "Upload .xvg / .csv result files and get instant interactive plots with statistics."),
        ("🔬 Visualize", "Load a PDB file and explore your protein-ligand complex in 3D interactively."),
    ]
    for col, (title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f'<div class="feature-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🗺️ Workflow Overview")
    steps = [
        ("1", "Molecular Docking", "AutoDock Vina / Glide"),
        ("2", "Ligand Topology", "ACPYPE / CGenFF"),
        ("3", "Protein Topology", "pdb2gmx"),
        ("4", "Solvation & Ions", "editconf + genion"),
        ("5", "Energy Minimization", "gmx mdrun"),
        ("6", "NVT Equilibration", "300K, V-rescale"),
        ("7", "NPT Equilibration", "1 bar, Parrinello-Rahman"),
        ("8", "Production MD", "50–200 ns"),
        ("9", "RMSD / RMSF", "Stability & Flexibility"),
        ("10", "H-Bond Analysis", "Interaction Persistence"),
        ("11", "MM-PBSA", "Binding Free Energy"),
    ]
    cols = st.columns(4)
    for i, (num, name, detail) in enumerate(steps):
        with cols[i % 4]:
            st.markdown(
                f'<div class="step-card">'
                f'<span class="step-number">{num}</span>'
                f'<b>{name}</b><br>'
                f'<small style="color:#6b7280;">{detail}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.info("👈 **Navigate using the sidebar** to access each module. Upload your GROMACS output files (.xvg, .csv, .pdb) directly in each page.")

    with st.expander("📦 Supported File Formats"):
        df = pd.DataFrame({
            "Page": ["RMSD/RMSF Viewer", "H-Bond Analysis", "MM-PBSA Analysis", "3D Structure Viewer"],
            "Input Format": [".xvg / .csv / .txt", ".xvg / .csv / .txt", ".csv", ".pdb"],
            "GROMACS Command": [
                "gmx rms / gmx rmsf",
                "gmx hbond -num",
                "gmx_MMPBSA",
                "gmx trjconv -dump",
            ],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: WORKFLOW GUIDE
# ═══════════════════════════════════════════════════════════════════════════════
def page_workflow():
    st.title("📚 Step-by-Step MD Simulation Guide")
    st.markdown("Complete GROMACS workflow for post-docking molecular dynamics simulation.")

    workflow_steps = [
        {
            "title": "Step 1 – Prepare Docking Output",
            "icon": "🎯",
            "desc": "Extract the best binding pose from docking software and separate protein + ligand.",
            "details": """
- Export best docked pose (lowest ΔG binding energy)
- Separate protein (**ATOM records**) and ligand (**HETATM records**)
- Remove crystallographic waters, co-factors unless essential
- Ensure proper protonation at physiological pH (pH 7.4)
            """,
            "commands": "# Separate protein and ligand from combined PDB\ngrep '^ATOM' complex.pdb > protein.pdb\ngrep '^HETATM' complex.pdb > ligand.pdb",
            "tip": "⚠️ Verify your ligand has the correct net charge before topology generation.",
        },
        {
            "title": "Step 2 – Generate Ligand Topology",
            "icon": "⚙️",
            "desc": "Create GROMACS-compatible force field parameters for the small molecule ligand.",
            "details": """
- **ACPYPE** → GAFF2 force field (AMBER-compatible, recommended)
- **CGenFF / CHARMM-GUI** → CHARMM36 force field
- **LigParGen** → OPLS-AA force field
- Choose force field consistent with protein force field
            """,
            "commands": "# Install ACPYPE\npip install acpype\n\n# Generate GROMACS topology (GAFF2 + BCC charges)\nacpype -i ligand.pdb -c bcc -n 0\n\n# Output files created:\n# ligand_GMX.itp   → topology parameters\n# ligand_GMX.gro   → coordinates\n# ligand_GMX.top   → standalone topology",
            "tip": "💡 If your ligand has a net charge (e.g. -1), specify with -n flag: acpype -i lig.pdb -c bcc -n -1",
        },
        {
            "title": "Step 3 – Generate Protein Topology",
            "icon": "🔬",
            "desc": "Use pdb2gmx to assign force field parameters to the protein.",
            "details": """
- **Force fields**: AMBER99SB-ILDN (most popular), CHARMM36m, GROMOS96
- **Water models**: TIP3P (AMBER), TIP4P/2005, SPC/E
- Handle histidine (HIS) protonation: HIE, HID, or HIP
- Fix missing heavy atoms using MODELLER or Swiss-PdbViewer
            """,
            "commands": "# Generate protein topology\ngmx pdb2gmx -f protein.pdb -o protein.gro -water tip3p -ff amber99sb-ildn\n\n# Files created:\n# topol.top    → master topology\n# protein.gro  → coordinates\n# posre.itp    → position restraints",
            "tip": "⚠️ Check terminal residues and non-standard amino acids. Use -ignh to ignore existing hydrogens.",
        },
        {
            "title": "Step 4 – Combine Protein + Ligand System",
            "icon": "🔗",
            "desc": "Merge protein and ligand GRO files and update the master topology.",
            "details": """
- Manually merge .gro coordinate files (or use CHARMM-GUI)
- Update atom count in the merged .gro header
- Add ligand `#include` to `topol.top`
- Append ligand to `[ molecules ]` section
            """,
            "commands": '# In topol.top — ADD before "[ system ]":\n; Include ligand topology\n#include "ligand_GMX.itp"\n\n# In topol.top — ADD in "[ molecules ]":\n; Protein_chain_A   1\nLIG               1\n\n# Merge GRO files (update first line atom count!)\nhead -1 protein.gro\n# Update count = protein_atoms + ligand_atoms',
            "tip": "💡 CHARMM-GUI's 'Ligand Reader & Modeler' automates steps 2–4 reliably.",
        },
        {
            "title": "Step 5 – Solvation & Ionization",
            "icon": "💧",
            "desc": "Surround the complex with a water box and add physiological salt concentration.",
            "details": """
- Box type: **dodecahedron** (efficient) or cubic
- Minimum distance from protein to box edge: **1.2 nm**
- Add **Na⁺ / Cl⁻** to neutralize and reach 0.15 M ionic strength
            """,
            "commands": "# Create simulation box (dodecahedral, 1.2 nm padding)\ngmx editconf -f complex.gro -o box.gro -c -d 1.2 -bt dodecahedron\n\n# Solvate with TIP3P water\ngmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top\n\n# Add ions (neutralize + 0.15 M NaCl)\ngmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 2\ngmx genion -s ions.tpr -o ionized.gro -p topol.top \\\n           -pname NA -nname CL -neutral -conc 0.15",
            "tip": "💡 Select group 'SOL' when prompted by genion to replace water molecules with ions.",
        },
        {
            "title": "Step 6 – Energy Minimization",
            "icon": "⚡",
            "desc": "Relax the solvated system to remove steric clashes and bad contacts.",
            "details": """
- Algorithm: **Steepest descent**
- Convergence criterion: **Fmax < 1000 kJ/mol/nm**
- Typical steps: 5000–50,000
- Monitor potential energy (should decrease monotonically)
            """,
            "commands": "# Prepare EM input\ngmx grompp -f em.mdp -c ionized.gro -p topol.top -o em.tpr\n\n# Run energy minimization\ngmx mdrun -v -deffnm em\n\n# Check potential energy convergence\ngmx energy -f em.edr -o em_potential.xvg\n# → Type '10' to select Potential Energy, then '0' to quit",
            "tip": "⚠️ If EM fails with 'Lincs Warning', check for overlapping atoms or incorrect topology.",
        },
        {
            "title": "Step 7 – NVT Equilibration (Temperature)",
            "icon": "🌡️",
            "desc": "Equilibrate system temperature at 300 K with position restraints on heavy atoms.",
            "details": """
- Duration: **100–200 ps**
- Thermostat: **V-rescale** (τ = 0.1 ps)
- Protein & ligand heavy atoms restrained
- Monitor: temperature should plateau at ~300 K
            """,
            "commands": "# Prepare NVT\ngmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr\n\n# Run NVT (GPU accelerated)\ngmx mdrun -v -deffnm nvt -ntmpi 1 -ntomp 4 -gpu_id 0\n\n# Check temperature equilibration\ngmx energy -f nvt.edr -o nvt_temp.xvg\n# → Select 'Temperature'",
            "tip": "💡 Temperature should converge to 300 ± 5 K. A stable plateau indicates good equilibration.",
        },
        {
            "title": "Step 8 – NPT Equilibration (Pressure)",
            "icon": "🗜️",
            "desc": "Equilibrate pressure at 1 bar while maintaining 300 K temperature.",
            "details": """
- Duration: **100–200 ps**
- Barostat: **Berendsen** for equilibration, **Parrinello-Rahman** for production
- Target pressure: **1 bar**
- Monitor: density should reach ~1000 kg/m³ (water)
            """,
            "commands": "# Prepare NPT (continues from NVT checkpoint)\ngmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt \\\n           -p topol.top -o npt.tpr\n\n# Run NPT\ngmx mdrun -v -deffnm npt\n\n# Check pressure and density\ngmx energy -f npt.edr -o npt_pressure.xvg  # Select 'Pressure'\ngmx energy -f npt.edr -o npt_density.xvg   # Select 'Density'",
            "tip": "💡 Pressure fluctuates heavily (±100 bar is normal). Focus on the average value (~1 bar).",
        },
        {
            "title": "Step 9 – Production MD Run",
            "icon": "🚀",
            "desc": "The main simulation — no restraints, typically 50–200 ns.",
            "details": """
- Duration: **50 ns** (minimum), **100–200 ns** (recommended)
- Save coordinates every **10–20 ps**
- No position restraints
- Remove PBC artifacts before analysis
            """,
            "commands": "# Prepare production MD\ngmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr\n\n# Run production (GPU, 8 CPU threads)\ngmx mdrun -v -deffnm md -ntmpi 1 -ntomp 8 -gpu_id 0\n\n# Continue if interrupted\ngmx mdrun -v -deffnm md -cpi md.cpt -append\n\n# Fix PBC BEFORE any analysis\ngmx trjconv -s md.tpr -f md.xtc -o md_noPBC.xtc -pbc mol -center\n# → Select 'Protein' for centering, 'System' for output",
            "tip": "💡 Use -ntmpi 1 to avoid MPI conflicts. Adjust -ntomp to match your CPU core count.",
        },
        {
            "title": "Step 10 – RMSD & RMSF Analysis",
            "icon": "📊",
            "desc": "Quantify structural stability (RMSD) and residue flexibility (RMSF).",
            "details": """
- **RMSD < 0.2–0.3 nm** = stable system
- **RMSF peaks** = flexible loops / hinge regions
- Analyze **protein backbone** and **ligand** separately
- Use PBC-corrected trajectory
            """,
            "commands": "# Protein backbone RMSD vs time\ngmx rms -s md.tpr -f md_noPBC.xtc -o rmsd_protein.xvg -tu ns\n# → Select 'Backbone' for reference\n# → Select 'Backbone' for group\n\n# Ligand RMSD vs time\ngmx rms -s md.tpr -f md_noPBC.xtc -o rmsd_ligand.xvg -tu ns\n# → Select 'Backbone' for reference\n# → Select 'LIG' for group\n\n# Per-residue RMSF\ngmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res\n# → Select 'C-alpha'",
            "tip": "💡 Upload the generated .xvg files to the RMSD/RMSF Viewer page of this app!",
        },
        {
            "title": "Step 11 – H-Bond Analysis",
            "icon": "🔗",
            "desc": "Analyze hydrogen bonds between protein and ligand over the trajectory.",
            "details": """
- Average H-bonds: meaningful metric of interaction stability
- **Occupancy > 50%** = persistent interaction
- Map specific donor-acceptor pairs for publication
            """,
            "commands": "# Create custom index: protein + ligand\ngmx make_ndx -f md.tpr -o index.ndx\n# → Create group for ligand if needed\n\n# Count H-bonds over time\ngmx hbond -s md.tpr -f md_noPBC.xtc -num hbond_num.xvg -tu ns -n index.ndx\n# → Select 'Protein' (group 1)\n# → Select 'LIG' (group 2)\n\n# H-bond distance/angle distributions\ngmx hbond -s md.tpr -f md_noPBC.xtc -dist hbond_dist.xvg -ang hbond_ang.xvg",
            "tip": "💡 Use the H-Bond Analysis page to upload hbond_num.xvg and get occupancy statistics.",
        },
        {
            "title": "Step 12 – MM-PBSA Binding Free Energy",
            "icon": "⚗️",
            "desc": "Calculate absolute binding free energy using MM-PBSA/MM-GBSA method.",
            "details": """
- **ΔGbind = ΔEvdW + ΔEelec + ΔGpolar + ΔGnonpolar**
- Tool: **gmx_MMPBSA** (AMBER interface for GROMACS)
- Use equilibrated portion of trajectory (last 30–50%)
- Entropy correction (TΔS) optional but important for ranking
            """,
            "commands": "# Install gmx_MMPBSA\nconda install -c conda-forge gmx_mmpbsa\n\n# Create receptor+ligand index groups\ngmx make_ndx -f md.tpr -o index.ndx\n\n# Run MM-PBSA\ngmx_MMPBSA -O \\\n  -i mmpbsa.in \\\n  -cs md.tpr \\\n  -ct md_noPBC.xtc \\\n  -ci index.ndx \\\n  -cg 1 13 \\\n  -cp topol.top\n\n# mmpbsa.in (PB method):\n&general\n  startframe=500, endframe=1000, interval=5,\n/\n&pb\n  istrng=0.15, inp=2,\n/",
            "tip": "💡 Export results as CSV and upload to the MM-PBSA Analysis page of this app.",
        },
    ]

    for step in workflow_steps:
        with st.expander(f"{step['icon']}  {step['title']}", expanded=False):
            col_left, col_right = st.columns([1, 1], gap="large")
            with col_left:
                st.markdown(f"**{step['desc']}**")
                st.markdown(step["details"])
                st.markdown(f'<div class="tip-box">{step["tip"]}</div>', unsafe_allow_html=True)
            with col_right:
                st.markdown("**Commands:**")
                st.code(step["commands"], language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: RMSD / RMSF
# ═══════════════════════════════════════════════════════════════════════════════
def page_rmsd_rmsf():
    st.title("📊 RMSD / RMSF Viewer")
    st.markdown("Upload GROMACS `.xvg` output files to visualize structural stability and per-residue flexibility.")

    tab1, tab2 = st.tabs(["📈 RMSD — Stability", "📉 RMSF — Flexibility"])

    # ── RMSD ──
    with tab1:
        st.markdown("### Root Mean Square Deviation (RMSD)")
        st.markdown('<div class="info-box">📌 RMSD measures deviation from the reference structure. A stable plateau (< 0.3 nm) indicates a well-equilibrated complex.</div>', unsafe_allow_html=True)

        col_up, col_opt = st.columns([2, 1])
        with col_up:
            uploaded_rmsd = st.file_uploader("Upload RMSD .xvg / .csv file", type=["xvg", "txt", "csv"], key="rmsd_up")
        with col_opt:
            smooth = st.slider("Smoothing window", 1, 100, 20, key="rmsd_smooth")
            show_raw = st.checkbox("Show raw data", value=True, key="rmsd_raw")

        # Multi-file comparison
        uploaded_rmsd2 = st.file_uploader("Optional: Upload a 2nd RMSD file to compare (e.g. ligand vs protein)", type=["xvg","txt","csv"], key="rmsd_up2")

        if uploaded_rmsd:
            df, x_lbl, y_lbl, title, _ = parse_xvg(uploaded_rmsd)
            if df is not None and not df.empty:
                x, y = df.iloc[:, 0], df.iloc[:, 1]
                y_sm = pd.Series(y.values).rolling(smooth, center=True, min_periods=1).mean()

                fig = go.Figure()
                if show_raw:
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Raw",
                                             line=dict(color="#93c5fd", width=1), opacity=0.6))
                fig.add_trace(go.Scatter(x=x, y=y_sm, mode="lines", name="Smoothed (File 1)",
                                         line=dict(color="#1d4ed8", width=2.5)))

                if uploaded_rmsd2:
                    df2, x2, y2_lbl, t2, _ = parse_xvg(uploaded_rmsd2)
                    if df2 is not None and not df2.empty:
                        y2 = df2.iloc[:, 1]
                        y2_sm = pd.Series(y2.values).rolling(smooth, center=True, min_periods=1).mean()
                        if show_raw:
                            fig.add_trace(go.Scatter(x=df2.iloc[:,0], y=y2, mode="lines", name="Raw (File 2)",
                                                     line=dict(color="#fca5a5", width=1), opacity=0.5))
                        fig.add_trace(go.Scatter(x=df2.iloc[:,0], y=y2_sm, mode="lines", name="Smoothed (File 2)",
                                                 line=dict(color="#dc2626", width=2.5)))

                fig.update_layout(
                    title=title or "RMSD vs Time",
                    xaxis_title=x_lbl,
                    yaxis_title=y_lbl if y_lbl != "Value" else "RMSD (nm)",
                    height=430,
                    template="plotly_white",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Stats row
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean RMSD", f"{y.mean():.3f} nm")
                c2.metric("Max RMSD", f"{y.max():.3f} nm")
                c3.metric("Std Dev", f"{y.std():.3f} nm")
                c4.metric("Final RMSD", f"{y.iloc[-1]:.3f} nm")

                mean_v = y.mean()
                if mean_v < 0.25:
                    st.success("✅ Excellent stability — complex is tightly bound and well-equilibrated.")
                elif mean_v < 0.40:
                    st.info("ℹ️ Moderate RMSD — some flexibility but generally stable.")
                elif mean_v < 0.60:
                    st.warning("⚠️ High RMSD — consider longer equilibration or check binding pose.")
                else:
                    st.error("❌ Very high RMSD — ligand may have dissociated. Review trajectory in VMD/PyMOL.")

                with st.expander("📋 View raw data table"):
                    st.dataframe(df.rename(columns={0: x_lbl, 1: y_lbl}).head(500), use_container_width=True)
            else:
                st.error("Could not parse file. Ensure it is a valid GROMACS .xvg or two-column CSV.")
        else:
            st.markdown("#### 📎 Expected file format")
            st.code("# Time (ns)  RMSD (nm)\n  0.000       0.000\n  0.010       0.087\n  0.020       0.112\n  0.030       0.145\n  ...", language="text")

    # ── RMSF ──
    with tab2:
        st.markdown("### Root Mean Square Fluctuation (RMSF)")
        st.markdown('<div class="info-box">📌 RMSF measures per-residue flexibility. Peaks identify loop regions, binding site flexibility, and disordered segments.</div>', unsafe_allow_html=True)

        uploaded_rmsf = st.file_uploader("Upload RMSF .xvg / .csv file", type=["xvg", "txt", "csv"], key="rmsf_up")

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            threshold = st.slider("Highlight threshold (nm)", 0.05, 1.0, 0.20, step=0.05)
        with col_opt2:
            chart_type = st.radio("Chart type", ["Bar", "Line"], horizontal=True)

        if uploaded_rmsf:
            df, x_lbl, y_lbl, title, _ = parse_xvg(uploaded_rmsf)
            if df is not None and not df.empty:
                x, y = df.iloc[:, 0], df.iloc[:, 1]
                colors = ["#dc2626" if v > threshold else "#0077b6" for v in y]

                fig = go.Figure()
                if chart_type == "Bar":
                    fig.add_trace(go.Bar(x=x, y=y, marker_color=colors, name="RMSF"))
                else:
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy",
                                             line=dict(color="#0077b6", width=1.5)))
                    high_mask = y > threshold
                    fig.add_trace(go.Scatter(x=x[high_mask], y=y[high_mask], mode="markers",
                                             marker=dict(color="#dc2626", size=5), name="High flexibility"))

                fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626",
                              annotation_text=f"Threshold {threshold} nm", annotation_position="top right")
                fig.update_layout(
                    title=title or "Per-Residue RMSF",
                    xaxis_title=x_lbl if x_lbl != "Value" else "Residue Number",
                    yaxis_title=y_lbl if y_lbl != "Value" else "RMSF (nm)",
                    height=430,
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

                high_flex = df[df.iloc[:, 1] > threshold]
                c1, c2, c3 = st.columns(3)
                c1.metric("Mean RMSF", f"{y.mean():.3f} nm")
                c2.metric("Max RMSF", f"{y.max():.3f} nm")
                c3.metric("Flexible residues", f"{len(high_flex)} (>{threshold} nm)")

                if not high_flex.empty:
                    with st.expander(f"⚡ {len(high_flex)} flexible residues (RMSF > {threshold} nm)"):
                        st.dataframe(
                            high_flex.rename(columns={0: "Residue", 1: "RMSF (nm)"}).sort_values("RMSF (nm)", ascending=False),
                            use_container_width=True,
                        )
            else:
                st.error("Could not parse file.")
        else:
            st.markdown("#### 📎 How to generate RMSF file:")
            st.code("gmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res\n# Select: C-alpha", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MM-PBSA
# ═══════════════════════════════════════════════════════════════════════════════
def page_mmpbsa():
    st.title("⚗️ MM-PBSA Binding Energy Analysis")

    st.markdown("""
    **MM-PBSA binding free energy equation:**

    > **ΔG_bind = ΔE_vdW + ΔE_elec + ΔG_polar solvation + ΔG_nonpolar solvation**

    More negative ΔG_bind = stronger, more favorable binding.
    """)

    tab1, tab2 = st.tabs(["📁 Upload Results", "✏️ Manual Entry"])

    sample_df = pd.DataFrame({
        "Component": ["ΔEvdW", "ΔEelec", "ΔGpolar", "ΔGnonpolar", "ΔGbind"],
        "Mean (kcal/mol)": [-32.45, -18.72, 24.31, -3.82, -30.68],
        "Std Dev": [2.10, 3.40, 1.80, 0.40, 2.90],
    })

    with tab1:
        st.markdown("**Expected CSV format:**")
        st.dataframe(sample_df, use_container_width=True, hide_index=True)
        uploaded = st.file_uploader("Upload MM-PBSA results (.csv)", type=["csv", "txt"])

        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.success("✅ File loaded successfully!")
                st.dataframe(df, use_container_width=True, hide_index=True)
                if "Component" in df.columns and "Mean (kcal/mol)" in df.columns:
                    _render_mmpbsa(df)
                else:
                    st.warning("Please ensure columns are named 'Component', 'Mean (kcal/mol)', 'Std Dev'.")
            except Exception as e:
                st.error(f"Failed to parse file: {e}")
        else:
            if st.button("📊 Use sample data"):
                _render_mmpbsa(sample_df)

    with tab2:
        st.markdown("### Enter values from your MM-PBSA output")
        c1, c2 = st.columns(2)
        with c1:
            vdw = st.number_input("ΔEvdW (kcal/mol)", value=-32.45, step=0.01)
            elec = st.number_input("ΔEelec (kcal/mol)", value=-18.72, step=0.01)
        with c2:
            polar = st.number_input("ΔGpolar (kcal/mol)", value=24.31, step=0.01)
            nonpolar = st.number_input("ΔGnonpolar (kcal/mol)", value=-3.82, step=0.01)

        total = vdw + elec + polar + nonpolar
        delta_color = "normal" if total < 0 else "inverse"
        st.metric("Calculated ΔGbind", f"{total:.2f} kcal/mol",
                  delta="Favorable binding" if total < 0 else "Unfavorable binding",
                  delta_color=delta_color)

        if st.button("Generate Charts", type="primary"):
            manual_df = pd.DataFrame({
                "Component": ["ΔEvdW", "ΔEelec", "ΔGpolar", "ΔGnonpolar", "ΔGbind"],
                "Mean (kcal/mol)": [vdw, elec, polar, nonpolar, total],
                "Std Dev": [0.0, 0.0, 0.0, 0.0, 0.0],
            })
            _render_mmpbsa(manual_df)


def _render_mmpbsa(df):
    comp_df = df[df["Component"] != "ΔGbind"].copy()
    total_row = df[df["Component"] == "ΔGbind"]

    colors = ["#16a34a" if v < 0 else "#dc2626" for v in comp_df["Mean (kcal/mol)"]]

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Energy Components", "Contribution Breakdown"])

    # Bar chart
    error_y = comp_df["Std Dev"].tolist() if "Std Dev" in comp_df.columns else None
    fig.add_trace(
        go.Bar(x=comp_df["Component"], y=comp_df["Mean (kcal/mol)"],
               marker_color=colors, name="Energy",
               error_y=dict(type="data", array=error_y, visible=True) if error_y else None),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)

    # Pie chart
    fig.add_trace(
        go.Pie(labels=comp_df["Component"],
               values=comp_df["Mean (kcal/mol)"].abs(),
               hole=0.4,
               marker_colors=["#1d4ed8","#7c3aed","#dc2626","#ea580c"],
               textinfo="label+percent",
               showlegend=False),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Component", row=1, col=1)
    fig.update_yaxes(title_text="Energy (kcal/mol)", row=1, col=1)
    fig.update_layout(height=420, template="plotly_white", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if not total_row.empty:
        gbind = total_row["Mean (kcal/mol)"].values[0]
        if gbind < -20:
            st.success(f"✅ **Strong binding**: ΔGbind = {gbind:.2f} kcal/mol — potent inhibitor candidate.")
        elif gbind < -10:
            st.info(f"ℹ️ **Moderate binding**: ΔGbind = {gbind:.2f} kcal/mol — promising lead compound.")
        elif gbind < 0:
            st.warning(f"⚠️ **Weak binding**: ΔGbind = {gbind:.2f} kcal/mol — may need structural optimization.")
        else:
            st.error(f"❌ **Unfavorable binding**: ΔGbind = {gbind:.2f} kcal/mol — consider redesigning ligand.")

    # Interpretation table
    with st.expander("📖 Component Interpretation Guide"):
        guide = pd.DataFrame({
            "Component": ["ΔEvdW", "ΔEelec", "ΔGpolar", "ΔGnonpolar", "ΔGbind"],
            "Favorable?": ["Negative (−)", "Negative (−)", "Positive (+) destabilizes", "Negative (−)", "Negative (−)"],
            "Physical Meaning": [
                "van der Waals contacts (hydrophobic packing)",
                "Electrostatic interactions (H-bonds, salt bridges)",
                "Cost of desolvating polar groups",
                "Hydrophobic burial / SASA-based term",
                "Net binding free energy",
            ],
        })
        st.dataframe(guide, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: 3D STRUCTURE VIEWER
# ═══════════════════════════════════════════════════════════════════════════════
def page_structure_viewer():
    st.title("🔬 3D Protein-Ligand Structure Viewer")
    st.markdown("Upload a PDB file to explore your complex interactively using **3Dmol.js**.")

    col_up, col_opts = st.columns([2, 1])
    with col_up:
        uploaded_pdb = st.file_uploader("Upload PDB file", type=["pdb"])
    with col_opts:
        prot_style = st.selectbox("Protein style", ["cartoon", "stick", "sphere", "surface"])
        lig_style = st.selectbox("Ligand style", ["stick", "sphere", "ballAndStick"])
        bg_color = st.selectbox("Background", ["0x1a1a2e", "0xffffff", "0x000000", "0xf0f9ff"])
        color_scheme = st.selectbox("Protein color", ["spectrum", "chain", "ss", "residue"])
        show_water = st.checkbox("Show water molecules", value=False)

    if uploaded_pdb:
        pdb_str = uploaded_pdb.read().decode("utf-8", errors="ignore")
        pdb_escaped = pdb_str.replace("`", "\\`").replace("\\", "\\\\")

        water_js = "" if show_water else "viewer.setStyle({resn:'HOH'}, {sphere:{radius:0.2, color:'cyan'}});"

        html_3d = f"""<!DOCTYPE html><html><head>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
<script src="https://3dmol.org/build/3Dmol-min.js"></script>
<style>
  body{{margin:0;background:{bg_color};overflow:hidden;}}
  #glv{{width:100%;height:520px;position:relative;}}
  #info{{position:absolute;bottom:10px;left:10px;background:rgba(0,0,0,0.6);
         color:white;padding:6px 12px;border-radius:6px;font-family:monospace;font-size:12px;}}
</style></head><body>
<div id="glv"></div>
<div id="info">Left-click drag: rotate | Scroll: zoom | Right-click drag: translate</div>
<script>
var v = $3Dmol.createViewer('glv', {{backgroundColor:'{bg_color}'}});
var pdb = `{pdb_escaped}`;
v.addModel(pdb,'pdb');
v.setStyle({{}},{{{prot_style}:{{color:'{color_scheme}'}}}});
v.setStyle({{hetflag:true}},{{{lig_style}:{{}}}});
v.setStyle({{hetflag:true,resn:'HOH'}},{{"sphere":{{"radius":0.15}}}});
{"v.removeStyle({resn:'HOH'});" if not show_water else ""}
v.addSurface($3Dmol.SurfaceType.VDW,{{opacity:0.0}},{{}});
v.zoomTo({{hetflag:true}});
v.render();
v.spin(false);
</script></body></html>"""

        components.html(html_3d, height=530, scrolling=False)

        # Metadata
        lines = pdb_str.split("\n")
        atoms = [l for l in lines if l.startswith("ATOM")]
        hetatms = [l for l in lines if l.startswith("HETATM") and "HOH" not in l]
        waters = [l for l in lines if l.startswith("HETATM") and "HOH" in l]
        residues = len({l[17:26] for l in atoms if len(l) > 26})
        lig_names = list({l[17:20].strip() for l in hetatms})

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Protein Atoms", len(atoms))
        c2.metric("Protein Residues", residues)
        c3.metric("Ligand Atoms", len(hetatms))
        c4.metric("Water Molecules", len(waters))

        if lig_names:
            st.markdown(f"**Ligand(s) detected:** {', '.join(lig_names)}")
    else:
        st.info("👆 Upload a PDB file to start visualization")
        with st.expander("💡 How to extract a frame from your MD trajectory"):
            st.code("# Extract the last frame\ngmx trjconv -s md.tpr -f md_noPBC.xtc -o last_frame.pdb -dump 100000\n# → Select 'System'\n\n# Extract frame at specific time (e.g. 50 ns)\ngmx trjconv -s md.tpr -f md_noPBC.xtc -o frame_50ns.pdb -dump 50000\n\n# Extract docked pose from AutoDock Vina\n# Just open .pdbqt in Chimera and save as .pdb", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: H-BOND ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def page_hbond():
    st.title("🔗 Hydrogen Bond Analysis")
    st.markdown("Analyze the number and persistence of H-bonds between protein and ligand.")

    tab1, tab2 = st.tabs(["📈 H-Bond Count over Time", "📋 Interaction Occupancy Table"])

    with tab1:
        st.markdown('<div class="info-box">📌 An average of ≥2 H-bonds throughout the simulation indicates strong, stable interactions.</div>', unsafe_allow_html=True)
        uploaded_hb = st.file_uploader("Upload H-bond count .xvg / .csv", type=["xvg", "txt", "csv"])

        if uploaded_hb:
            df, x_lbl, y_lbl, title, _ = parse_xvg(uploaded_hb)
            if df is not None and not df.empty:
                x, y = df.iloc[:, 0], df.iloc[:, 1].astype(float)

                fig = make_subplots(
                    rows=2, cols=1, row_heights=[0.68, 0.32],
                    subplot_titles=["H-Bonds vs Time", "Frequency Distribution"],
                    vertical_spacing=0.12,
                )

                mean_val = y.mean()
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="tozeroy",
                                         line=dict(color="#0891b2", width=1.2), fillcolor="rgba(8,145,178,0.15)",
                                         name="H-bond count"), row=1, col=1)
                fig.add_hline(y=mean_val, line_dash="dash", line_color="#f59e0b",
                              annotation_text=f"Mean: {mean_val:.1f}", row=1, col=1)

                fig.add_trace(go.Histogram(x=y, nbinsx=int(y.max())+1,
                                            marker_color="#0891b2", name="Distribution"), row=2, col=1)

                fig.update_xaxes(title_text=x_lbl if x_lbl != "X" else "Time (ns)", row=1, col=1)
                fig.update_xaxes(title_text="H-Bond Count", row=2, col=1)
                fig.update_yaxes(title_text="H-Bond Count", row=1, col=1)
                fig.update_yaxes(title_text="Frequency", row=2, col=1)
                fig.update_layout(height=560, template="plotly_white", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Average H-bonds", f"{mean_val:.2f}")
                c2.metric("Max H-bonds", f"{int(y.max())}")
                c3.metric("≥1 H-bond (%)", f"{(y >= 1).mean()*100:.1f}%")
                c4.metric("≥2 H-bonds (%)", f"{(y >= 2).mean()*100:.1f}%")

                if mean_val >= 2.5:
                    st.success("✅ Very strong H-bond network — ligand is tightly anchored in binding site.")
                elif mean_val >= 1.5:
                    st.success("✅ Good H-bond interactions — stable complex throughout simulation.")
                elif mean_val >= 0.5:
                    st.info("ℹ️ Moderate H-bonds — binding may be primarily hydrophobic-driven.")
                else:
                    st.warning("⚠️ Minimal H-bonds — check if ligand remained bound. Inspect RMSD.")
            else:
                st.error("Could not parse file.")
        else:
            st.code("# Generate H-bond count .xvg\ngmx hbond -s md.tpr -f md_noPBC.xtc -num hbond_num.xvg -tu ns\n# → Select 'Protein' for group 1\n# → Select 'LIG' for group 2", language="bash")

    with tab2:
        st.markdown("### Residue-level Interaction Occupancy")
        st.markdown("Enter key interactions identified from your trajectory analysis (e.g. from PyMOL, PLIP, or LigPlot+).")

        default_data = pd.DataFrame({
            "Residue": ["ASP101", "SER203", "HIS305", "TYR107", "LYS209", "GLU98"],
            "Interaction Type": ["H-bond (acceptor)", "H-bond (donor)", "π-stacking", "H-bond (donor)", "Electrostatic", "H-bond (acceptor)"],
            "Occupancy (%)": [82.4, 67.1, 45.3, 91.2, 38.7, 54.9],
            "Avg Distance (Å)": [2.8, 3.1, 3.9, 2.7, 3.5, 2.9],
        })

        edited_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)

        if st.button("📊 Generate Occupancy Chart"):
            color_map = {
                "H-bond (acceptor)": "#1d4ed8",
                "H-bond (donor)": "#0891b2",
                "π-stacking": "#7c3aed",
                "Electrostatic": "#dc2626",
                "Hydrophobic": "#ea580c",
            }
            fig = px.bar(
                edited_df.sort_values("Occupancy (%)", ascending=False),
                x="Residue", y="Occupancy (%)",
                color="Interaction Type",
                text="Occupancy (%)",
                title="Protein-Ligand Interaction Occupancy",
                color_discrete_map=color_map,
                height=420,
            )
            fig.add_hline(y=50, line_dash="dash", line_color="gray",
                          annotation_text="50% occupancy threshold")
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(template="plotly_white", xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

            persistent = edited_df[edited_df["Occupancy (%)"] >= 50]
            if not persistent.empty:
                st.success(f"✅ {len(persistent)} key residues show persistent interactions (≥50% occupancy): "
                           f"{', '.join(persistent['Residue'].tolist())}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: GROMACS CHEATSHEET
# ═══════════════════════════════════════════════════════════════════════════════
def page_cheatsheet():
    st.title("📋 GROMACS Command Reference")
    st.markdown("Quick-reference for all common GROMACS commands used in post-docking MD simulation.")

    search = st.text_input("🔍 Search commands", placeholder="e.g. RMSD, solvate, energy, GPU...")

    CHEATSHEET = {
        "🔧 System Preparation": [
            ("pdb2gmx", "gmx pdb2gmx -f protein.pdb -o protein.gro -water tip3p -ff amber99sb-ildn", "Generate protein topology & coordinates"),
            ("editconf", "gmx editconf -f complex.gro -o box.gro -c -d 1.2 -bt dodecahedron", "Define simulation box (dodecahedron)"),
            ("solvate", "gmx solvate -cp box.gro -cs spc216.gro -o solvated.gro -p topol.top", "Add TIP3P water molecules"),
            ("genion", "gmx genion -s ions.tpr -o ionized.gro -p topol.top -pname NA -nname CL -neutral -conc 0.15", "Add NaCl ions (0.15 M physiological)"),
            ("make_ndx", "gmx make_ndx -f md.tpr -o index.ndx", "Create custom atom index groups"),
        ],
        "⚡ Preprocessing & Running": [
            ("grompp", "gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr", "Compile run input file (.tpr)"),
            ("mdrun (CPU)", "gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 8", "Run on CPU (8 threads)"),
            ("mdrun (GPU)", "gmx mdrun -v -deffnm md -ntmpi 1 -ntomp 4 -gpu_id 0", "Run with GPU acceleration"),
            ("mdrun (continue)", "gmx mdrun -v -deffnm md -cpi md.cpt -append", "Continue from checkpoint"),
            ("mdrun (multi-GPU)", "gmx mdrun -v -deffnm md -ntmpi 2 -ntomp 4 -gpu_id 01", "Run on 2 GPUs"),
        ],
        "🔄 Trajectory Processing": [
            ("trjconv (fix PBC)", "gmx trjconv -s md.tpr -f md.xtc -o md_noPBC.xtc -pbc mol -center", "Remove periodic boundary artifacts"),
            ("trjconv (center)", "gmx trjconv -s md.tpr -f md.xtc -o md_center.xtc -center -pbc nojump", "Center protein in box"),
            ("trjconv (extract frame)", "gmx trjconv -s md.tpr -f md.xtc -o frame.pdb -dump 100000", "Extract frame at 100 ns"),
            ("trjcat", "gmx trjcat -f run1.xtc run2.xtc -o combined.xtc -settime", "Concatenate trajectory files"),
            ("trjconv (skip)", "gmx trjconv -s md.tpr -f md.xtc -o md_skip10.xtc -skip 10", "Reduce trajectory size (every 10th frame)"),
        ],
        "📊 Analysis Commands": [
            ("rms (RMSD)", "gmx rms -s md.tpr -f md_noPBC.xtc -o rmsd.xvg -tu ns", "RMSD vs time (select Backbone)"),
            ("rmsf (RMSF)", "gmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res", "Per-residue RMSF"),
            ("gyrate (Rg)", "gmx gyrate -s md.tpr -f md_noPBC.xtc -o gyration.xvg", "Radius of gyration"),
            ("hbond", "gmx hbond -s md.tpr -f md_noPBC.xtc -num hbond.xvg -tu ns", "H-bond count (select Protein, then LIG)"),
            ("sasa", "gmx sasa -s md.tpr -f md_noPBC.xtc -o sasa.xvg -surface Protein -output LIG", "Solvent accessible surface area"),
            ("mindist", "gmx mindist -s md.tpr -f md_noPBC.xtc -od mindist.xvg -tu ns", "Minimum distance protein-ligand"),
            ("angle", "gmx angle -s md.tpr -f md_noPBC.xtc -ov angle.xvg", "Dihedral angles over time"),
        ],
        "🌡️ Energy Extraction": [
            ("energy (potential)", "gmx energy -f em.edr -o potential.xvg  # Select: Potential", "Potential energy (EM check)"),
            ("energy (temperature)", "gmx energy -f nvt.edr -o temp.xvg  # Select: Temperature", "Temperature (NVT check)"),
            ("energy (pressure)", "gmx energy -f npt.edr -o pressure.xvg  # Select: Pressure", "Pressure (NPT check)"),
            ("energy (density)", "gmx energy -f npt.edr -o density.xvg  # Select: Density", "Density (NPT check, ~1000 kg/m³)"),
            ("energy (LJ+Coul)", "gmx energy -f md.edr -o interaction.xvg  # Select: LJ-SR:Prot-LIG, Coul-SR:Prot-LIG", "Protein-ligand interaction energy"),
        ],
        "⚗️ MM-PBSA": [
            ("install", "conda install -c conda-forge gmx_mmpbsa", "Install gmx_MMPBSA"),
            ("run MM-PBSA", "gmx_MMPBSA -O -i mmpbsa.in -cs md.tpr -ct md_noPBC.xtc -ci index.ndx -cg 1 13 -cp topol.top", "Run MM-PBSA calculation"),
            ("run MM-GBSA", "gmx_MMPBSA -O -i mmgbsa.in -cs md.tpr -ct md_noPBC.xtc -ci index.ndx -cg 1 13 -cp topol.top", "Run MM-GBSA (faster, less accurate)"),
            ("reanalyze", "gmx_MMPBSA_ana -p _GMXMMPBSA_info", "Reanalyze with GUI (requires Tk)"),
        ],
        "🔗 Ligand Tools": [
            ("acpype", "acpype -i ligand.pdb -c bcc -n 0", "Generate GAFF2 topology (AMBER)"),
            ("acpype (charged)", "acpype -i ligand.pdb -c bcc -n -1", "Negatively charged ligand (net -1)"),
            ("obabel (mol2→pdb)", "obabel ligand.mol2 -O ligand.pdb", "Convert mol2 to PDB"),
            ("obabel (sdf→pdb)", "obabel ligand.sdf -O ligand.pdb --gen3D", "Convert SDF to PDB with 3D coords"),
        ],
    }

    found_any = False
    for category, entries in CHEATSHEET.items():
        if search:
            filtered = [e for e in entries
                        if search.lower() in e[0].lower()
                        or search.lower() in e[1].lower()
                        or search.lower() in e[2].lower()]
            if not filtered:
                continue
            entries = filtered

        found_any = True
        with st.expander(category, expanded=bool(search)):
            for cmd_name, cmd_text, description in entries:
                col_name, col_code = st.columns([1, 3])
                with col_name:
                    st.markdown(f"**`{cmd_name}`**")
                    st.caption(description)
                with col_code:
                    st.code(cmd_text, language="bash")
                st.markdown("---")

    if search and not found_any:
        st.info(f"No commands found matching '**{search}**'. Try a different search term.")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR & ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
PAGES = {
    "🏠 Home": page_home,
    "📚 MD Workflow Guide": page_workflow,
    "📊 RMSD / RMSF Viewer": page_rmsd_rmsf,
    "⚗️ MM-PBSA Analysis": page_mmpbsa,
    "🔬 3D Structure Viewer": page_structure_viewer,
    "🔗 H-Bond Analysis": page_hbond,
    "📋 GROMACS Cheatsheet": page_cheatsheet,
}

with st.sidebar:
    st.markdown('<div class="sidebar-header">🧬 MD Sim Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Post-Docking Analysis Toolkit</div>', unsafe_allow_html=True)
    st.markdown("---")
    page_sel = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Force Fields:** AMBER99SB-ILDN · GAFF2")
    st.markdown("**MD Engine:** GROMACS 2024")
    st.markdown("**Free Energy:** gmx_MMPBSA")
    st.markdown("---")
    st.caption("Built with Streamlit · Deploy on Streamlit Cloud")

PAGES[page_sel]()
