# 🧬 MD Simulation Explorer

An interactive web application for **learning and analyzing Molecular Dynamics (MD) simulations** after molecular docking — built with Streamlit.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## ✨ Features

| Page | Description |
|------|-------------|
| 🏠 **Home** | Overview of the full MD simulation workflow |
| 📚 **MD Workflow Guide** | Step-by-step GROMACS tutorial (12 steps) |
| 📊 **RMSD / RMSF Viewer** | Upload `.xvg` files → interactive stability plots |
| ⚗️ **MM-PBSA Analysis** | Binding free energy visualization |
| 🔬 **3D Structure Viewer** | Interactive protein-ligand PDB viewer |
| 🔗 **H-Bond Analysis** | H-bond count & residue occupancy |
| 📋 **GROMACS Cheatsheet** | Searchable command reference |

---

## 🚀 Quick Start (Local)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/md-sim-explorer.git
cd md-sim-explorer
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit: MD Simulation Explorer"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/md-sim-explorer.git
git push -u origin main
```

2. **Deploy:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click **"New app"**
   - Select your repository, branch: `main`, file: `app.py`
   - Click **"Deploy"**

Your app will be live at: `https://YOUR_USERNAME-md-sim-explorer.streamlit.app`

---

## 📂 Project Structure

```
md-sim-explorer/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── .streamlit/
    └── config.toml           # Streamlit theme configuration
```

---

## 📎 Supported Input File Formats

### RMSD / RMSF / H-Bond pages
Upload GROMACS `.xvg` output files:
```bash
# RMSD
gmx rms -s md.tpr -f md_noPBC.xtc -o rmsd_protein.xvg -tu ns

# RMSF
gmx rmsf -s md.tpr -f md_noPBC.xtc -o rmsf.xvg -res

# H-Bond count
gmx hbond -s md.tpr -f md_noPBC.xtc -num hbond_num.xvg -tu ns
```

### MM-PBSA page
Upload a CSV with columns: `Component`, `Mean (kcal/mol)`, `Std Dev`

```csv
Component,Mean (kcal/mol),Std Dev
ΔEvdW,-32.45,2.10
ΔEelec,-18.72,3.40
ΔGpolar,24.31,1.80
ΔGnonpolar,-3.82,0.40
ΔGbind,-30.68,2.90
```

### 3D Structure Viewer
Upload any `.pdb` file (docked complex or MD snapshot):
```bash
# Extract a frame from MD trajectory
gmx trjconv -s md.tpr -f md_noPBC.xtc -o last_frame.pdb -dump 100000
```

---

## 🔬 Full MD Workflow (GROMACS)

```
Molecular Docking (AutoDock Vina / Glide)
          ↓
Ligand Topology (ACPYPE / CGenFF)
          ↓
Protein Topology (pdb2gmx)
          ↓
System Setup (editconf + solvate + genion)
          ↓
Energy Minimization (gmx mdrun)
          ↓
NVT Equilibration (300 K, 200 ps)
          ↓
NPT Equilibration (1 bar, 200 ps)
          ↓
Production MD (50–200 ns)
          ↓
Analysis (RMSD, RMSF, H-bonds, MM-PBSA)
```

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) 1.35+
- **3D Visualization**: [3Dmol.js](https://3dmol.org/) (via HTML component)
- **Plotting**: [Plotly](https://plotly.com/python/)
- **Data**: [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/)

---

## 📧 Contact & Contribution

Feel free to open issues or submit pull requests for:
- New analysis modules (Rg, SASA, energy plots)
- Additional file format support
- Bug fixes

---

## 📄 License

MIT License — free to use and modify.
