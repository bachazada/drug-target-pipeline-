"""
Week 9 - Streamlit Dashboard
AI Drug Target Discovery — K. pneumoniae
Author: Bacha Zada | University of Göttingen
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Target Discovery — K. pneumoniae",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{font-size:2rem;font-weight:700;color:#1a5276;margin-bottom:0}
.sub-header{color:#666;font-size:1rem;margin-top:0;margin-bottom:1.5rem}
.metric-card{background:#f8f9fa;border-radius:10px;padding:1rem;
             text-align:center;border:1px solid #e9ecef}
.metric-val{font-size:2rem;font-weight:700;color:#1a5276}
.metric-lab{font-size:0.85rem;color:#666;margin-top:0.2rem}
.section-title{font-size:1.3rem;font-weight:600;color:#1a5276;
               border-bottom:2px solid #1a5276;padding-bottom:0.3rem;
               margin-top:1.5rem}
.info-box{background:#e8f4fd;border-left:4px solid #1a5276;
          padding:0.8rem 1rem;border-radius:0 8px 8px 0;
          font-size:0.9rem;margin:0.5rem 0}
.warn-box{background:#fff8e1;border-left:4px solid #f9a825;
          padding:0.8rem 1rem;border-radius:0 8px 8px 0;
          font-size:0.9rem;margin:0.5rem 0}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_csv(path):
    try:
        df = pd.read_csv(path)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str)
        return df
    except Exception:
        return pd.DataFrame()


def strip_rank(df):
    if df.empty or "gene" not in df.columns:
        return df
    df = df.copy()
    df["gene"] = df["gene"].astype(str).str.replace(r"^\d+_", "", regex=True).str.strip()
    return df


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    combined = strip_rank(safe_csv("results/combined_results.csv"))
    ml       = strip_rank(safe_csv("results/target_scores.csv"))
    plddt    = strip_rank(safe_csv("results/structure_results_local.csv"))
    docking  = strip_rank(safe_csv("results/docking_scores.csv"))

    # Pockets: resolve gene_clean vs gene
    pk_raw = safe_csv("results/pocket_summary.csv")
    if not pk_raw.empty:
        if "gene_clean" in pk_raw.columns and "gene" not in pk_raw.columns:
            pk_raw = pk_raw.rename(columns={"gene_clean": "gene"})
        elif "gene_clean" in pk_raw.columns and "gene" in pk_raw.columns:
            pk_raw = pk_raw.drop(columns=["gene"]).rename(columns={"gene_clean": "gene"})
    pockets = strip_rank(pk_raw)

    # Dedup each on gene
    def dedup(df, sort_col=None, asc=False):
        if df.empty or "gene" not in df.columns:
            return df
        if sort_col and sort_col in df.columns:
            df = df.copy()
            df[sort_col] = to_num(df[sort_col])
            df = df.sort_values(sort_col, ascending=asc)
        return df.drop_duplicates("gene").reset_index(drop=True)

    ml      = dedup(ml,      "druggability_score", asc=False)
    plddt   = dedup(plddt,   "mean_plddt",         asc=False)
    pockets = dedup(pockets, "best_pocket_score",  asc=False)
    docking = dedup(docking, "best_affinity",       asc=True)

    # Numeric coerce for combined
    for col in ["druggability_score","mean_plddt","best_pocket_score",
                "best_affinity","ref_affinity","delta_vs_ref","final_score"]:
        if col in combined.columns:
            combined[col] = to_num(combined[col])

    return combined, ml, plddt, pockets, docking


combined, ml_df, plddt_df, pocket_df, docking_df = load_all()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧬 Navigation")
    page = st.radio(
        "Select page",
        ["🏠 Overview", "🤖 ML Predictions", "🔬 Structures",
         "🕳️ Binding Pockets", "💊 Docking Results",
         "📊 Full Ranking", "ℹ️ Methods"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Project**")
    st.markdown("🦠 *K. pneumoniae* MGH 78578")
    st.markdown("🎓 Univ. of Göttingen")
    st.markdown("👤 Bacha Zada · 2026")
    st.markdown("---")
    st.markdown("**Tools**")
    for t in ["BLAST 2.14", "scikit-learn RF", "ColabFold",
               "P2Rank 2.4", "AutoDock Vina", "Snakemake"]:
        st.markdown(f"• {t}")


# ─────────────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🧬 AI Drug Target Discovery Pipeline</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Klebsiella pneumoniae · M.Sc. Computational Biology · University of Göttingen</p>',
                unsafe_allow_html=True)

    # Metrics
    n_p1     = int((ml_df["priority"] == "Priority 1 (high)").sum()) \
               if not ml_df.empty else 14
    n_struct = len(plddt_df)
    n_docked = len(docking_df)
    best_aff = float(to_num(docking_df["best_affinity"]).min()) \
               if not docking_df.empty else -8.36

    for col, (val, lab) in zip(st.columns(5), [
        ("5,126",        "Proteome proteins"),
        ("95",           "Candidate targets"),
        (str(n_p1),      "Priority 1 (ML)"),
        (str(n_struct),  "Structures predicted"),
        (f"{best_aff:.1f}", "Best docking (kcal/mol)"),
    ]):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-lab">{lab}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Pipeline Steps</p>',
                unsafe_allow_html=True)

    # Pipeline diagram — use rgba() strings, NOT hex+"22"
    steps = [
        ("Week 1", "Data Collection",
         "UniProt 5,126 proteins + DEG 154 essential genes",
         "#1a5276", "rgba(26,82,118,0.13)"),
        ("Week 2", "Biological Filtering",
         "BLAST homolog removal → 95 candidates",
         "#1a5276", "rgba(26,82,118,0.13)"),
        ("Week 3", "Feature Engineering",
         "44 physicochemical features per protein",
         "#6c3483", "rgba(108,52,131,0.13)"),
        ("Week 4", "ML Model",
         "Random Forest · CV AUC 0.725 · 14 Priority 1 targets",
         "#6c3483", "rgba(108,52,131,0.13)"),
        ("Week 5", "Structure Prediction",
         "ColabFold · 13 structures · pLDDT ≥ 87.5",
         "#1d6a39", "rgba(29,106,57,0.13)"),
        ("Week 6", "Pocket Detection",
         "P2Rank 2.4 (AlphaFold config) · 13/13",
         "#1d6a39", "rgba(29,106,57,0.13)"),
        ("Week 7", "Molecular Docking",
         "AutoDock Vina · 8 real results · best -8.36 kcal/mol",
         "#7b241c", "rgba(123,36,28,0.13)"),
    ]

    fig = go.Figure()
    n = len(steps)
    for i, (week, title, desc, border_col, fill_col) in enumerate(steps):
        y = n - i
        fig.add_shape(
            type="rect", x0=0.05, x1=0.95,
            y0=y - 0.33, y1=y + 0.33,
            fillcolor=fill_col,                  # rgba string — always valid
            line=dict(color=border_col, width=1.5),
        )
        fig.add_annotation(x=0.13, y=y, text=f"<b>{week}</b>",
                           font=dict(size=11, color=border_col), showarrow=False)
        fig.add_annotation(x=0.32, y=y + 0.10, text=f"<b>{title}</b>",
                           font=dict(size=11, color="#222"),
                           showarrow=False, xanchor="left")
        fig.add_annotation(x=0.32, y=y - 0.13, text=desc,
                           font=dict(size=9, color="#555"),
                           showarrow=False, xanchor="left")
        if i < n - 1:
            fig.add_annotation(
                x=0.5, y=y - 0.33, ay=y - 0.52, ax=0.5,
                arrowhead=2, arrowcolor="#bbb", showarrow=True,
            )

    fig.update_layout(
        height=450,
        margin=dict(l=5, r=5, t=5, b=5),
        xaxis=dict(visible=False, range=[0, 1]),
        yaxis=dict(visible=False, range=[0, n + 0.6]),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    if not combined.empty:
        st.markdown('<p class="section-title">Top 5 Candidates</p>',
                    unsafe_allow_html=True)
        show = [c for c in ["final_rank","gene","druggability_score",
                             "mean_plddt","best_pocket_score",
                             "best_affinity","final_score","ligand"]
                if c in combined.columns]
        st.dataframe(combined[show].head(5),
                     use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="info-box">
    <b>Key validation:</b> fabI/triclosan docked at -7.67 kcal/mol vs published -8.1 —
    within docking error margin. The pipeline independently reproduced a published
    experimental result starting from raw protein sequences only.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ML PREDICTIONS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Predictions":
    st.markdown('<p class="section-title">🤖 ML Druggability Predictions</p>',
                unsafe_allow_html=True)

    if ml_df.empty:
        st.warning("target_scores.csv not found")
    else:
        ml_df["druggability_score"] = to_num(ml_df["druggability_score"])

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.histogram(
                ml_df, x="druggability_score", nbins=25,
                color_discrete_sequence=["#1a5276"],
                title="Druggability score distribution (n=93)",
                labels={"druggability_score": "Score"},
            )
            fig.add_vline(x=0.75, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Priority 1")
            fig.add_vline(x=0.50, line_dash="dash", line_color="#f39c12",
                          annotation_text="Priority 2")
            fig.update_layout(height=340)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            counts = ml_df["priority"].value_counts()
            fig2 = px.pie(
                values=counts.values, names=counts.index,
                color_discrete_sequence=["#e74c3c","#f39c12","#3498db"],
                height=300, title="Priority breakdown",
            )
            fig2.update_traces(textinfo="percent+label")
            st.plotly_chart(fig2, use_container_width=True)

        top20 = ml_df.sort_values("druggability_score", ascending=False).head(20)
        cmap = {"Priority 1 (high)": "#e74c3c",
                "Priority 2 (medium)": "#f39c12",
                "Priority 3 (low)": "#3498db"}
        fig3 = px.bar(
            top20, x="gene", y="druggability_score",
            color="priority", color_discrete_map=cmap,
            title="Top 20 targets by ML druggability score",
        )
        fig3.add_hline(y=0.75, line_dash="dash", line_color="#e74c3c")
        fig3.update_layout(xaxis_tickangle=-45, height=370)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>Model:</b> Random Forest (300 trees, balanced class weights)<br>
        <b>CV ROC-AUC:</b> 0.725 ± 0.019 (5-fold stratified)<br>
        <b>Features:</b> 44 physicochemical (AA composition, MW, pI, hydrophobicity…)
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Structures":
    st.markdown('<p class="section-title">🔬 Structure Prediction (ColabFold)</p>',
                unsafe_allow_html=True)

    if plddt_df.empty:
        st.warning("structure_results_local.csv not found")
    else:
        plddt_df["mean_plddt"] = to_num(plddt_df["mean_plddt"])
        plddt_df["pct_confident"] = to_num(plddt_df["pct_confident"])
        df = plddt_df.sort_values("mean_plddt", ascending=False)

        col1, col2 = st.columns(2)
        cmap2 = {"very_high": "#1d6a39", "high": "#5dade2",
                 "medium": "#f39c12",    "low": "#e74c3c"}
        with col1:
            fig = px.bar(df, x="gene", y="mean_plddt",
                         color="plddt_status", color_discrete_map=cmap2,
                         title="Mean pLDDT per structure")
            fig.add_hline(y=90, line_dash="dash", line_color="#1d6a39",
                          annotation_text="90 very high")
            fig.add_hline(y=70, line_dash="dash", line_color="#f39c12",
                          annotation_text="70 confident")
            fig.update_layout(height=380, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(df, x="mean_plddt", y="pct_confident",
                              text="gene", color="plddt_status",
                              color_discrete_map=cmap2,
                              title="pLDDT vs % confident residues")
            fig2.update_traces(textposition="top center")
            fig2.update_layout(height=380)
            st.plotly_chart(fig2, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Structures", len(df))
        c2.metric("pLDDT ≥ 90", int((df["mean_plddt"] >= 90).sum()))
        c3.metric("Mean pLDDT", f"{df['mean_plddt'].mean():.1f}")
        best_i = df["mean_plddt"].idxmax()
        c4.metric("Best", f"{df.loc[best_i,'mean_plddt']:.1f} ({df.loc[best_i,'gene']})")

        show_cols = [c for c in ["gene","mean_plddt","pct_confident","plddt_status"]
                     if c in df.columns]
        st.dataframe(df[show_cols].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
        <b>Tool:</b> ColabFold (AlphaFold2_ptm, 3 recycles, T4 GPU — Google Colab free)<br>
        <b>All 13 structures passed</b> pLDDT ≥ 70 quality cutoff (range 87.5–96.8)
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BINDING POCKETS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🕳️ Binding Pockets":
    st.markdown('<p class="section-title">🕳️ Binding Pocket Detection (P2Rank)</p>',
                unsafe_allow_html=True)

    if pocket_df.empty:
        st.warning("pocket_summary.csv not found")
    else:
        pocket_df["best_pocket_score"] = to_num(pocket_df["best_pocket_score"])
        pocket_df["n_pockets"]         = to_num(pocket_df["n_pockets"])
        df = pocket_df.sort_values("best_pocket_score", ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            scores = df["best_pocket_score"].tolist()
            bar_cols = ["#e74c3c" if s >= 40 else "#f39c12" if s >= 20
                        else "#3498db" for s in scores]
            fig = go.Figure(go.Bar(
                x=scores, y=df["gene"].tolist(),
                orientation="h", marker_color=bar_cols,
                text=[f"{v:.1f}" for v in scores],
                textposition="outside",
            ))
            fig.update_layout(title="Best pocket score (P2Rank)",
                              xaxis_title="Score", height=420,
                              yaxis_autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.scatter(
                df, x="n_pockets", y="best_pocket_score", text="gene",
                title="Pockets found vs best score",
                labels={"n_pockets": "# pockets",
                        "best_pocket_score": "Best score"},
            )
            fig2.update_traces(textposition="top center")
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)

        coord_cols = [c for c in ["gene","best_pocket_score","n_pockets",
                                   "center_x","center_y","center_z"]
                      if c in df.columns]
        st.dataframe(df[coord_cols].reset_index(drop=True),
                     use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
        <b>Tool:</b> P2Rank v2.4 with <code>-c alphafold</code> — ignores B-factors
        (ColabFold stores pLDDT there, not crystallographic temperature).<br>
        <b>Score guide:</b> >40 excellent · 20–40 good · &lt;20 moderate
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOCKING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💊 Docking Results":
    st.markdown('<p class="section-title">💊 Molecular Docking (AutoDock Vina)</p>',
                unsafe_allow_html=True)

    if docking_df.empty:
        st.warning("docking_scores.csv not found")
    else:
        for col in ["best_affinity","ref_affinity","delta_vs_ref"]:
            if col in docking_df.columns:
                docking_df[col] = to_num(docking_df[col])
        df = docking_df.sort_values("best_affinity")

        col1, col2 = st.columns(2)
        with col1:
            affs = df["best_affinity"].tolist()
            refs = df["ref_affinity"].tolist()
            bar_c = ["#1d6a39" if float(v) <= float(r) else "#e74c3c"
                     for v, r in zip(affs, refs)]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=df["gene"].tolist(), x=affs, orientation="h",
                name="Our docking", marker_color=bar_c,
                text=[f"{v:.2f}" for v in affs], textposition="outside",
            ))
            fig.add_trace(go.Scatter(
                y=df["gene"].tolist(), x=refs, mode="markers",
                name="Published ref",
                marker=dict(symbol="diamond", size=10, color="#f39c12"),
            ))
            fig.add_vline(x=-7.0, line_dash="dash", line_color="gray",
                          annotation_text="-7 drug-like")
            fig.update_layout(
                title="Binding affinity vs published",
                xaxis_title="kcal/mol (more negative = stronger)",
                height=380, yaxis_autorange="reversed",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            deltas = df["delta_vs_ref"].tolist()
            fig2 = go.Figure(go.Bar(
                y=df["gene"].tolist(), x=deltas, orientation="h",
                marker_color=["#1d6a39" if float(d) <= 0 else "#e74c3c"
                              for d in deltas],
                text=[f"{float(d):+.2f}" for d in deltas],
                textposition="outside",
            ))
            fig2.add_vline(x=0, line_color="black", line_width=1)
            fig2.update_layout(
                title="Δ vs published reference",
                xaxis_title="Δ kcal/mol (negative = better)",
                height=380, yaxis_autorange="reversed",
            )
            st.plotly_chart(fig2, use_container_width=True)

        show = [c for c in ["gene","ligand","best_affinity",
                             "ref_affinity","delta_vs_ref"] if c in df.columns]
        display = df[show].copy()
        display["result"] = display["delta_vs_ref"].apply(
            lambda d: "✅ Better" if float(d) <= 0 else "⚠️ Weaker")
        st.dataframe(display.reset_index(drop=True),
                     use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="info-box">
        <b>fabI/triclosan:</b> -7.67 kcal/mol vs published -8.1 — strong pipeline validation.<br>
        <b>ftsI/ampicillin:</b> -8.36 kcal/mol — exceeds published -6.5. PBP3 is the direct beta-lactam target.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="warn-box">
        <b>Known limitations:</b>
        murA/fosfomycin (-4.42): covalent inhibitor — underestimated by non-covalent docking. &nbsp;
        gyrB/novobiocin (-2.42): box center mismatch. &nbsp;
        groEL + pheT: timeout due to large protein size.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FULL RANKING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Full Ranking":
    st.markdown('<p class="section-title">📊 Full Combined Ranking</p>',
                unsafe_allow_html=True)

    if combined.empty:
        st.warning("combined_results.csv not found. Run Week 8 setup first.")
    else:
        c1, c2, c3 = st.columns(3)
        min_ml       = c1.slider("Min ML score", 0.0, 1.0, 0.5, 0.05)
        only_struct  = c2.checkbox("Only proteins with structure")
        only_docking = c3.checkbox("Only proteins with docking")

        df = combined[to_num(combined["druggability_score"]) >= min_ml].copy()
        if only_struct and "mean_plddt" in df.columns:
            df = df[df["mean_plddt"].notna()]
        if only_docking and "best_affinity" in df.columns:
            df = df[df["best_affinity"].notna()]

        st.markdown(f"**{len(df)} proteins shown**")

        # Bubble chart
        plot_df = df.dropna(subset=["best_pocket_score"]).head(15).copy()
        if not plot_df.empty:
            aff_num = to_num(plot_df.get("best_affinity",
                             pd.Series([0]*len(plot_df)))).fillna(0)
            size_v  = np.abs(aff_num).clip(1, 15).tolist()
            color_v = to_num(plot_df.get("final_score",
                             pd.Series([0.5]*len(plot_df)))).fillna(0.5)
            fig = px.scatter(
                plot_df,
                x="druggability_score", y="best_pocket_score",
                size=size_v, color=color_v,
                text="gene", color_continuous_scale="RdYlGn",
                title="Drug target landscape (bubble = |docking affinity|)",
                labels={"druggability_score": "ML score",
                        "best_pocket_score": "Pocket score",
                        "color": "Final score"},
                height=460,
            )
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, use_container_width=True)

        show_cols = [c for c in ["final_rank","gene","priority",
                                  "druggability_score","mean_plddt",
                                  "best_pocket_score","best_affinity",
                                  "ligand","final_score"]
                     if c in df.columns]
        st.dataframe(
            df[show_cols].reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={
                "final_score": st.column_config.ProgressColumn(
                    "Final score", min_value=0, max_value=1),
                "druggability_score": st.column_config.ProgressColumn(
                    "ML score", min_value=0, max_value=1),
                "best_affinity": st.column_config.NumberColumn(
                    "Docking (kcal/mol)", format="%.2f"),
            },
        )
        st.download_button(
            "⬇️ Download filtered results (CSV)",
            df[show_cols].to_csv(index=False),
            "drug_targets.csv", "text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# METHODS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ Methods":
    st.markdown('<p class="section-title">ℹ️ Methods & Pipeline Details</p>',
                unsafe_allow_html=True)

    methods_path = Path("docs/METHODS.md")
    if methods_path.exists():
        st.markdown(methods_path.read_text())
    else:
        st.info("Run `python scripts/08_pipeline_setup.py` to generate docs/METHODS.md")

    st.markdown("---")
    st.markdown("### Cite this pipeline")
    st.code(
        "Zada B. (2026). AI-Assisted Drug Target Discovery Pipeline for\n"
        "Klebsiella pneumoniae. M.Sc. Computational Biology & Bioinformatics,\n"
        "University of Göttingen. GitHub: github.com/bachazada/drug-target-pipeline"
    )