import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import glob
import os
import re
import sys

# ==========================================
# 1. USER SETTINGS
# ==========================================
REACTOR_TYPE   = 'lmfr'                     # Change to 'pwr', 'bwr', 'phwr', etc.
SHEET_NAME     = 'End of Cycle'
MASS_FACTOR    = 2528                       # Conversion factor for total Pu mass → kg

# Reactor display names (add new types here as needed)
REACTOR_LABELS = {
    'agr':  'AGR  (Advanced Gas-cooled Reactor)',
    'pwr':  'PWR  (Pressurized Water Reactor)',
    'phwr': 'PHWR (Pressurized Heavy Water Reactor)',
    'lmfr': 'LMFR  (Liquid Metal Fast Reactor)',
}

ISOTOPES = ['Pu238', 'Pu239', 'Pu240', 'Pu241', 'Pu242']
MARKERS  = ['o', 's', '^', 'D', 'v']
COLORS   = ['#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# LMFR zone configuration  (prefix → display name)
# When REACTOR_TYPE='lmfr', the script also processes axial_ and radial_ files.
LMFR_ZONES = {
    'lmfr':   'Core',
    'axial':  'Axial Blanket',
    'radial': 'Radial Blanket',
}
ZONE_COLORS  = {'lmfr': '#1f77b4', 'axial': '#ff7f0e', 'radial': '#2ca02c'}
ZONE_MARKERS = {'lmfr': 'o',       'axial': 's',       'radial': '^'}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def _detect_enrichment(filename):
    """Extract enrichment value from filename like 'agr_4.5.xlsx'.
    Handles special cases like 'agr_0711.xlsx' → 0.711 (natural U).
    Returns None for non-numeric suffixes (e.g. 'ma4')."""
    base = os.path.splitext(os.path.basename(filename))[0]        # agr_4.5
    match = re.search(r'_(\d+\.?\d*)', base)
    if not match:
        return None
    raw = match.group(1)
    # Detect leading-zero convention: '0711' → 0.711
    if raw.startswith('0') and len(raw) > 1 and '.' not in raw:
        return float(raw) / 1000.0
    return float(raw)


def _detect_case_label(filename):
    """Extract a human-readable case label for non-numeric files like 'lmfr_ma4.xlsx'.
    Returns e.g. 'MA-4', 'MA', 'MA-2', or None if it's a normal enrichment file."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Check for pattern like _ma, _ma2, _ma4
    match = re.search(r'_([a-zA-Z]+\d*)\s*$', base)
    if match:
        raw = match.group(1).upper()
        # Format nicely: 'MA4' → 'MA-4', 'MA' → 'MA'
        m2 = re.match(r'([A-Z]+)(\d+)', raw)
        if m2:
            return f"{m2.group(1)}-{m2.group(2)}"
        return raw
    return None


def _load_single_file(file_path):
    """Load one NFCSS Excel file and return a cleaned DataFrame."""
    df_raw = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=None)
    header_row_idx = df_raw.index[df_raw.iloc[:, 0] == "DataYear"][0]
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=header_row_idx)

    needed = ['Burnup', 'TotPu'] + ISOTOPES
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns {missing} in {file_path}")

    for col in needed:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('Burnup').reset_index(drop=True)

    # Isotopic fractions
    for iso in ISOTOPES:
        df[f'{iso}_frac'] = np.where(
            (df['TotPu'] > 1e-4) & (df[iso] > 1e-4),
            (df[iso] / df['TotPu']) * 100,
            0.0,
        )
    return df


def _apply_style(ax, title, xlabel, ylabel, logx=False):
    """Apply a consistent publication-quality style to an axes."""
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=10, direction='in', top=True, right=True)
    ax.grid(True, alpha=0.25, linestyle='--')
    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)


# ==========================================
# 3. SINGLE-FILE PLOTS  (per enrichment)
# ==========================================
def plot_single_enrichment(df, enrichment, reactor, out_dir):
    """Generate all individual plots for ONE enrichment level."""
    label = REACTOR_LABELS.get(reactor.lower(), reactor.upper())
    is_numeric = isinstance(enrichment, (int, float))
    tag   = f"{reactor.upper()}_{enrichment}%" if is_numeric else f"{reactor.upper()}_{enrichment}"
    enr_str = f"Enrichment = {enrichment} %" if is_numeric else f"Case: {enrichment}"

    # ---- 3a. All isotopic fractions on one plot ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, iso in enumerate(ISOTOPES):
        ax.plot(df['Burnup'], df[f'{iso}_frac'],
                label=iso, marker=MARKERS[i], color=COLORS[i],
                linewidth=2, markersize=5, markeredgecolor='white', markeredgewidth=0.4)
    ax.axhline(y=93, color=COLORS[1], ls='--', alpha=0.45, label='WG Limit (Pu-239 > 93 %)')
    _apply_style(ax,
                 f"Pu Isotopic Quality vs Burnup — {label}\n{enr_str}",
                 "Burnup (GWd/t)", "Isotopic Fraction (% of Total Pu)", logx=True)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, framealpha=0.9, loc='center left', bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tag}_all_isotopes.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved {tag}_all_isotopes.png")

    # ---- 3b. Individual isotope plots ----
    for i, iso in enumerate(ISOTOPES):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df['Burnup'], df[f'{iso}_frac'],
                marker=MARKERS[i], color=COLORS[i],
                linewidth=2.2, markersize=6, markeredgecolor='white', markeredgewidth=0.5)
        ax.fill_between(df['Burnup'], df[f'{iso}_frac'], alpha=0.10, color=COLORS[i])
        _apply_style(ax,
                     f"{iso} Fraction vs Burnup — {label}\n{enr_str}",
                     "Burnup (GWd/t)", f"{iso} (% of Total Pu)", logx=True)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'{tag}_{iso}.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved {tag}_{iso}.png")

    # ---- 3c. Total Pu production ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['Burnup'], df['TotPu'] * MASS_FACTOR,
            color='black', marker='o', linewidth=2.2, markersize=5,
            markeredgecolor='white', markeredgewidth=0.5)
    ax.fill_between(df['Burnup'], df['TotPu'] * MASS_FACTOR, alpha=0.08, color='black')
    _apply_style(ax,
                 f"Total Pu Production vs Burnup — {label}\n{enr_str}",
                 "Burnup (GWd/t)", "Total Pu (kg)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{tag}_TotalPu.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved {tag}_TotalPu.png")


# ==========================================
# 4. PARAMETRIC STUDY PLOTS  (all enrichments)
# ==========================================
def plot_parametric_study(all_data, reactor, out_dir):
    """
    Generate comparison plots across every enrichment level.
    all_data : dict  {enrichment: DataFrame}
    """
    label = REACTOR_LABELS.get(reactor.lower(), reactor.upper())
    cmap  = plt.cm.viridis
    enrichments = sorted(all_data.keys())
    norm  = plt.Normalize(vmin=min(enrichments), vmax=max(enrichments))

    # ---- 4a. Each isotope: fraction vs burnup for all enrichments ----
    for iso in ISOTOPES:
        fig, ax = plt.subplots(figsize=(10, 6))
        for enr in enrichments:
            df = all_data[enr]
            ax.plot(df['Burnup'], df[f'{iso}_frac'],
                    label=f'{enr} %', color=cmap(norm(enr)),
                    linewidth=1.8, marker='o', markersize=3.5)
        _apply_style(ax,
                     f"{iso} Fraction vs Burnup — {label}\nParametric Study (Enrichment Sweep)",
                     "Burnup (GWd/t)", f"{iso} (% of Total Pu)", logx=True)
        ax.set_ylim(bottom=0)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Enrichment (%)', fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'parametric_{reactor.upper()}_{iso}.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved parametric_{reactor.upper()}_{iso}.png")

    # ---- 4b. Total Pu vs burnup for all enrichments ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for enr in enrichments:
        df = all_data[enr]
        ax.plot(df['Burnup'], df['TotPu'] * MASS_FACTOR,
                label=f'{enr} %', color=cmap(norm(enr)),
                linewidth=1.8, marker='o', markersize=3.5)
    _apply_style(ax,
                 f"Total Pu Production vs Burnup — {label}\nParametric Study (Enrichment Sweep)",
                 "Burnup (GWd/t)", "Total Pu (kg)", logx=False)
    ax.set_ylim(bottom=0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Enrichment (%)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'parametric_{reactor.upper()}_TotalPu.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved parametric_{reactor.upper()}_TotalPu.png")

    # ---- 4c. Combined all-isotope subplot grid ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    for idx, iso in enumerate(ISOTOPES):
        ax = axes_flat[idx]
        for enr in enrichments:
            df = all_data[enr]
            ax.plot(df['Burnup'], df[f'{iso}_frac'],
                    label=f'{enr} %', color=cmap(norm(enr)), linewidth=1.4)
        _apply_style(ax, iso, "Burnup (GWd/t)", "Fraction (%)", logx=True)
        ax.set_ylim(bottom=0)

    # Use last subplot for Total Pu
    ax = axes_flat[5]
    for enr in enrichments:
        df = all_data[enr]
        ax.plot(df['Burnup'], df['TotPu'] * MASS_FACTOR,
                label=f'{enr} %', color=cmap(norm(enr)), linewidth=1.4)
    _apply_style(ax, "Total Pu (kg)", "Burnup (GWd/t)", "kg")
    ax.set_ylim(bottom=0)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat.tolist(), pad=0.01, shrink=0.6)
    cbar.set_label('Enrichment (%)', fontsize=12)

    fig.suptitle(f"Parametric Study — {label}", fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(out_dir, f'parametric_{reactor.upper()}_summary.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved parametric_{reactor.upper()}_summary.png")

    # ---- 4d. Heatmap: Pu-239 fraction at each (enrichment, burnup) ----
    try:
        common_burnups = sorted(
            set.intersection(*[set(all_data[e]['Burnup'].round(2)) for e in enrichments])
        )
        if len(common_burnups) >= 3:
            matrix = np.zeros((len(enrichments), len(common_burnups)))
            for i, enr in enumerate(enrichments):
                df = all_data[enr].copy()
                df['Burnup_r'] = df['Burnup'].round(2)
                for j, bu in enumerate(common_burnups):
                    row = df.loc[df['Burnup_r'] == bu, 'Pu239_frac']
                    matrix[i, j] = row.values[0] if len(row) else np.nan

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', origin='lower')
            ax.set_yticks(range(len(enrichments)))
            ax.set_yticklabels([f'{e} %' for e in enrichments])
            ax.set_xticks(range(len(common_burnups)))
            ax.set_xticklabels([f'{b:.1f}' for b in common_burnups], rotation=45, ha='right', fontsize=8)
            ax.set_xlabel("Burnup (GWd/t)", fontsize=12)
            ax.set_ylabel("Enrichment", fontsize=12)
            ax.set_title(f"Pu-239 Fraction (%) — {label}", fontsize=14, fontweight='bold')
            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label('Pu-239 (%)', fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f'parametric_{reactor.upper()}_Pu239_heatmap.png'), dpi=300)
            plt.close(fig)
            print(f"  Saved parametric_{reactor.upper()}_Pu239_heatmap.png")
    except Exception as e:
        print(f"  (Heatmap skipped — {e})")


# ==========================================
# 5. LMFR ZONE COMPARISON PLOTS
# ==========================================
def _load_zone_data(zone_prefix, src_dir):
    """Load all files for one LMFR zone prefix → {enrichment: DataFrame}."""
    pattern = os.path.join(src_dir, f'{zone_prefix}_*.xlsx')
    files = sorted(glob.glob(pattern))
    zone_data = {}
    for fpath in files:
        enr = _detect_enrichment(fpath)
        if enr is None:
            continue
        try:
            df = _load_single_file(fpath)
            zone_data[enr] = df
        except Exception as e:
            print(f"    ERROR loading {os.path.basename(fpath)}: {e}")
    return zone_data


def plot_zone_comparison(zone_datasets, reactor, out_dir):
    """
    Compare Core / Axial Blanket / Radial Blanket for each isotope.
    zone_datasets : dict  {zone_prefix: {enrichment: DataFrame}}
    Plots each zone's files together, distinguishing zones by colour.
    """
    label = REACTOR_LABELS.get(reactor.lower(), reactor.upper())
    os.makedirs(out_dir, exist_ok=True)

    # ---- Per-isotope comparison ----
    for iso in ISOTOPES:
        fig, ax = plt.subplots(figsize=(11, 6))
        for zone_prefix, enr_dict in zone_datasets.items():
            zone_name = LMFR_ZONES.get(zone_prefix, zone_prefix)
            c = ZONE_COLORS.get(zone_prefix, 'gray')
            m = ZONE_MARKERS.get(zone_prefix, 'x')
            for enr in sorted(enr_dict.keys(), key=lambda x: (isinstance(x, str), x)):
                df = enr_dict[enr]
                lbl = f"{zone_name} ({enr} %)" if isinstance(enr, (int, float)) else f"{zone_name} ({enr})"
                ax.plot(df['Burnup'], df[f'{iso}_frac'],
                        label=lbl, color=c, marker=m,
                        linewidth=1.6, markersize=4, alpha=0.85)
        _apply_style(ax,
                     f"{iso} Fraction — {label}\nZone Comparison (Core vs Blankets)",
                     "Burnup (GWd/t)", f"{iso} (% of Total Pu)", logx=True)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, framealpha=0.9, loc='center left',
                  bbox_to_anchor=(1.02, 0.5), ncol=1)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f'zone_cmp_{iso}.png'), dpi=300)
        plt.close(fig)
        print(f"  Saved zone_cmp_{iso}.png")

    # ---- Total Pu comparison ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for zone_prefix, enr_dict in zone_datasets.items():
        zone_name = LMFR_ZONES.get(zone_prefix, zone_prefix)
        c = ZONE_COLORS.get(zone_prefix, 'gray')
        m = ZONE_MARKERS.get(zone_prefix, 'x')
        for enr in sorted(enr_dict.keys(), key=lambda x: (isinstance(x, str), x)):
            df = enr_dict[enr]
            lbl = f"{zone_name} ({enr} %)" if isinstance(enr, (int, float)) else f"{zone_name} ({enr})"
            ax.plot(df['Burnup'], df['TotPu'] * MASS_FACTOR,
                    label=lbl, color=c, marker=m,
                    linewidth=1.6, markersize=4, alpha=0.85)
    _apply_style(ax,
                 f"Total Pu Production — {label}\nZone Comparison (Core vs Blankets)",
                 "Burnup (GWd/t)", "Total Pu (kg)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, framealpha=0.9, loc='center left',
              bbox_to_anchor=(1.02, 0.5), ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'zone_cmp_TotalPu.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved zone_cmp_TotalPu.png")

    # ---- Summary subplot (2×3): one per isotope + TotPu ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    axes_flat = axes.flatten()
    items = ISOTOPES + ['TotPu']
    for idx, col_key in enumerate(items):
        ax = axes_flat[idx]
        for zone_prefix, enr_dict in zone_datasets.items():
            zone_name = LMFR_ZONES.get(zone_prefix, zone_prefix)
            c = ZONE_COLORS.get(zone_prefix, 'gray')
            for enr in sorted(enr_dict.keys(), key=lambda x: (isinstance(x, str), x)):
                df = enr_dict[enr]
                if col_key == 'TotPu':
                    y = df['TotPu'] * MASS_FACTOR
                    ylabel = 'kg'
                else:
                    y = df[f'{col_key}_frac']
                    ylabel = '%'
                ax.plot(df['Burnup'], y, color=c, linewidth=1.2, alpha=0.8)
        _apply_style(ax, col_key if col_key != 'TotPu' else 'Total Pu (kg)',
                     "Burnup (GWd/t)", ylabel, logx=True)
        ax.set_ylim(bottom=0)

    # Custom legend for zones
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=ZONE_COLORS[z], lw=2,
                       label=LMFR_ZONES[z]) for z in zone_datasets]
    fig.legend(handles=handles, loc='lower right', fontsize=11, framealpha=0.9)
    fig.suptitle(f"LMFR Zone Comparison — {label}", fontsize=16, fontweight='bold')
    fig.savefig(os.path.join(out_dir, 'zone_cmp_summary.png'), dpi=300)
    plt.close(fig)
    print(f"  Saved zone_cmp_summary.png")


# ==========================================
# 6. MAIN DRIVER
# ==========================================
def _process_file_set(files, reactor_tag, src_dir, out_dir):
    """Process a list of Excel files (enrichment sweep + named cases).
    Returns (all_data dict for enrichment files,  named_cases dict)."""
    single_dir = os.path.join(out_dir, 'per_enrichment')
    cases_dir  = os.path.join(out_dir, 'named_cases')
    param_dir  = os.path.join(out_dir, 'parametric')
    os.makedirs(single_dir, exist_ok=True)

    all_data     = {}   # {enrichment_float : DataFrame}  — numeric enrichments
    named_cases  = {}   # {'MA-4': DataFrame}             — non-numeric cases

    for fpath in files:
        enrichment = _detect_enrichment(fpath)
        case_label = _detect_case_label(fpath)

        if enrichment is not None:
            # Normal enrichment sweep file
            print(f"\n--- Enrichment = {enrichment} % ({os.path.basename(fpath)}) ---")
            try:
                df = _load_single_file(fpath)
            except Exception as e:
                print(f"  ERROR loading: {e}")
                continue
            all_data[enrichment] = df
            plot_single_enrichment(df, enrichment, reactor_tag, single_dir)

        elif case_label is not None:
            # Named case (e.g. MA-4, MA, MA-2)
            print(f"\n--- Named case: {case_label} ({os.path.basename(fpath)}) ---")
            try:
                df = _load_single_file(fpath)
            except Exception as e:
                print(f"  ERROR loading: {e}")
                continue
            named_cases[case_label] = df
            os.makedirs(cases_dir, exist_ok=True)
            plot_single_enrichment(df, case_label, reactor_tag, cases_dir)
        else:
            print(f"  SKIP: cannot parse '{os.path.basename(fpath)}'")

    # Parametric study for enrichment sweep (needs ≥ 2)
    if len(all_data) >= 2:
        os.makedirs(param_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("  Generating parametric study plots …")
        print(f"{'='*60}")
        plot_parametric_study(all_data, reactor_tag, param_dir)
    else:
        print("\n  (Need ≥ 2 enrichment levels for parametric study.)")

    return all_data, named_cases


def main(reactor_type=None):
    reactor = (reactor_type or REACTOR_TYPE).lower()
    src_dir = os.path.dirname(os.path.abspath(__file__)) or '.'

    # Discover all matching Excel files
    pattern = os.path.join(src_dir, f'{reactor}_*.xlsx')
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"ERROR: No files matching '{pattern}' found.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Reactor type : {REACTOR_LABELS.get(reactor, reactor.upper())}")
    print(f"  Files found  : {len(files)}")
    print(f"{'='*60}\n")

    # Create output directory
    out_dir = os.path.join(src_dir, f'results_{reactor.upper()}')
    os.makedirs(out_dir, exist_ok=True)

    # Process core files
    all_data, named_cases = _process_file_set(files, reactor, src_dir, out_dir)

    n_enr = len(all_data)
    n_cases = len(named_cases)
    print(f"\n✓ Core plots saved — {n_enr} enrichment levels, {n_cases} named cases")

    # ==========================================
    # LMFR-specific: process blanket zones
    # ==========================================
    if reactor == 'lmfr':
        blanket_prefixes = [z for z in LMFR_ZONES if z != 'lmfr']
        zone_datasets = {'lmfr': all_data}   # core enrichment data

        for zone_prefix in blanket_prefixes:
            zone_pattern = os.path.join(src_dir, f'{zone_prefix}_*.xlsx')
            zone_files = sorted(glob.glob(zone_pattern))
            if not zone_files:
                continue

            zone_name = LMFR_ZONES[zone_prefix]
            print(f"\n{'='*60}")
            print(f"  Processing LMFR zone: {zone_name}  ({len(zone_files)} files)")
            print(f"{'='*60}")

            blanket_dir = os.path.join(out_dir, f'{zone_prefix}_blanket')
            os.makedirs(blanket_dir, exist_ok=True)

            zone_data, zone_named = _process_file_set(
                zone_files, f"{reactor}_{zone_prefix}", src_dir, blanket_dir
            )

            if zone_data:
                zone_datasets[zone_prefix] = zone_data
            # Named blanket cases also go into zone_datasets with modified keys
            if zone_named:
                for case_lbl, case_df in zone_named.items():
                    key = f"{zone_prefix}_{case_lbl}"
                    # Add to zone_datasets so they appear in comparison
                    zone_datasets.setdefault(zone_prefix, {})[case_lbl] = case_df

        # Cross-zone comparison plots
        if len(zone_datasets) >= 2:
            print(f"\n{'='*60}")
            print("  Generating LMFR zone comparison plots …")
            print(f"{'='*60}")
            zone_cmp_dir = os.path.join(out_dir, 'zone_comparison')
            plot_zone_comparison(zone_datasets, reactor, zone_cmp_dir)

        print(f"\n✓ LMFR complete.  Output:  {out_dir}")
        print(f"  ├── per_enrichment/              (Core enrichment sweep)")
        if named_cases:
            print(f"  ├── named_cases/                 (Core MA cases: {list(named_cases.keys())})")
        for z in blanket_prefixes:
            if z in zone_datasets:
                print(f"  ├── {z}_blanket/              ({LMFR_ZONES[z]})")
        print(f"  ├── parametric/                  (Core parametric)")
        print(f"  └── zone_comparison/             (Core vs Blankets)\n")
    else:
        print(f"\n✓ All plots saved to:  {out_dir}")
        if named_cases:
            print(f"  ├── per_enrichment/   ({n_enr} enrichment levels)")
            print(f"  ├── named_cases/      ({list(named_cases.keys())})")
            print(f"  └── parametric/\n")
        else:
            print(f"  ├── per_enrichment/   ({n_enr} enrichment levels)")
            print(f"  └── parametric/\n")


if __name__ == "__main__":
    # Allow override from command line:  python const_enrich pwr
    rtype = sys.argv[1] if len(sys.argv) > 1 else None
    main(reactor_type=rtype)