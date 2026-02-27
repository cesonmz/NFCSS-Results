import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata

# ==========================================
# 1. USER SETTINGS
# ==========================================
REACTOR_TYPE = 'lmfr'                        
SHEET_NAME   = 'End of Cycle'

REACTOR_LABELS = {
    'agr':  'AGR  (Advanced Gas-cooled Reactor)',
    'pwr':  'PWR  (Pressurized Water Reactor)',
    'phwr': 'PHWR (Pressurized Heavy Water Reactor)',
    'lmfr': 'LMFR  (Liquid Metal Fast Reactor)',
}

ISOTOPES = ['Pu238', 'Pu239', 'Pu240', 'Pu241', 'Pu242']

# Weapons-grade / reactor-grade thresholds (% of total Pu)
WG_THRESHOLD = 93    # Pu-239 > 93 % → weapons grade
RG_THRESHOLD = 60    # Pu-239 < 60 % → reactor grade (MOX-usable)

# LMFR zone configuration
LMFR_ZONES = {
    'lmfr':   'Core',
    'axial':  'Axial Blanket',
    'radial': 'Radial Blanket',
}

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def _detect_enrichment(filename):
    """Extract enrichment from filename.  '0711' → 0.711 %."""
    base = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(r'_(\d+\.?\d*)', base)
    if not match:
        return None
    raw = match.group(1)
    if raw.startswith('0') and len(raw) > 1 and '.' not in raw:
        return float(raw) / 1000.0
    return float(raw)


def _load_single_file(file_path):
    """Load one NFCSS Excel file → cleaned DataFrame with isotopic fractions (%)."""
    df_raw = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=None)
    header_row_idx = df_raw.index[df_raw.iloc[:, 0] == "DataYear"][0]
    df = pd.read_excel(file_path, sheet_name=SHEET_NAME, header=header_row_idx)

    needed = ['Enrichment', 'Burnup', 'TotPu'] + ISOTOPES
    for col in needed:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['Burnup']).sort_values('Burnup').reset_index(drop=True)

    for iso in ISOTOPES:
        df[f'{iso}_frac'] = np.where(
            (df['TotPu'] > 1e-4) & (df[iso] > 1e-4),
            (df[iso] / df['TotPu']) * 100,   # percentage
            0.0,
        )
    return df


def _load_all_files(reactor, src_dir):
    """Discover all {reactor}_*.xlsx files, return combined DataFrame + enrichment list."""
    pattern = os.path.join(src_dir, f'{reactor}_*.xlsx')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}'.")

    frames = []
    enrichments_found = []
    for fpath in files:
        enr = _detect_enrichment(fpath)
        if enr is None:
            print(f"  SKIP: cannot parse enrichment from '{os.path.basename(fpath)}'")
            continue
        try:
            df = _load_single_file(fpath)
            df['Enrichment'] = enr
            frames.append(df)
            enrichments_found.append(enr)
            print(f"  Loaded {os.path.basename(fpath):20s}  →  enrichment = {enr} %")
        except Exception as e:
            print(f"  ERROR loading {os.path.basename(fpath)}: {e}")

    data = pd.concat(frames, ignore_index=True)
    return data, sorted(set(enrichments_found))


def _build_grid(data, column):
    """Pivot scatter data into a regular (Burnup × Enrichment) grid.
    Uses scipy griddata interpolation when the raw pivot has gaps."""
    pivot = data.pivot_table(index='Burnup', columns='Enrichment', values=column)

    burnups     = pivot.index.values.astype(float)
    enrichments = pivot.columns.values.astype(float)
    E, B        = np.meshgrid(enrichments, burnups)
    Z           = pivot.values

    # If there are NaNs, fill with linear interpolation
    if np.isnan(Z).any():
        valid = ~np.isnan(Z)
        points = np.column_stack([E[valid], B[valid]])
        values = Z[valid]
        Z = griddata(points, values, (E, B), method='linear')

    return E, B, Z, enrichments, burnups


# ==========================================
# 3. PLOTTING — 2D CONTOUR MAP
# ==========================================

def plot_contour_map(data, isotope, reactor, out_dir):
    """
    Filled contour map:  Burnup (x) vs Enrichment (y) coloured by isotope fraction.
    Adds WG / RG hatching for Pu-239.
    """
    label  = REACTOR_LABELS.get(reactor.lower(), reactor.upper())
    column = f'{isotope}_frac'
    E, B, Z, enrichments, burnups = _build_grid(data, column)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Main filled contour
    levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 50)
    cf = ax.contourf(B, E, Z, levels=levels, cmap='viridis')
    cbar = fig.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(f'{isotope} (% of Total Pu)', fontsize=11)

    # Pu-239 specific: overlay WG and RG regions
    if isotope == 'Pu239':
        # Weapons-grade hatching  (> 93 %)
        ax.contourf(B, E, Z, levels=[WG_THRESHOLD, 100],
                    colors='none', hatches=['\\\\\\'])
        ax.contour(B, E, Z, levels=[WG_THRESHOLD],
                   colors='red', linewidths=2, linestyles='--')

        # Reactor-grade hatching  (< 60 %)
        ax.contourf(B, E, Z, levels=[0, RG_THRESHOLD],
                    colors='none', hatches=['///'])
        ax.contour(B, E, Z, levels=[RG_THRESHOLD],
                   colors='orange', linewidths=1.5, linestyles='-.')

        # Legend patches
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor='none', edgecolor='red',   hatch='\\\\', label=f'Weapons Grade (>{WG_THRESHOLD} %)'),
            Patch(facecolor='none', edgecolor='orange', hatch='//',   label=f'Reactor Grade (<{RG_THRESHOLD} %)'),
        ]
        ax.legend(handles=legend_handles, loc='upper right', fontsize=9, framealpha=0.9)

    ax.set_xlabel('Burnup (GWd/t)', fontsize=12)
    ax.set_ylabel('Enrichment (wt% $^{235}$U)', fontsize=12)
    ax.set_title(f'{isotope} Fraction — {label}\nContour Map (Enrichment × Burnup)',
                 fontsize=13, fontweight='bold')
    ax.tick_params(labelsize=10, direction='in')
    fig.tight_layout()
    fname = os.path.join(out_dir, f'contour_{reactor.upper()}_{isotope}.png')
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f'  Saved contour_{reactor.upper()}_{isotope}.png')


# ==========================================
# 4. PLOTTING — 3D SURFACE
# ==========================================

def plot_3d_surface(data, isotope, reactor, out_dir, elev=28, azim=225):
    """
    3D surface:  Enrichment (x) × Burnup (y) × Isotope fraction (z).
    For Pu-239, overlays a transparent WG-threshold plane and boundary contour.
    """
    label  = REACTOR_LABELS.get(reactor.lower(), reactor.upper())
    column = f'{isotope}_frac'
    E, B, Z, enrichments, burnups = _build_grid(data, column)

    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # Surface
    surf = ax.plot_surface(
        E, B, Z,
        cmap='viridis', linewidth=0, antialiased=True,
        edgecolor='none', alpha=0.95, shade=True,
    )

    # Pu-239: add weapons-grade threshold plane + boundary
    if isotope == 'Pu239':
        ax.plot_surface(E, B, np.full_like(Z, WG_THRESHOLD),
                        color='red', alpha=0.12)

        # Project WG boundary onto the surface
        try:
            cs = ax.contour(E, B, Z, levels=[WG_THRESHOLD],
                            colors='red', linewidths=2.5, linestyles='--',
                            offset=WG_THRESHOLD)
        except Exception:
            pass  # some mpl versions don't support offset on 3D contour

    ax.set_xlabel('Enrichment (wt% $^{235}$U)', fontsize=11, labelpad=10)
    ax.set_ylabel('Burnup (GWd/t)', fontsize=11, labelpad=10)
    ax.set_zlabel(f'{isotope} (% of Total Pu)', fontsize=11, labelpad=8)
    ax.set_title(f'{isotope} Fraction — {label}\n3D Surface',
                 fontsize=13, fontweight='bold', pad=18)
    ax.set_zlim(0, 100)
    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08)
    cbar.set_label(f'{isotope} (%)', fontsize=10)

    fig.tight_layout()
    fname = os.path.join(out_dir, f'surface3d_{reactor.upper()}_{isotope}.png')
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved surface3d_{reactor.upper()}_{isotope}.png')


# ==========================================
# 5. PLOTTING — MULTI-VIEW SUMMARY
# ==========================================

def plot_summary_grid(data, reactor, out_dir):
    """2×5 grid: one row of contour maps, one row of 3D surfaces for all 5 isotopes."""
    label  = REACTOR_LABELS.get(reactor.lower(), reactor.upper())

    fig = plt.figure(figsize=(26, 10))

    for idx, iso in enumerate(ISOTOPES):
        column = f'{iso}_frac'
        E, B, Z, _, _ = _build_grid(data, column)

        # --- top row: contour ---
        ax_c = fig.add_subplot(2, 5, idx + 1)
        levels = np.linspace(np.nanmin(Z), np.nanmax(Z), 30)
        cf = ax_c.contourf(B, E, Z, levels=levels, cmap='viridis')
        fig.colorbar(cf, ax=ax_c, shrink=0.8)
        ax_c.set_title(iso, fontsize=11, fontweight='bold')
        ax_c.set_xlabel('Burnup', fontsize=9)
        ax_c.set_ylabel('Enrichment', fontsize=9)
        ax_c.tick_params(labelsize=8)

        # --- bottom row: 3D ---
        ax_s = fig.add_subplot(2, 5, idx + 6, projection='3d')
        ax_s.plot_surface(E, B, Z, cmap='viridis', linewidth=0,
                          antialiased=True, alpha=0.92, shade=True)
        ax_s.set_xlabel('Enr.', fontsize=8, labelpad=4)
        ax_s.set_ylabel('BU', fontsize=8, labelpad=4)
        ax_s.set_zlabel('%', fontsize=8, labelpad=2)
        ax_s.set_title(iso, fontsize=10)
        ax_s.set_zlim(0, 100)
        ax_s.view_init(elev=25, azim=225)
        ax_s.tick_params(labelsize=7)

    fig.suptitle(f'Pu Isotopic Fractions — {label}',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fname = os.path.join(out_dir, f'summary_{reactor.upper()}_all_isotopes.png')
    fig.savefig(fname, dpi=250, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved summary_{reactor.upper()}_all_isotopes.png')


# ==========================================
# 6. MAIN DRIVER
# ==========================================

def _process_zone(prefix, label_tag, src_dir, out_root):
    """Load files for one zone prefix and generate all its plots.
    Returns (data, enrichments) or (None, []) if no files found."""
    try:
        data, enrichments = _load_all_files(prefix, src_dir)
    except (FileNotFoundError, SystemExit):
        return None, []

    print(f"\n  Enrichments : {enrichments}")
    print(f"  Burnup range: {data['Burnup'].min():.2f} – {data['Burnup'].max():.2f} GWd/t")
    print(f"  Total rows  : {len(data)}\n")

    if len(enrichments) < 2:
        print(f"  (Need ≥ 2 enrichment levels for contour/surface plots — skipping {label_tag})")
        return data, enrichments

    contour_dir = os.path.join(out_root, f'contour_maps_{label_tag}')
    surface_dir = os.path.join(out_root, f'3d_surfaces_{label_tag}')
    os.makedirs(contour_dir, exist_ok=True)
    os.makedirs(surface_dir, exist_ok=True)

    print(f'--- 2D Contour Maps ({label_tag}) ---')
    for iso in ISOTOPES:
        plot_contour_map(data, iso, label_tag, contour_dir)

    print(f'\n--- 3D Surface Plots ({label_tag}) ---')
    for iso in ISOTOPES:
        plot_3d_surface(data, iso, label_tag, surface_dir)

    print(f'\n--- Summary Grid ({label_tag}) ---')
    plot_summary_grid(data, label_tag, out_root)

    return data, enrichments


def main(reactor_type=None):
    reactor = (reactor_type or REACTOR_TYPE).lower()
    src_dir = os.path.dirname(os.path.abspath(__file__)) or '.'

    print(f"\n{'=' * 60}")
    print(f"  3D Projections — {REACTOR_LABELS.get(reactor, reactor.upper())}")
    print(f"{'=' * 60}")

    out_root = os.path.join(src_dir, f'results_{reactor.upper()}')
    os.makedirs(out_root, exist_ok=True)

    # --- Process main reactor files ---
    data, enrichments = _process_zone(reactor, reactor.upper(), src_dir, out_root)

    # --- LMFR-specific: process blanket zones ---
    if reactor == 'lmfr':
        for zone_prefix in [z for z in LMFR_ZONES if z != 'lmfr']:
            zone_pattern = os.path.join(src_dir, f'{zone_prefix}_*.xlsx')
            zone_files = glob.glob(zone_pattern)
            if not zone_files:
                continue

            zone_name = LMFR_ZONES[zone_prefix]
            label_tag = f'LMFR_{zone_prefix.upper()}'
            print(f"\n{'=' * 60}")
            print(f"  Processing zone: {zone_name}  ({len(zone_files)} files)")
            print(f"{'=' * 60}")
            _process_zone(zone_prefix, label_tag, src_dir, out_root)

    print(f"\n✓ All 3D projection plots saved to:  {out_root}\n")


if __name__ == '__main__':
    rtype = sys.argv[1] if len(sys.argv) > 1 else None
    main(reactor_type=rtype)