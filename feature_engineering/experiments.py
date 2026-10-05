"""Feature-group ablation definitions, independent of file loading."""

def experiment_columns(groups, sources=('ins', 'buro', 'bb')):
    """Map E0--E6 and E_bins to ordered feature lists for a shared cohort.

    E6 adds only dynamics to E5. Disjoint aggregates are tested in E_bins.
    IDs and TARGET are metadata and do not appear in any predictor list.
    """
    base=groups['application']
    def cols(labels):
        """Collect historical predictor columns for the requested window labels."""
        return [c for s in sources for label in labels for c in groups[f'{s}_{label}']]
    experiments={'E0':base,'E1':base+cols(['all']), 'E2':base+cols(['w3']),
      'E3':base+cols(['w6']), 'E4':base+cols(['w12']), 'E5':base+cols(['w3','w6','w12'])}
    # E6 adds dynamics to E5; standalone disjoint profiles remain in E_bins.
    experiments['E6']=experiments['E5']+[c for s in sources for c in groups[f'{s}_dynamics']]
    experiments['E_bins']=base+cols(['b0_3','b3_6','b6_12'])
    # Preserve E0--E6; new comparisons explicitly identify longer horizons.
    experiments['E7_24m'] = base + cols(['w24'])
    experiments['E8_multiscale24'] = base + cols(['w3','w6','w12','w24'])
    experiments['E9_dynamics24'] = experiments['E8_multiscale24'] + [c for s in sources for c in groups[f'{s}_dynamics24']]
    experiments['E_bins24'] = base + cols(['b0_3','b3_6','b6_12','b12_24'])
    experiments['E10_36m'] = base + cols(['w36'])
    experiments['E11_multiscale36'] = base + cols(['w3','w6','w12','w24','w36'])
    experiments['E12_dynamics36'] = experiments['E11_multiscale36'] + [c for s in sources for c in groups[f'{s}_dynamics36']]
    experiments['E_bins36'] = base + cols(['b0_3','b3_6','b6_12','b12_24','b24_36'])
    return experiments

