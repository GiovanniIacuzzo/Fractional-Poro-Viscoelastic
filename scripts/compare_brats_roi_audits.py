#!/usr/bin/env python3
"""Compare archived coarse BraTS ROI audit to the new graded-mesh audit.

Fewer interface-crossing *cells* is NOT required: refinement can increase
absolute counts while reducing misclassified area. Compare Dice/IoU and
cell-diameter statistics first; inspect anatomical PDFs before acceptance.
"""
import argparse, json
from pathlib import Path


def compare(old, new):
    if old['subject_id'] != new['subject_id'] or old['slice_index'] != new['slice_index']:
        raise ValueError('Cannot compare different subjects or slices.')
    if old['segmentation_sha256'] != new['segmentation_sha256']:
        raise ValueError('The source segmentation changed; use a matching BraTS volume.')
    result = {}
    for label in ('1','2','4','whole_tumor'):
        a,b=old['region_metrics'][label],new['region_metrics'][label]
        result[label]={
            'name':b.get('region_name',label),
            'old_dice':a['dice'],'new_dice':b['dice'],
            'dice_delta':round(b['dice']-a['dice'],6),
            'old_iou':a['iou'],'new_iou':b['iou'],
            'iou_delta':round(b['iou']-a['iou'],6),
            'old_h95_mm':a['hausdorff95_boundary_mm'],
            'new_h95_mm':b['hausdorff95_boundary_mm'],
            'old_mean_cell_diameter_mm':a.get('mean_cell_diameter_mm'),
            'new_mean_cell_diameter_mm':b.get('mean_cell_diameter_mm'),
        }
    return {'subject_id':old['subject_id'],'slice_index':old['slice_index'],
            'old_cells':old['num_quad_cells'],'new_cells':new['num_quad_cells'],
            'old_edema_mixed_fraction':round(old['num_cells_crossing_edema_interface']/old['num_quad_cells'],6),
            'new_edema_mixed_fraction':round(new['num_cells_crossing_edema_interface']/new['num_quad_cells'],6),
            'region_metrics':result,
            'roi_fidelity_improves':all(result[k]['dice_delta']>=-0.005 for k in ('1','2','4')),
            'notes':'Dice alone cannot establish biomechanical mesh convergence.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before',required=True)
    ap.add_argument('--after',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    old=json.loads(Path(args.before).read_text())
    new=json.loads(Path(args.after).read_text())
    out=compare(old,new)
    p=Path(args.out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(out,indent=2,ensure_ascii=False))
    print(json.dumps(out,indent=2,ensure_ascii=False))
    if not new.get('geometry_report_all_mandatory_checks_passed',False):
        raise SystemExit('Geometry report has failed mandatory checks; do not run FEM.')
    if not out['roi_fidelity_improves']:
        raise SystemExit('At least one ROI Dice decreased by more than 0.005; inspect mesh before FEM.')

if __name__=='__main__':
    main()
