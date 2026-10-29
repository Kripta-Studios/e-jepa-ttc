#!/usr/bin/env python3
"""Read-only analysis of the pinned handoff; no fitting or checkpoint access."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def phase(t):
    if not math.isfinite(t) or not (t<0 or t>0.1):raise ValueError('Invalid signed phase domain')
    return -math.log1p(-0.1/t)
def write_csv(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    a=argparse.ArgumentParser();a.add_argument('--v12-root',type=Path,required=True);a.add_argument('--output',type=Path,required=True);v=a.parse_args()
    root=v.v12_root.resolve();out=v.output.resolve()
    if out.is_relative_to(root):raise ValueError('Write output outside the immutable snapshot')
    out.mkdir(parents=True,exist_ok=True)
    p=root/'docs/agent_handoff_20261010/evidence/comparison/FCWD_PREDICTIONS.csv'
    if hashlib.sha256(p.read_bytes()).hexdigest()!='3afe9dcba1ad59bc2535d438026ce0c9b97c2a7df882610776a824ca2ea32082':raise ValueError('Unexpected FCWD source')
    raw=read_csv(p);rows=[];excluded=[]
    for r in raw:
        try:y=float(r['truth_ttc_seconds'])
        except ValueError:excluded.append(r);continue
        if not math.isfinite(y) or y==0:excluded.append(r);continue
        h=float(r['H8_median3']);g=float(r['public_Garl_event_lhr'])
        hm=10000*abs(phase(h)-phase(y));gm=10000*abs(phase(g)-phase(y))
        rows.append(dict(query_id=r['query_id'],sequence=r['sequence_id'],time_s=float(r['anchor_relative_seconds']),truth_s=y,h8_s=h,garl_s=g,h8_mid=hm,garl_mid=gm,gap=hm-gm))
    if len(raw)!=630 or len(rows)!=597 or len({r['query_id'] for r in rows})!=597:raise ValueError('Population changed')
    gap=statistics.mean(r['gap'] for r in rows);totalgap=gap*len(rows)
    ranked=sorted(rows,key=lambda r:r['gap'],reverse=True)
    concentration=[{'top_k':k,'net_gap_share_pct':100*sum(r['gap'] for r in ranked[:k])/totalgap,'all_in_crucial':all(r['truth_s']<=3 for r in ranked[:k])} for k in (10,20,50,100)]
    bands=[]
    for band,predicate in [('c',lambda t:0<t<=3),('s',lambda t:3<t<=6),('l',lambda t:6<t<=10),('outside',lambda t:t>10)]:
        rr=[r for r in rows if predicate(r['truth_s'])]
        bands.append(dict(band=band,n=len(rr),h8_mid=statistics.mean(r['h8_mid'] for r in rr),garl_mid=statistics.mean(r['garl_mid'] for r in rr),h8_mae=statistics.mean(abs(r['h8_s']-r['truth_s']) for r in rr),bias=statistics.mean(r['h8_s']-r['truth_s'] for r in rr),gap_contribution=sum(r['gap'] for r in rr)/len(rows)))
    c=bands[0];c_tie=c['h8_mid']-gap*len(rows)/c['n']
    pred_bins=[]
    for lo,hi in ((1,2),(2,3),(3,4),(4,6),(6,10),(10,60)):
        rr=[r for r in rows if lo<r['h8_s']<=hi]
        pred_bins.append(dict(prediction_interval=f'({lo},{hi}]',n=len(rr),truth_min=min(r['truth_s'] for r in rr) if rr else None,truth_median=statistics.median(r['truth_s'] for r in rr) if rr else None,truth_max=max(r['truth_s'] for r in rr) if rr else None))
    seq=[dict(sequence=s,n=sum(r['sequence']==s for r in rows),mid_gap=statistics.mean(r['gap'] for r in rows if r['sequence']==s)) for s in sorted({r['sequence'] for r in rows})]
    result={'scope':'Descriptive arithmetic on exposed FCWD. No model training, threshold tuning or new network inference. Native median3 outputs; CPU replay differs at rounding tolerance.', 'population':len(raw),'eligible':len(rows),'excluded':len(excluded),'independent_sequences':3,'truth_min':min(r['truth_s'] for r in rows),'truth_max':max(r['truth_s'] for r in rows),'targets_at_most_1':sum(0<r['truth_s']<=1 for r in rows),'h8_predictions_at_cap60':sum(abs(r['h8_s'])>=59.999 for r in rows),'h8_mid':statistics.mean(r['h8_mid'] for r in rows),'garl_mid':statistics.mean(r['garl_mid'] for r in rows),'mid_gap':gap,'h8_query_win_fraction':sum(r['gap']<0 for r in rows)/len(rows),'conditional_c_mid_to_tie_with_other_predictions_unchanged':c_tie,'conditional_c_improvement_fraction_to_tie':1-c_tie/c['h8_mid'],'overall_mid':None,'official_test_score':None,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'optimizer_updates':0}
    write_csv(out/'bands.csv',bands);write_csv(out/'gap_concentration.csv',concentration);write_csv(out/'prediction_bins.csv',pred_bins);write_csv(out/'sequences.csv',seq);write_csv(out/'ranked_queries.csv',ranked);write_csv(out/'ineligible.csv',excluded)
    (out/'ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
