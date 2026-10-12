"""Recompute supplied V13 public CSV metrics without models, sensors, or training."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
from metrics_core import number,phase,summarize

MODELS=['E_H1_MATCHED','E_CTX_MATCHED','R_H1','R_CTX','F_TRUE','F_ZERO',
        'H8_seed7','H8_seed13','H8_seed23','public_Garl_event_lhr','public_Garl_rgb_event_full']

def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parents[1]/'evidence')
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use an empty output directory; source evidence is read-only')
    args.output.mkdir(parents=True,exist_ok=True)
    tables=[];details={}
    for dataset in ['FCWD','DEV32']:
        src=args.evidence/(dataset+'_SOURCE.csv')
        rows=list(csv.DictReader(src.open(encoding='utf-8',newline='')))
        if len({r['query_id'] for r in rows})!=len(rows):raise ValueError('Duplicate query IDs')
        labelled=[r for r in rows if math.isfinite(number(r['truth_ttc_seconds']))]
        rgb=lambda r:r['rgb_available'].lower() in {'true','1'}
        parts={'all':labelled,'RGB': [r for r in labelled if rgb(r)],
               'fallback':[r for r in labelled if not rgb(r)]}
        contribution=[]
        for scope,rr in parts.items():
            for band,fn in [('all',lambda t:True),('c',lambda t:0<t<=3),
                            ('s',lambda t:3<t<=6),('l',lambda t:6<t<=10),
                            ('gt10',lambda t:t>10)]:
                chosen=[r for r in rr if fn(number(r['truth_ttc_seconds']))]
                for model in MODELS:
                    values=summarize((number(r['truth_ttc_seconds']),number(r[model])) for r in chosen)
                    tables.append({'dataset':dataset,'scope':scope,'band':band,'model':model,**values})
                if scope!='all' and band!='all':
                    delta=sum(1e4*(abs(phase(number(r['F_TRUE']))-phase(number(r['truth_ttc_seconds'])))
                              -abs(phase(number(r['public_Garl_event_lhr']))-phase(number(r['truth_ttc_seconds']))))
                              for r in chosen)
                    contribution.append({'scope':scope,'band':band,'n':len(chosen),
                                         'contribution_to_micro_gap':delta/len(labelled)})
        unavailable=parts['fallback']
        fallback_ok=all(number(r['F_TRUE'])==number(r['E_CTX_MATCHED']) and
                        number(r['F_ZERO'])==number(r['E_CTX_MATCHED']) for r in unavailable)
        if not fallback_ok:raise ValueError('Fallback differs from frozen E_CTX')
        total=sum(abs(number(r['F_TRUE'])-number(r['truth_ttc_seconds'])) for r in labelled)
        missing=sum(abs(number(r['F_TRUE'])-number(r['truth_ttc_seconds'])) for r in unavailable)
        details[dataset]={'csv_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
                          'queries':len(rows),'labelled':len(labelled),'unlabelled':len(rows)-len(labelled),
                          'rgb_labelled':len(parts['RGB']),'fallback_labelled':len(unavailable),
                          'fallback_equal':fallback_ok,'fallback_AE_share':missing/total,
                          'band_and_RGB_gap':contribution,'new_inferences':0,'optimizer_updates':0}
    with (args.output/'METRICS.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(tables[0]));w.writeheader();w.writerows(tables)
    (args.output/'ANALYSIS.json').write_text(json.dumps(details,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(details,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
