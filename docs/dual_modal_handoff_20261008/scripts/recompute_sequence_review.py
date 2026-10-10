#!/usr/bin/env python3
"""Recompute this review's MAE arithmetic from its explicitly transcribed source fields."""
from pathlib import Path
import argparse, csv, json
import numpy as np

def run(path):
    rows=list(csv.DictReader(path.read_text(encoding='utf-8').splitlines()))
    count_col='n' if 'n' in rows[0] else 'queries'
    n=np.array([int(r[count_col]) for r in rows]);models=['H8_seed7','H8_seed13','H8_seed23','Garl_E','Garl_ER']
    out={}
    for model in models:
        x=np.array([float(r[model]) for r in rows]);out[model]={'pooled_mae':float(np.dot(n,x)/n.sum())}
        if model.startswith('H8'):
            ref=np.array([float(r['Garl_ER']) for r in rows]);out[model]['sequence_wins_vs_Garl_ER']=int((x<ref).sum())
    return {'scope':'MAE arithmetic from transcribed published per-sequence fields, no model inference','sequences':len(rows),'n':int(n.sum()),'models':out}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,default=Path(__file__).resolve().parents[1]/'evidence/mae_by_sequence_transcribed.csv');a=p.parse_args();print(json.dumps(run(a.csv),indent=2))
