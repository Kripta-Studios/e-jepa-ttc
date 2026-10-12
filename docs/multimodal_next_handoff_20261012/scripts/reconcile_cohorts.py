"""Join the 1024-query campaign with the historical/expanded CSVs by verified sensor identity."""
import argparse,csv,json,math
from pathlib import Path
from metrics_core import joined_by_sensor_time,number
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--snapshot-root',type=Path,required=True,help='Extracted original ZIP with V12 and V13')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
def load(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
new=load(a.snapshot_root/'V13/docs/dev32_transfer_20261012/SCORED_PREDICTIONS.csv')
result={}
for name,path in [('original_1024','V12/docs/research_review_20261004_08/evidence/rgb_event/SCORED_PREDICTIONS.csv'),('expanded_3947','V12/docs/agent_handoff_20261010/evidence/comparison/DEV32_PREDICTIONS.csv')]:
    pairs=joined_by_sensor_time(new,load(a.snapshot_root/path))
    fields=['truth_ttc_seconds','H8_seed7','H8_seed13','H8_seed23','public_Garl_event_lhr','public_Garl_rgb_event_full']
    maximum={}
    for field in fields:
        diffs=[]
        for x,y in pairs:
            u,v=number(x[field]),number(y[field])
            if math.isfinite(u)!=math.isfinite(v):raise ValueError('Missingness differs')
            if math.isfinite(u):diffs.append(abs(u-v))
        maximum[field]=max(diffs,default=None)
    result[name]={'timestamp_matches':len(pairs),'identical_query_id':sum(x['query_id']==y['query_id'] for x,y in pairs),'max_numeric_delta':maximum}
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
