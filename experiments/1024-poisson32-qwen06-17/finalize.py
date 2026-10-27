"""Validate configured complete runs, compare outputs and plot final results."""
import json
import os
from pathlib import Path
import time
from common import ROOT,write_json
from report import load_rows,main as report


def main():
    matrix=json.loads((ROOT/'matrix.json').read_text())
    configured_rates=sorted({c['rate'] for c in matrix})
    configured_gammas=sorted({c['gamma'] for c in matrix})
    total=0;outputs={}
    for cell in matrix:
        directory=ROOT/cell['id']
        assert json.loads((directory/'status.json').read_text())['status']=='completed',cell['id']
        rows=load_rows(directory)
        assert len(rows)==1319 and {r['global_sample_index'] for r in rows}==set(range(1319))
        assert all(0<r['generated_tokens']<=1024 for r in rows)
        energy=json.loads((directory/'energy.json').read_text())
        assert energy['gpu_uuid']=='GPU-422501a2-5189-7f5b-5ce6-95464276b27f' and energy['energy_j']>0
        outputs[(cell['gamma'],cell['rate'],cell['method'])]={r['global_sample_index']:r['output_token_ids'] for r in rows}
        total+=len(rows)
    assert total==sum(c['expected_requests'] for c in matrix)
    comparisons=[]
    for gamma in configured_gammas:
        for rate in configured_rates:
            base=outputs[(gamma,rate,'standard')]
            for method in ('fastsd','specedge'):
                other=outputs[(gamma,rate,method)]
                different=[i for i in base if base[i]!=other[i]]
                comparisons.append({'gamma':gamma,'lambda':rate,'method':method,'identical':1319-len(different),'different_sample_indices':different})
    write_json(ROOT/'output_parity.json',comparisons)
    report()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'plot-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=json.loads((ROOT/'results.json').read_text())
    fig,axes=plt.subplots(3,len(configured_gammas),figsize=(5*len(configured_gammas),10),squeeze=False)
    colors={'standard':'#31688e','fastsd':'#e76f51','specedge':'#2a9d8f'}
    labels={'standard':'Standard SD','fastsd':'FastSD','specedge':'SpecEdge (single chain)'}
    for column,gamma in enumerate(configured_gammas):
        axes[0,column].set_title(f'gamma = {gamma}')
        for method_index,method in enumerate(('standard','fastsd','specedge')):
            rows=sorted([r for r in results if r['gamma']==gamma and r['method']==method],key=lambda r:r['rate'])
            rates=[r['rate'] for r in rows]
            series=([r['arrival_ttft_ms']['mean']/1000 for r in rows],
                    [r['tpot_ms']['mean'] for r in rows],
                    [r['energy']['joules_per_token'] for r in rows])
            for index,values in enumerate(series):
                if len(configured_rates)==1:
                    axes[index,column].bar(method_index,values[0],color=colors[method],label=labels[method])
                else:
                    axes[index,column].plot(rates,values,'o-',color=colors[method],label=labels[method])
        for index in range(3):
            ax=axes[index,column]
            if len(configured_rates)==1:
                ax.set_xticks([0,1,2],['Standard SD','FastSD','SpecEdge'])
            else:
                ax.set_xscale('log');ax.set_xticks(configured_rates,[f'{r:g}' for r in configured_rates])
            ax.grid(axis='y',alpha=.25)
            if column==0:ax.set_ylabel(('Mean TTFT incl. draft queue (s)','Mean TPOT (ms/token)','GPU board energy (J/output token)')[index])
            if index==2:ax.set_xlabel(f'Method (Poisson rate={configured_rates[0]:g} requests/s)' if len(configured_rates)==1 else 'Global Poisson arrival rate (requests/s)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Full GSM8K test (1319 requests/run); CPU32 Qwen3-0.6B / GPU Qwen3-1.7B; EOS / 1024-token cap')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.pdf');plt.close(fig)
    manifest=json.loads((ROOT/'manifest.json').read_text());manifest.update(status='completed',completed_requests=total,completed_runs=len(matrix),finalized_epoch=time.time())
    write_json(ROOT/'manifest.json',manifest)
    write_json(ROOT/'finalization.json',{'status':'completed','runs':len(matrix),'requests':total,'epoch':time.time(),'parity':'output_parity.json','plots':['comparison.png','comparison.pdf']})


if __name__=='__main__':main()
