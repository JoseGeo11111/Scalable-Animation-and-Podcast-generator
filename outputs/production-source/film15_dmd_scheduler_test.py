"""CPU equivalence against downloaded pinned official FastVideo functions."""
import ast, json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import torch
from film15_dmd_scheduler import FastWanDMDScheduler,SOURCE_COMMIT
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'work/film15-scheduler-research'
ns={'torch':torch,'Any':Any}
def load_function(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
load_function(SRC/'utils.py','pred_noise_to_pred_video')
load_function(SRC/'scheduling_flow_match_euler_discrete.py','add_noise')
cases=[]
for dtype in (torch.float32,torch.bfloat16):
    adapter=FastWanDMDScheduler(); adapter.set_timesteps(3,device='cpu',shift=999)
    table=torch.arange(1000,0,-1,dtype=torch.float32)/1000
    sigmas=8*table/(1+7*table)
    ref=SimpleNamespace(timesteps=sigmas*1000,sigmas=sigmas)
    gen=torch.Generator().manual_seed(123); expected_gen=torch.Generator().manual_seed(123)
    x=torch.arange(96,dtype=torch.float32).reshape(1,2,3,4,4).to(dtype)/96
    for i,t in enumerate((1000,757,522)):
        prediction=torch.ones_like(x)*.125
        btchw=x.permute(0,2,1,3,4); pred=prediction.permute(0,2,1,3,4)
        clean=ns['pred_noise_to_pred_video'](pred.flatten(0,1),btchw.flatten(0,1),torch.tensor([t]),ref).unflatten(0,(1,3))
        if i<2:
            noise=torch.randn(btchw.shape,generator=expected_gen,dtype=dtype)
            expected=ns['add_noise'](ref,clean.flatten(0,1),noise.flatten(0,1),torch.tensor([(757,522)[i]])).unflatten(0,(1,3))
        else: expected=clean
        result=adapter.step(prediction,torch.tensor(t),x,generator=gen)
        torch.testing.assert_close(result.prev_sample,expected.permute(0,2,1,3,4),atol=0,rtol=0)
        torch.testing.assert_close(result.pred_original_sample,clean.permute(0,2,1,3,4),atol=0,rtol=0)
        assert torch.equal(gen.get_state(),expected_gen.get_state())
        x=result[0]
    cases.append({'dtype':str(dtype),'steps':3,'exact_tensor_equality':True,'exact_rng_state':True})
report={'source_commit':SOURCE_COMMIT,'device':'CPU','cases':cases,'sigmas':adapter.sigmas.tolist(),'scope':'Step recurrence and RNG draw ordering only, not conditioning or whole-pipeline equivalence.'}
(SRC/'cpu-equivalence.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
