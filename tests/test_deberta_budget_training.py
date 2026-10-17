import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
import deberta_bf16_training as core
from deberta_budget_training import position,validate_progress,can_start_update,training_complete


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(20,5)
        self.drop=torch.nn.Dropout(.2);self.classifier=torch.nn.Linear(5,2)
    def forward(self,input_ids,attention_mask):
        mask=attention_mask.unsqueeze(-1)
        return SimpleNamespace(logits=self.classifier(self.drop((self.embedding(input_ids)*mask).sum(1)/mask.sum(1))))


def progress(epoch,group,count=22100):
    return dict(**position(epoch,group,count),history=[{'epoch':i+1} for i in range(epoch)],
                epoch_loss_sum=float(group),epoch_seconds=float(group),initial_state_sha256='bound')


class BudgetTrainingTests(unittest.TestCase):
    def test_positions_and_tail_match_every_group(self):
        groups=core.groups_for(22100,0)
        self.assertEqual(len(groups),691);self.assertEqual(len(groups[-1]),20)
        self.assertEqual(sorted(i for group in groups for i in group),list(range(22100)))
        for epoch in range(2):
            seen=epoch*22100
            for group,indices in enumerate(groups):
                p=progress(epoch,group);validate_progress(p);self.assertEqual(p['exposures'],seen)
                self.assertEqual(p['updates'],epoch*691+group);seen+=len(indices)
            self.assertEqual(seen,(epoch+1)*22100)
        self.assertTrue(training_complete(progress(2,0)))

    def test_impossible_or_inconsistent_resume_rejected(self):
        for e,g in [(-1,0),(0,-1),(0,691),(2,1),(3,0),(True,0)]:
            with self.assertRaises(ValueError):position(e,g)
        for field,value in [('updates',999),('exposures',999),('history',[]),('epoch_loss_sum',math.nan)]:
            p=progress(1,256);p[field]=value
            with self.assertRaises(ValueError):validate_progress(p)
        p=progress(1,0);p['epoch_loss_sum']=1
        with self.assertRaises(ValueError):validate_progress(p)

    def test_mid_epoch_restore_exact_next_group_tail_rng_optimizer_schedule(self):
        device=torch.device('cpu');core.seed_all();model=Tiny();opt,sched=core.optimizer_for(model)
        items=[dict(input_ids=[i%17+1,2],label=i%2) for i in range(69)]
        groups=core.groups_for(len(items),0)
        core.update(model,[items[i] for i in groups[0]],0,opt,sched,device)
        p=progress(0,1,len(items));validate_progress(p,len(items))
        state=copy.deepcopy(core.snapshot(model,opt,sched,device,{'id':'test'},p))
        weights=copy.deepcopy(model.state_dict())
        for indices in groups[1:]:core.update(model,[items[i] for i in indices],0,opt,sched,device)
        expected=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        expected_sched=copy.deepcopy(sched.state_dict());expected_rng=torch.get_rng_state().clone()
        other=Tiny();other.load_state_dict(weights);oo,ss=core.optimizer_for(other)
        core.restore(state,other,oo,ss,device,{'id':'test'})
        resumed_groups=core.groups_for(len(items),state['progress']['next_epoch'])[state['progress']['next_group']:]
        self.assertEqual(resumed_groups,groups[1:]);self.assertEqual(len(resumed_groups[-1]),5)
        for indices in resumed_groups:core.update(other,[items[i] for i in indices],0,oo,ss,device)
        for key,value in expected.items():self.assertTrue(torch.equal(value,other.state_dict()[key]))
        for key,values in expected_opt['state'].items():
            for name,v in values.items():self.assertTrue(torch.equal(v,oo.state_dict()['state'][key][name]))
        self.assertEqual(expected_sched,ss.state_dict());self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))

    def test_cutoff_prevents_another_update_without_completing_experiment(self):
        self.assertTrue(can_start_update(819.99,1000));self.assertFalse(can_start_update(820,1000))
        self.assertFalse(training_complete(progress(1,680)))
        self.assertTrue(training_complete(progress(2,0)))

    def run_driver(self,stop_after=None):
        import tempfile
        from contextlib import ExitStack
        from unittest.mock import patch
        from common import write_json,file_digest
        from deberta_budget_study import train,STUDY
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        items=[dict(id=f'x{i}',input_ids=[1,2],label=i%2) for i in range(22100)]
        saved=[];calls=[];model=Tiny()
        def save(m,opt,sched,device,p,rows,pad,folder,stage):
            folder.mkdir(parents=True);write_json(folder/'bundle.json',{'stage':stage})
            saved.append((stage,copy.deepcopy(p)))
            write_json(root/'reports'/STUDY/'acks'/(stage+'.json'),
                       {'bundle_sha256':file_digest(folder/'bundle.json')})
        cp=SimpleNamespace(ROOT=root,training_inputs=lambda:(items,{'pad_token_id':0}),save=save,
                           bundle=lambda epoch:root/'checkpoints'/STUDY/f'epoch-{epoch:03d}')
        def update(m,rows,*args):
            calls.append([x['id'] for x in rows]);return {'loss':.25,'seconds':1.,'samples':len(rows)}
        with ExitStack() as stack:
            stack.enter_context(patch('deberta_budget_study.config',return_value={'model_dir':'model','periodic_groups':[256,512]}))
            stack.enter_context(patch.object(core,'load_initial',return_value=model))
            stack.enter_context(patch.object(core,'setup_model',side_effect=lambda m,d:m))
            stack.enter_context(patch.object(core,'update',side_effect=update))
            stack.enter_context(patch('deberta_budget_study.can_start_update',side_effect=lambda n,d:stop_after is None or len(calls)<stop_after))
            result=train(cp,0,float('inf'))
        return result,saved,calls,root

    def test_driver_budget_stop_records_real_resume_position_and_journal(self):
        from common import read_jsonl
        result,saved,calls,root=self.run_driver(17)
        self.assertFalse(result);self.assertEqual(len(calls),17)
        self.assertEqual(len(saved),1);stage,p=saved[0]
        self.assertTrue(stage.startswith('budget-stop-'));self.assertEqual(p['next_group'],17)
        self.assertEqual(p['updates'],17);self.assertEqual(p['exposures'],544)
        entries=list(read_jsonl(root/p['audit_path']))
        self.assertEqual([entry['ids'] for entry in entries],calls)
        self.assertEqual(p['epoch_loss_sum'],136.)

    def test_driver_full_epoch_periodic_and_boundary_checkpoints_keep_tail(self):
        result,saved,calls,root=self.run_driver()
        self.assertTrue(result);self.assertEqual([p['updates'] for _,p in saved],[256,512,691])
        self.assertEqual(len(calls[-1]),20);p=saved[-1][1]
        validate_progress(p);self.assertEqual(p['next_epoch'],1);self.assertEqual(p['exposures'],22100)
        self.assertEqual(p['history'][0]['mean_loss'],.25);self.assertFalse(training_complete(p))

    def test_private_adapter_retains_bf16_recipe_and_old_study(self):
        import deberta_training as original
        from deberta_budget_study import install,STUDY
        before=core.fp32.update
        cp=install();self.assertEqual(cp.STUDY,STUDY);self.assertIs(core.fp32.update,before)
        self.assertEqual(core.RECIPE['epochs'],2);self.assertEqual(core.RECIPE['total_steps'],1382)


if __name__=='__main__':unittest.main()
