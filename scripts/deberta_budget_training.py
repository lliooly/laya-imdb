"""Exact positions for budget-limited training; no scientific recipe changes."""
import math


def position(epoch, group, count=22100, batch=32):
    steps=math.ceil(count/batch)
    if type(epoch) is not int or type(group) is not int or not 0<=epoch<=2 or not 0<=group<steps:
        raise ValueError('Invalid next epoch/group')
    if epoch==2 and group:raise ValueError('Completed training has no next group')
    return dict(next_epoch=epoch,next_group=group,updates=epoch*steps+group,
                exposures=epoch*count+min(group*batch,count))


def validate_progress(progress,count=22100,batch=32):
    expected=position(progress['next_epoch'],progress['next_group'],count,batch)
    if any(progress.get(k)!=v for k,v in expected.items()):raise ValueError('Progress counters disagree')
    if len(progress['history'])!=progress['next_epoch']:raise ValueError('Completed epoch history disagrees')
    loss=progress['epoch_loss_sum'];seconds=progress['epoch_seconds']
    if not all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in (loss,seconds)):
        raise ValueError('Invalid accumulated training measurements')
    if not progress['next_group'] and (loss or seconds):raise ValueError('Epoch boundary must reset accumulators')
    return expected


def can_start_update(now,deadline):return now<deadline-180


def training_complete(progress):
    validate_progress(progress)
    return progress['next_epoch']==2
