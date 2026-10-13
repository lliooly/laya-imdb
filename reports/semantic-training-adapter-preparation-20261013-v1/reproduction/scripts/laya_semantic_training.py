"""Review-gated semantic epoch inputs with audited actual upstream consumption."""
import ast
from collections import Counter
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import random
import textwrap

from common import ROOT, file_digest, write_json
from laya_resampling import TRAIN_FUNCTION_SHA256
from semantic_pair_pilot import digest, load_source, make_schedule, require_human_review, VARIANTS


FILES = ('pack', 'review', 'manifest')


def validate_semantic(value):
    if value is None:
        return None
    required = {'arm'} | {key + suffix for key in FILES for suffix in ('_file', '_sha256')}
    if not isinstance(value, dict) or set(value) - {'smoke_items'} != required or value['arm'] not in ('A', 'B', 'C'):
        raise ValueError('Invalid semantic_pairs fields or arm')
    for key in FILES:
        path = Path(value[key+'_file'])
        sha = value[key+'_sha256']
        if (path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != 'reports'
                or not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('Semantic files require repository reports paths and SHA256')
    smoke = value.get('smoke_items')
    if smoke is not None and (type(smoke) is not int or not 16 <= smoke <= 128 or smoke % 2):
        raise ValueError('Semantic smoke_items must be an even integer within 16..128')
    return dict(value)


def load_semantic(value, root=ROOT):
    options = validate_semantic(value)
    if options is None:
        return None
    paths = {key: root/options[key+'_file'] for key in FILES}
    for key, path in paths.items():
        if file_digest(path) != options[key+'_sha256']:
            raise ValueError('Semantic file hash mismatch: '+key)
    pack = json.loads(paths['pack'].read_text())
    source, gradient, manifest, source_sha = load_source(paths['manifest'])
    if pack['source_sha256'] != source_sha or pack['manifest_sha256'] != options['manifest_sha256']:
        raise ValueError('Semantic source provenance mismatch')
    if pack['question'] != manifest['question']:
        raise ValueError('Semantic question mismatch')
    rows = pack['rows']
    parents = [r['parent_id'] for r in rows]
    if not rows or len(parents) != len(set(parents)) or not set(parents) <= set(gradient):
        raise ValueError('Semantic parents must be unique original gradient sources')
    for row in rows:
        original = source[row['parent_id']]
        if (type(row['label']) is not int or row['label'] != original['label']
                or set(row['variants']) != set(VARIANTS)
                or row['variants']['original']['text'] != original['text']):
            raise ValueError('Semantic original text or label mismatch')
        for value in row['variants'].values():
            if not value['text'].strip() or digest(value['text']) != value['sha256']:
                raise ValueError('Semantic variant text SHA mismatch')
        if len({v['sha256'] for v in row['variants'].values()}) != 3:
            raise ValueError('Semantic variants must be distinct')
    require_human_review(rows, json.loads(paths['review'].read_text()))
    return {'options': options, 'pack': pack, 'source': source, 'gradient_ids': gradient,
            'calibration_ids': manifest['calibration_ids'], 'tokenizer_sha256': pack['tokenizer_sha256']}


def fingerprint(item):
    # Identity ignores smoothing; targets are checked separately at train entry.
    return digest(json.dumps({k:item[k] for k in ('state_ids', 'q')}, sort_keys=True))


def schedule_digest(slots):
    return digest(json.dumps(slots, sort_keys=True, separators=(',', ':')))


def train_semantic(upstream, bundle, report_dir):
    """Return a wrapper; bundle has already passed source and review gates."""
    original = upstream.train_model
    source = textwrap.dedent(inspect.getsource(original))
    if hashlib.sha256(source.encode()).hexdigest() != TRAIN_FUNCTION_SHA256:
        raise ValueError('Unsupported upstream train_model implementation')
    tree = ast.parse(source)
    hits = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'epoch_items']
    if len(hits) != 1 or ast.unparse(hits[0].value) != 'list(items)':
        raise ValueError('Unsupported upstream semantic epoch expression')
    hits[0].value = ast.parse('_semantic_epoch_items(items, epoch)').body[0].value
    ast.fix_missing_locations(tree)
    signature = inspect.signature(original)

    @wraps(original)
    def train(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        ids = bundle['gradient_ids']
        config, items, tok = (bound.arguments[k] for k in ('config', 'items', 'tok'))
        if (config.epochs != 2 or config.seed not in (42, 43) or len(items) != len(ids)
                or config.loss != 'soft-ce' or bound.arguments.get('parallel', False)):
            raise ValueError('Requires original two-epoch seed42/43 soft-ce gradient partition')
        original_count = len(ids)
        pack = bundle['pack']
        if bundle['options'].get('smoke_items'):
            size = bundle['options']['smoke_items']
            parents = {r['parent_id'] for r in pack['rows']}
            selected = set()
            # Deterministic bounded selection, shared by all arms, independent of predictions.
            for label in (0,1):
                group = sorted(i for i in parents if bundle['source'][i]['label']==label)
                selected.update(group[:size//4] if len(group)>size//2 else group)
                pool = [i for i in ids if i not in selected and bundle['source'][i]['label']==label]
                needed = size//2 - sum(bundle['source'][i]['label']==label for i in selected)
                if len(pool)<needed:
                    raise ValueError('Semantic smoke exceeds gradient population')
                selected.update(pool[:needed])
            indices = [i for i,parent in enumerate(ids) if parent in selected]
            ids,items = [ids[i] for i in indices],[items[i] for i in indices]
            bound.arguments['items'] = items
            pack = dict(pack,rows=[r for r in pack['rows'] if r['parent_id'] in selected])
        labels = {i:bundle['source'][i]['label'] for i in ids}
        def encode_rows(rows, smoothing):
            typed = [{'state':r['text'], 'questions':{'sentiment':pack['question']},
                      'expected':{'sentiment':('negative','positive')[r['label']]}} for r in rows]
            prepared, skipped = upstream.items_from_rows(tok, typed, bound.arguments['max_len'],
                                                         bound.arguments['head_max_len'], label_smoothing=smoothing)
            if skipped or len(prepared) != len(rows):
                raise ValueError('Semantic encoding skipped a source or variant')
            return prepared
        original_rows = [bundle['source'][i] for i in ids]
        expected_original = encode_rows(original_rows, 0.0)
        if config.label_smoothing:
            expected_original = [dict(item,target=[(1-config.label_smoothing)*v+config.label_smoothing/item['k']
                                                  for v in item['target']]) for item in expected_original]
        if any(fingerprint(a) != fingerprint(b) or a['target'] != b['target']
               for a,b in zip(items,expected_original)):
            raise ValueError('Original gradient item order, text or targets differ from frozen sources')
        by_id = dict(zip(ids,items))
        arm = bundle['options']['arm']
        variant = {'A':'original','B':'ordinary','C':'semantic'}[arm]
        variants = {r['parent_id']:r['variants'][variant] for r in pack['rows']}
        variant_items = {}
        if arm != 'A':
            rows = [{'text':variants[r['parent_id']]['text'],'label':r['label']} for r in pack['rows']]
            # optimize_training wraps items_from_rows to keep calibration hard.
            # This is called *inside* train_model: new items are gradient-only,
            # so restore the same smoothing explicitly after hard encoding.
            prepared = encode_rows(rows,0.0)
            for row,item in zip(pack['rows'],prepared):
                if config.label_smoothing:
                    item = dict(item,target=[(1-config.label_smoothing)*v+config.label_smoothing/item['k']
                                             for v in item['target']])
                if item['target'] != by_id[row['parent_id']]['target']:
                    raise ValueError('Semantic variant changed label or smoothing')
                variant_items[row['parent_id']] = item
        schedule = make_schedule(ids,labels,list(variants),seed=config.seed)
        audit = {'state':'running','arm':arm,'options':bundle['options'],'test_accessed':False,
                 'original_gradient_count':original_count,'actual_train_count':len(ids),
                 'actual_gradient_ids':ids,'parent_count_in_pack':len(bundle['pack']['rows']),
                 'parent_ids_used':[r['parent_id'] for r in pack['rows']],'calibration_count':len(bundle['calibration_ids']),
                 'epochs_observed':[], 'consumed_counts':[], 'expected_exposures':2*len(ids),
                 'augmented_slots':sum(s['augmented'] for e in schedule for s in e),
                 'expected_schedule_sha256':[schedule_digest(e) for e in schedule],
                 'actual_schedule_sha256':[], 'encoded_inputs_sha256':[],
                 'upstream_function_sha256':TRAIN_FUNCTION_SHA256}
        audit_path = Path(report_dir)/'semantic_audit.json'
        current_epoch = -1
        cursor = 0
        identity = {}
        encoded = hashlib.sha256()
        actual_slots = []
        journal_path = Path(report_dir)/'semantic_consumption.jsonl'
        def finish_epoch():
            if current_epoch < 0:
                return
            if cursor != len(schedule[current_epoch]):
                raise ValueError('Training did not consume every semantic slot')
            audit['consumed_counts'].append(cursor)
            if schedule_digest(actual_slots) != audit['expected_schedule_sha256'][current_epoch]:
                raise ValueError('Consumed semantic schedule hash mismatch')
            audit['actual_schedule_sha256'].append(schedule_digest(actual_slots))
            audit['encoded_inputs_sha256'].append(encoded.hexdigest())
        def epoch_items(current, epoch):
            nonlocal current_epoch,cursor,identity,encoded,actual_slots
            finish_epoch()
            if current is not items or epoch != len(audit['epochs_observed']) or epoch not in (0,1):
                raise ValueError('Unexpected semantic epoch invocation')
            current_epoch,cursor,identity,encoded = epoch,0,{},hashlib.sha256()
            actual_slots = []
            audit['epochs_observed'].append(epoch+1)
            ordered = []
            for slot in schedule[epoch]:
                # Unique object per position makes repeated parents unambiguous.
                item = dict(variant_items[slot['parent_id']] if slot['augmented'] and arm!='A'
                            else by_id[slot['parent_id']])
                identity[id(item)] = slot
                ordered.append(item)
            permutation = list(range(len(items)))
            random.Random(config.seed+epoch).shuffle(permutation)
            unshuffled = [None]*len(items)
            for pos,index in enumerate(permutation):
                unshuffled[index] = ordered[pos]
            write_json(audit_path,audit)
            return unshuffled
        original_encode = original.__globals__['encode_item']
        def encode(item_tok,item,*values,**kwargs):
            nonlocal cursor
            if (current_epoch < 0 or cursor >= len(schedule[current_epoch])
                    or identity.get(id(item)) != schedule[current_epoch][cursor]):
                raise ValueError('Actual semantic source or order differs from plan')
            result = original_encode(item_tok,item,*values,**kwargs)
            input_sha = digest(json.dumps({k:result[k] for k in ('ids','markers','qtype')},sort_keys=True))
            encoded.update((input_sha+'\n').encode())
            slot = identity[id(item)]
            actual_slots.append(dict(slot))
            journal.write(json.dumps({'epoch':current_epoch+1,'position':cursor,**slot,
                'variant':variant if slot['augmented'] else 'original',
                'input_sha256':input_sha},sort_keys=True)+'\n')
            journal.flush()
            cursor += 1
            return result
        namespace = dict(original.__globals__,_semantic_epoch_items=epoch_items,encode_item=encode)
        exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
        write_json(audit_path,audit)
        try:
            with journal_path.open('x',encoding='utf-8') as journal:
                result = namespace[original.__name__](*bound.args,**bound.kwargs)
                finish_epoch()
            if audit['epochs_observed'] != [1,2]:
                raise ValueError('Both semantic epochs were not consumed')
            audit['state'] = 'completed'
            return result
        except BaseException as exc:
            audit.update(state='failed',error=repr(exc))
            raise
        finally:
            audit.update(current_epoch=current_epoch+1,current_epoch_consumed=cursor)
            write_json(audit_path,audit)
    return train


@contextmanager
def semantic_training(upstream,bundle,report_dir):
    if bundle is None:
        yield
        return
    original = upstream.train_model
    upstream.train_model = train_semantic(upstream,bundle,report_dir)
    try:
        yield
    finally:
        upstream.train_model = original
