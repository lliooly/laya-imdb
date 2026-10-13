"""Gradient-only window sampling; retain upstream order, losses and RNG streams."""
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json

from common import write_json
from laya_decision_views import BUDGET, full_tokens, sampled_rule, validate_window_training, window_tokens
from laya_resampling import TRAIN_FUNCTION_SHA256


@contextmanager
def window_training(upstream, options, gradient_ids, report_dir):
    options = validate_window_training(options)
    if options is None:
        yield
        return
    original_items, original_train = upstream.items_from_rows, upstream.train_model
    source = inspect.getsource(original_train)
    if hashlib.sha256(source.encode()).hexdigest() != TRAIN_FUNCTION_SHA256:
        raise ValueError('Window adaptation requires the pinned upstream train_model')
    original_encode = upstream.encode_item
    signature = inspect.signature(original_train)

    @wraps(original_items)
    def retain_full(tok, rows, max_len, head_max_len, label_smoothing=0.0):
        items, skipped = original_items(tok, rows, max_len, head_max_len, label_smoothing)
        if skipped or len(items) != len(rows) or (max_len, head_max_len) != (512, 192):
            raise ValueError('Window training requires one fixed binary item per source row')
        tokens = full_tokens(tok, [row['state'] for row in rows])
        for item, ids in zip(items, tokens):
            if item['k'] != 2 or item['qtype'] != 0 or item['state_ids'] != ids[:max_len]:
                raise ValueError('Full tokenization differs from upstream fixed binary items')
            item['state_ids'] = ids
        return items, skipped

    @wraps(original_train)
    def train(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        items, config, tok = (bound.arguments[k] for k in ('items', 'config', 'tok'))
        if len(items) != len(gradient_ids) or len(set(gradient_ids)) != len(gradient_ids):
            raise ValueError('Gradient IDs do not match the window training partition')
        if bound.arguments.get('parallel', False):
            raise ValueError('Window experiment requires the current sequential layout')
        from laya.common import build_head
        for item in items:
            head, _, _ = build_head(tok, item['q'], 192)
            if len(head) != 45:
                raise ValueError('Window training template must leave 466 review tokens')
        identity = {id(item): sid for item, sid in zip(items, gradient_ids)}
        if len(identity) != len(items):
            raise ValueError('Gradient items must be distinct objects')
        callback = bound.arguments.get('on_epoch_end')
        epoch, consumed = 0, set()
        audit = {'state': 'running', 'test_accessed': False, 'mode': options['mode'],
                 'gradient_count': len(items), 'review_token_budget': BUDGET,
                 'seed': config.seed, 'epochs': [], 'rng_scope': 'SHA256; no global RNG draws',
                 'upstream_function_sha256': TRAIN_FUNCTION_SHA256}
        counts = {'head': 0, 'head_tail': 0, 'untruncated': 0}
        audit_path = report_dir / 'window_audit.json'
        write_json(audit_path, audit)

        def encode(item_tok, item, *values, **encode_kwargs):
            sid = identity.get(id(item))
            if sid is None or sid in consumed:
                raise ValueError('Window training saw an unknown or repeated gradient item')
            rule = sampled_rule(config.seed, epoch, sid) if options['mode'] == 'random_head_tail' else 'head'
            changed = len(item['state_ids']) > BUDGET
            # Short items use exactly the original object and encoded input.
            selected = dict(item, state_ids=window_tokens(item['state_ids'], rule)) if changed else item
            result = original_encode(item_tok, selected, *values, **encode_kwargs)
            consumed.add(sid)
            counts[rule] += 1
            counts['untruncated'] += int(not changed)
            journal.write(json.dumps({'epoch': epoch+1, 'sample_id': sid, 'rule': rule,
                                      'truncated': changed, 'input_sha256': hashlib.sha256(json.dumps(
                                          {k: result[k] for k in ('ids', 'markers', 'qtype')},
                                          sort_keys=True, separators=(',', ':')).encode()).hexdigest()}, sort_keys=True)+'\n')
            return result

        def finish_epoch(index, loss):
            nonlocal epoch, consumed, counts
            if index != epoch or consumed != set(gradient_ids):
                raise ValueError('Window epoch did not consume the exact gradient partition')
            audit['epochs'].append({'epoch': epoch+1, 'consumed': len(consumed), **counts})
            write_json(audit_path, audit)
            # Restore for the checkpoint callback; no window changes in evaluation.
            upstream.encode_item = original_encode
            try:
                if callback:
                    callback(index, loss)
            finally:
                upstream.encode_item = encode
            epoch += 1
            consumed, counts = set(), {'head': 0, 'head_tail': 0, 'untruncated': 0}

        bound.arguments['on_epoch_end'] = finish_epoch
        upstream.encode_item = encode
        try:
            with (report_dir / 'window_consumption.jsonl').open('x', encoding='utf-8') as journal:
                result = original_train(*bound.args, **bound.kwargs)
            if epoch != config.epochs or consumed:
                raise ValueError('Incomplete window training epoch accounting')
            audit['state'] = 'completed'
            return result
        except BaseException as exc:
            audit.update(state='failed', error=repr(exc))
            raise
        finally:
            upstream.encode_item = original_encode
            write_json(audit_path, audit)

    upstream.items_from_rows, upstream.train_model = retain_full, train
    try:
        yield
    finally:
        upstream.items_from_rows, upstream.train_model = original_items, original_train
        upstream.encode_item = original_encode
