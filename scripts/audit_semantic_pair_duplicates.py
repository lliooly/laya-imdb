"""Exact and lexical five-word-shingle overlap audit; official test is never read."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from common import ROOT, file_digest, read_jsonl, write_json
from semantic_pair_pilot import digest,load_source


def shingles(text):
    words=re.findall(r'[a-z0-9]+',text.lower())
    return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))} or {tuple(words)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'reports/laya-l512-s42-ce/manifest.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('Choose an unused audit output')
    pack=json.loads(args.pack.read_text());source,gradient,manifest,source_sha=load_source(args.manifest)
    if pack['source_sha256']!=source_sha or pack['manifest_sha256']!=file_digest(args.manifest):raise ValueError('Pack provenance changed')
    variants=[]
    for row in pack['rows']:
        if row['parent_id'] not in set(gradient) or row['variants']['original']['text']!=source[row['parent_id']]['text']:raise ValueError('Pack source changed')
        for name in ('ordinary','semantic'):
            value=row['variants'][name]
            if digest(value['text'])!=value['sha256']:raise ValueError('Variant SHA mismatch')
            variants.append({'parent_id':row['parent_id'],'variant':name,'sha256':value['sha256'],'shingles':shingles(value['text'])})
    inverted=defaultdict(set)
    for i,row in enumerate(variants):
        for shingle in row['shingles']:inverted[shingle].add(i)
    split=json.loads((ROOT/'reports/split_manifest.json').read_text());dev_path=ROOT/'data/processed/validation.jsonl'
    if file_digest(dev_path)!=split['jsonl_sha256']['validation']:raise ValueError('Frozen validation changed')
    matches=[];calibration=set(manifest['calibration_ids'])
    rows=((r,'calibration' if i in calibration else 'gradient') for i,r in source.items())
    def scan(rows):
        for row,partition in rows:
            grams=shingles(row['text']);candidates=set()
            for gram in grams:candidates.update(inverted.get(gram,()))
            for i in candidates:
                value=variants[i]
                if row['id']==value['parent_id']:continue
                similarity=len(grams & value['shingles'])/len(grams | value['shingles'])
                if similarity>=.8 or digest(row['text'])==value['sha256']:
                    matches.append({'parent_id':value['parent_id'],'variant':value['variant'],
                        'other_id':row['id'],'other_partition':partition,'jaccard':similarity,
                        'exact':digest(row['text'])==value['sha256']})
    scan(rows);scan((r,'validation') for r in read_jsonl(dev_path))
    for i,a in enumerate(variants):
        for b in variants[i+1:]:
            if a['parent_id']==b['parent_id']:continue
            similarity=len(a['shingles'] & b['shingles'])/len(a['shingles'] | b['shingles'])
            if similarity>=.8:matches.append({'parent_id':a['parent_id'],'variant':a['variant'],
                'other_id':b['parent_id'],'other_partition':'other_variant','other_variant':b['variant'],
                'jaccard':similarity,'exact':a['sha256']==b['sha256']})
    protected=[m for m in matches if m['other_partition'] in ('calibration','validation')]
    write_json(args.out,{'state':'lexical_audit_completed','pack_sha256':file_digest(args.pack),
        'source_manifest_sha256':file_digest(args.manifest),'train_source_sha256':source_sha,
        'validation_source_sha256':split['jsonl_sha256']['validation'],'variant_count':len(variants),
        'method':'lowercase alphanumeric five-word shingle Jaccard >= 0.8; same-parent overlap expected and excluded',
        'threshold':.8,'matches':matches,'protected_matches':protected,'protected_overlap_found':bool(protected),
        'semantic_similarity_proven_absent':False,'test_accessed':False,'script_sha256':file_digest(Path(__file__))})
    print(f'Lexical audit: {len(matches)} matches, {len(protected)} calibration/validation matches.')


if __name__=='__main__':main()
