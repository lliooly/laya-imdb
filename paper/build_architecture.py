"""Draw the implemented classifier architecture as an editable vector figure."""
from pathlib import Path
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/imdb-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle


def create_architecture(here, palette):
    here=Path(here);root=here.parent
    config_paths=['.cache/laya/english/rl_agent_config.json',
                  '.cache/laya/english/encoder/config.json',
                  '.cache/deberta-v3-large/config.json']
    head,encoder,deberta=[json.loads((root/p).read_text()) for p in config_paths]
    for lock_path in ['configs/laya_model_lock.json','configs/deberta_model_lock.json']:
        lock=json.loads((root/lock_path).read_text())
        for name,expected in lock['sha256'].items():
            if name.endswith('config.json'):
                assert hashlib.sha256((root/lock['local_dir']/name).read_bytes()).hexdigest()==expected
    assert (head['head_layers'],encoder['num_hidden_layers'],deberta['num_hidden_layers'])==(2,28,24)
    assert encoder['hidden_size']==deberta['hidden_size']==1024
    assert encoder['num_attention_heads']==deberta['num_attention_heads']==16

    source_paths=config_paths+['configs/laya_model_lock.json','configs/deberta_model_lock.json',
        'scripts/laya_fusion_inference.py','scripts/laya_decision_views.py',
        'scripts/deberta_inputs.py','scripts/deberta_training.py',
        '.venv/lib/python3.12/site-packages/laya/common.py',
        '.venv/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py',
        '.venv/lib/python3.12/site-packages/transformers/models/modernbert/modeling_modernbert.py']
    provenance={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths}
    facts={'laya_encoder':{'name':head['encoder'],'layers':28,'hidden_size':1024,'attention_heads':16,
                          'attention':'local/global RoPE','checkpoints':['baseline-s43','combined-s42'],
                          'weights_shared_across_checkpoints':False},
           'laya_head':{'layers':2,'norm_first':True,'attention_heads':16,'ffn_width':4096,
                        'ffn_activation':'ReLU','type_embedding':'added to every encoder token',
                        'candidate_readout':'gather two MASK positions; shared LayerNorm/Linear/GELU/Linear scorer',
                        'action_head':'exists upstream; unused by this sentiment probability fusion'},
           'deberta':{'layers':24,'hidden_size':1024,'attention_heads':16,
                      'readout':'first-token ContextPooler, dropout, linear 1024 to 2'},
           'fusion':{'laya_internal_lambda':0.7020326697613102,'outer_weights':[.5,.5],
                     'view_averaging':'only reviews over 466 Laya tokens; otherwise reuse prefix',
                     'class_threshold':.5},
           'source_sha256':provenance,
           'drawing':'bottom-to-top neural blocks, repeated layer stacks and residual bypasses; original drawing of implemented architecture',
           'style_reference':'https://arxiv.org/abs/1706.03762; visual grammar only, not a reproduced encoder-decoder architecture'}

    ink='#263B46';muted='#5E6E78';rose=palette['light_rose'];blue=palette['light_blue']
    teal='#C9E8ED';lilac=palette['lilac'];light='#F4F6F8'
    with plt.rc_context({'font.family':'DejaVu Sans','font.size':10.2,'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig,ax=plt.subplots(figsize=(9.3,9.6))
        fig.subplots_adjust(left=.012,right=.988,bottom=.018,top=.988)
        ax.set(xlim=(0,14.4),ylim=(0,15.2));ax.axis('off')
        text_boxes=[]

        def box(x,y,w,h,text,fill=light,fs=10.2,weight='normal',stack=False):
            if stack:
                for off in [.14,.07]:
                    ax.add_patch(FancyBboxPatch((x-w/2+off,y-h/2+off),w,h,
                        boxstyle='round,pad=0.025,rounding_size=.09',fc='white',ec='#AAB7BF',lw=.8,zorder=1))
            patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.025,rounding_size=.09',
                                fc=fill,ec=ink,lw=.95,zorder=3)
            ax.add_patch(patch)
            t=ax.text(x,y,text,ha='center',va='center',color=ink,fontsize=fs,
                      fontweight=weight,linespacing=1.22,zorder=4)
            text_boxes.append((patch,t))
            return (x,y-h/2),(x,y+h/2)

        def arrow(a,b,via=None,lw=1.05):
            points=[a]+(via or [])+[b]
            for start,end in zip(points[:-2],points[1:-1]):
                ax.plot([start[0],end[0]],[start[1],end[1]],color=ink,lw=lw,zorder=2)
            ax.annotate('',xy=points[-1],xytext=points[-2],
                        arrowprops={'arrowstyle':'-|>','color':ink,'lw':lw,'mutation_scale':9,
                                    'shrinkA':0,'shrinkB':1.5},zorder=2)

        def chain(nodes):
            for first,second in zip(nodes[:-1],nodes[1:]):arrow(first[1],second[0])

        def add(x,y):
            ax.add_patch(Circle((x,y),.16,fc='white',ec=ink,lw=1,zorder=3))
            ax.text(x,y,'+',ha='center',va='center',fontsize=12,color=ink,zorder=4)

        ax.text(4.35,14.8,'(a) Frozen heterogeneous classifier',ha='center',fontsize=13,fontweight='bold',color=ink)
        ax.text(11.55,14.8,'(b) Laya head detail [A]',ha='center',fontsize=13,fontweight='bold',color=ink)
        ax.plot([8.75,8.75],[1.5,14.4],color='#CBD3D8',lw=.8,ls=(0,(3,4)))

        review=box(4.35,.5,6.9,.55,'Cleaned movie review  x',lilac,11)
        left=[];right=[];xl=2.35;xr=6.75
        left.append(box(xl,1.62,3.5,.6,'ModernBERT tokenizer',rose))
        right.append(box(xr,1.62,3.0,.6,'SentencePiece tokenizer',blue))
        arrow(review[1],left[0][0],via=[(4.35,1.03),(xl,1.03)])
        arrow(review[1],right[0][0],via=[(4.35,1.03),(xr,1.03)])
        left.append(box(xl,2.63,3.5,.72,'Prefix / head-tail views\n466 review tokens each',rose))
        right.append(box(xr,2.63,3.0,.72,'Head-tail view\nOwn token budget',blue))
        left.append(box(xl,3.76,3.5,.76,'Typed sequence packing\nQuestion + candidates + review\nTwo [MASK] markers; total ≤512',rose,9.8))
        right.append(box(xr,3.76,3.0,.76,'Add [CLS] / [SEP]\nTotal length ≤512',blue))
        left.append(box(xl,4.81,3.5,.5,'Token embeddings',rose))
        right.append(box(xr,4.81,3.0,.5,'Token embeddings',blue))
        left.append(box(xl,5.97,3.5,1.3,'ModernBERT-large\n28 encoder layers\nLocal/global attention + RoPE\nd = 1024; 16 heads',rose,10.1,stack=True))
        right.append(box(xr,5.97,3.0,1.3,'DeBERTa-v3-large\n24 encoder layers\nDisentangled attention + FFN\nd = 1024; 16 heads',blue,9.8,stack=True))
        left.append(box(xl,7.42,3.5,.9,'Typed decision head [A]\nType embedding + 2 blocks',rose,10.1))
        right.append(box(xr,7.42,3.0,.9,'First-token pooling\nDense + activation',blue))
        left.append(box(xl,8.58,3.5,.7,'Gather [MASK] positions\nShared scorer → 2 logits',rose))
        right.append(box(xr,8.58,3.0,.7,'Dropout + linear classifier\n1024 → 2 logits',blue,10.0))
        left.append(box(xl,9.6,3.5,.62,r'Softmax $(\mathbf{z}_{m,v}/T_m)$'+'\nEach checkpoint’s calibrated T',rose,9.8))
        right.append(box(xr,9.6,3.0,.62,r'Softmax $(\mathbf{z}_D/T_D)$'+'\n'+r'$T_D\approx1.137$',blue))
        left.append(box(xl,10.62,3.5,.68,'Average views for long reviews\nShort reviews: reuse prefix',rose,9.7))
        right.append(box(xr,10.62,3.0,.68,r'$p_D=P(y=1\mid x)$'+'\nOne DeBERTa checkpoint',blue))
        mixture=box(xl,11.63,4.1,.9,'Two independent checkpoints\nbaseline-s43 / combined-s42\n'+r'$p_L\approx0.7020\,\bar p_{43}+0.2980\,\bar p_{42}$',rose,9.7)
        chain(left+[mixture]);chain(right)
        # The tower denotes one architecture, evaluated with two independent Laya weight sets.
        fusion=box(4.35,12.72,7.75,.75,'Frozen probability fusion (50 : 50)\n'+r'$p_{\mathrm{final}}=\frac{1}{2}p_L+\frac{1}{2}p_D$',teal,11)
        arrow(mixture[1],(xl,fusion[0][1]))
        arrow(right[-1][1],(xr,fusion[0][1]))
        decision=box(4.35,13.87,6.9,.65,r'$p_{\mathrm{final}}\geq0.5$ → positive; otherwise negative',teal,11)
        arrow(fusion[1],decision[0])

        # Inset: faithful pre-norm PyTorch TransformerEncoderLayer, not the post-norm 2017 block.
        xi=11.25
        encoded=box(xi,2.55,3.5,.65,'Encoder hidden states H\nOne checkpoint and one view',light,10.0)
        add(xi,3.5);arrow(encoded[1],(xi,3.34))
        type_box=box(13.6,3.5,1.4,.65,'choice type\nembedding',lilac,8.8)
        arrow((12.9,3.5),(xi+.16,3.5))
        ax.add_patch(FancyBboxPatch((9.0,4.05),5.1,6.45,
                     boxstyle='round,pad=0.04,rounding_size=.14',fc='#FAF8FA',ec='#A7A1AC',lw=.95,zorder=0))
        ax.text(13.75,10.18,'×2',ha='center',fontsize=12,fontweight='bold',color=ink)
        n1=box(xi,4.75,3.35,.6,'LayerNorm',lilac)
        sa=box(xi,5.9,3.35,.85,'Multi-head self-attention\n16 heads + dropout',rose)
        arrow((xi,3.66),n1[0]);arrow(n1[1],sa[0]);add(xi,6.85)
        arrow(sa[1],(xi,6.69))
        arrow((xi,3.9),(xi-.16,6.85),via=[(9.25,3.9),(9.25,6.85)])
        n2=box(xi,7.7,3.35,.6,'LayerNorm',lilac)
        arrow((xi,7.01),n2[0])
        ffn=box(xi,8.9,3.35,1.0,'Feed-forward network\n1024 → 4096 → 1024\nReLU + dropout',rose)
        arrow(n2[1],ffn[0]);add(xi,9.95);arrow(ffn[1],(xi,9.79))
        arrow((xi,7.15),(xi+.16,9.95),via=[(13.45,7.15),(13.45,9.95)])
        gather=box(xi,11.32,3.8,.72,'Gather the two candidate\n[MASK] hidden states',light)
        arrow((xi,10.11),gather[0])
        scorer=box(xi,12.55,4.35,.95,'Shared candidate scorer\nLayerNorm → Linear → GELU → Linear\n1024 → 1024 → 1 per candidate',rose,9.7)
        arrow(gather[1],scorer[0])
        logits=box(xi,13.87,3.8,.65,r'$z_{\mathrm{negative}},\ z_{\mathrm{positive}}$',teal,12)
        arrow(scorer[1],logits[0])
        ax.text(11.5,1.43,'Residual bypasses are shown explicitly.\nPre-norm blocks; no causal attention mask.\nThe upstream action head is unused in this fusion.',
                ha='center',va='center',fontsize=9.1,linespacing=1.4,color=muted)

        # Check every label against its actual rendered box; resize only text, never scientific content.
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        for patch,text in text_boxes:
            bounds=patch.get_window_extent(renderer)
            while text.get_fontsize()>8.4:
                bb=text.get_window_extent(renderer)
                if bb.width<=bounds.width-6 and bb.height<=bounds.height-5:break
                text.set_fontsize(text.get_fontsize()-.2);fig.canvas.draw()
            bb=text.get_window_extent(renderer)
            assert bb.x0>=bounds.x0+1 and bb.x1<=bounds.x1-1 and bb.y0>=bounds.y0+1 and bb.y1<=bounds.y1-1,text.get_text()
        stem=here/'figures/11_architecture'
        for suffix in ['pdf','svg']:fig.savefig(stem.with_suffix('.'+suffix),facecolor='white')
        fig.savefig(stem.with_suffix('.png'),dpi=300,facecolor='white')
        plt.close(fig)
    (here/'analysis/architecture_sources.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
    return {'id':'11_architecture','description':'Implemented heterogeneous classifier and expanded Laya pre-norm decision head; schematic, not a new experiment.',
            'width_inches':9.3,'height_inches':9.6,'palette':palette,'source_files':source_paths,
            'formats':['pdf','png','svg'],'drawing_code':'paper/build_architecture.py'}


if __name__=='__main__':
    here=Path(__file__).resolve().parent
    palette=json.loads((here/'analysis/palette.json').read_text())['colors']
    print(json.dumps(create_architecture(here,palette),ensure_ascii=False))
