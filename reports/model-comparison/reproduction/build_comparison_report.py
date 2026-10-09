"""Build a source-backed comparison bundle from exported validation logits."""
import os
os.environ.setdefault("MPLCONFIGDIR", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(__import__("pathlib").Path(__file__).resolve().parents[1] / ".cache"))
import csv
import itertools
import json
import math
from pathlib import Path
import random
import re
import shutil
import zipfile

import numpy as np
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics, fit_temperature, paired, sigmoid

OUT = ROOT / "reports/model-comparison"
RUNS = {"modernbert_base": "modernbert-l256-s42", "laya_ce_full": "laya-l512-s42-ce",
        "laya_rlcd_pilot": "laya-l512-s42-pilot-rlcd", "laya_ce_pilot": "laya-l512-s42-pilot-ce"}
NAMES = {"modernbert_base": "ModernBERT-base full", "laya_ce_full": "Laya-CE full",
         "laya_rlcd_pilot": "Laya-RLCD PILOT", "laya_ce_pilot": "Laya-CE PILOT"}


def csv_write(path, rows, fields=None):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parameters(key, run):
    folder = OUT / key
    export = json.loads((folder / "export.json").read_text())
    report = ROOT / "reports" / run
    result = json.loads((report / "result.json").read_text())
    hardware = json.loads((ROOT / "reports/environment.json").read_text())["hardware"]
    if key == "modernbert_base":
        c = result["config"]
        model_lock = json.loads((ROOT / "configs/model_lock.json").read_text())
        train = {"model": NAMES[key], "pretrained_checkpoint": model_lock["model"], "revision": model_lock["revision"],
                 "initialization": "pretrained encoder + random masked-mean-pool binary linear classifier",
                 "max_sequence_length": c["length"], "epochs": result["epochs_completed"], "best_epoch": result["best_epoch"],
                 "checkpoint_selection": "highest validation Macro-F1; ties use lower NLL; full 2500 development validation",
                 "per_device_batch_size": c["batch_size"], "gradient_accumulation": 1, "effective_batch_size": c["batch_size"],
                 "last_batch_size": 4, "encoder_lr": c["learning_rate"], "head_lr": c["learning_rate"],
                 "optimizer": "MLX AdamW, bias_correction=True", "weight_decay": c["weight_decay"],
                 "scheduler": "linear warmup then linear decay to zero", "warmup_fraction": c["warmup_fraction"],
                 "warmup_updates": math.ceil(8439*c["warmup_fraction"]), "seed": c["seed"], "loss": "CE",
                 "freeze_encoder": False, "precision_training": "float32; MLX_ENABLE_TF32=0", "saved_precision": "float32",
                 "train_size_source": 22500, "gradient_train_size": 22500, "native_calibration_size": 0,
                 "training_time_seconds": result["elapsed_training_seconds"],
                 "time_scope": "sum of timed forward/backward/optimizer update intervals only; excludes validation/save/tokenization",
                 "peak_memory_gib": result["mlx_peak_gib"], "memory_scope": "MLX allocator peak, not full system use",
                 "optimizer_updates": 8439, "framework": "MLX 0.32.3", "head_config": "masked mean pooling + Linear(768,2)"}
        shutil.copy2(ROOT / "configs/day3.json", folder / "config.json")
        shutil.copy2(ROOT / "checkpoints" / run / "config.json", folder / "encoder_config.json")
        write_json(folder / "source_manifest.json", result["identity"])
        shutil.copy2(ROOT / "configs/model_lock.json", folder / "pretrained_model_lock.json")
    else:
        manifest = json.loads((report / "manifest.json").read_text())
        c = manifest["spec"]["training"]
        status = json.loads((report / "status.json").read_text())
        train = {"model": NAMES[key], "pretrained_checkpoint": manifest["model_lock"]["model_id"],
                 "revision": manifest["model_lock"]["revision"], "encoder_checkpoint": "answerdotai/ModernBERT-large",
                 "initialization": "original pretrained Laya encoder + pretrained decision head; not continued from pilot",
                 "max_sequence_length": c["max_len"], "head_max_len": c["head_max_len"], "epochs": c["epochs"],
                 "best_epoch": 1, "checkpoint_selection": "only epoch/final checkpoint; no multi-epoch validation selection",
                 "per_device_batch_size": c["micro_batch"], "gradient_accumulation": c["grad_accum"],
                 "effective_batch_size": c["micro_batch"]*c["grad_accum"],
                 "last_accumulation_window_samples": result["train_items"] % (c["micro_batch"]*c["grad_accum"]) or c["micro_batch"]*c["grad_accum"],
                 "encoder_lr": c["encoder_lr"], "head_lr": c["head_lr"], "optimizer": "PyTorch AdamW; default betas=(0.9,0.999), eps=1e-8",
                 "weight_decay": c["weight_decay"], "scheduler": "CosineAnnealingLR, eta_min=1e-6", "warmup_fraction": 0,
                 "seed": c["seed"], "loss": c["loss"], "freeze_encoder": c["freeze_encoder"],
                 "precision_training": "float32; autocast off", "saved_precision": "fp16; reloaded float32 for analysis",
                 "gradient_checkpointing": c["gradient_checkpointing"],
                 "train_size_source": result["train_items"]+result["calibration_items"], "gradient_train_size": result["train_items"],
                 "native_calibration_size": result["calibration_items"], "training_time_seconds": status["elapsed_seconds"],
                 "time_scope": "finetune pipeline elapsed time: load/preprocess/before-eval/train/calibration/after-eval/save; includes process suspension",
                 "pure_training_seconds": None, "peak_memory_gib": None,
                 "memory_scope": "peak not recorded; status mps_driver_allocated_bytes is END driver memory, not peak",
                 "optimizer_updates": math.ceil(result["train_items"] / c["micro_batch"] / c["grad_accum"]) * c["epochs"],
                 "framework": "PyTorch/MPS; package versions in source_manifest.json",
                 "head_config": {"transformer_layers": export["head_layers"], "hidden_size": export["hidden_size"],
                                  "attention_heads": export["head_attention_heads"], "feedforward_dim": 4*export["hidden_size"],
                                  "dropout": .1, "norm_first": True, "type_embedding": "3 x hidden_size",
                                  "scorer": "LayerNorm -> Linear(d,d) -> GELU -> Linear(d,1), scores MASK option markers",
                                  "action_head": "Linear(d+4,256) -> GELU -> Linear(256,2); NOT used by classification loss"},
                 "input_template": "[CLS] choice question: {instructions} [SEP] [MASK] {negative_option} [MASK] {positive_option} [SEP] {review_prefix} [SEP]",
                 "question_schema": manifest["question"], "head_token_length": export["head_token_length"],
                 "review_token_budget": export["review_token_budget"], "option_order_training": "randomly shuffled, labels permuted together",
                 "option_order_evaluation": "negative, positive (canonical)"}
        shutil.copy2(report / "manifest.json", folder / "source_manifest.json")
        write_json(folder / "config.json", manifest["spec"])
        shutil.copy2(report / "console.log", folder / "console.log")
        shutil.copy2(report / "status.json", folder / "source_status.json")
        shutil.copy2(ROOT / "checkpoints" / run / "encoder/config.json", folder / "encoder_config.json")
        shutil.copy2(ROOT / "checkpoints" / run / "rl_agent_config.json", folder / "decision_model_config.json")
        shutil.copy2(ROOT / "checkpoints" / run / "questions.json", folder / "questions.json")
    train.update({k:v for k,v in export.items() if "parameters" in k})
    train.update(hardware=hardware, gpu="Apple M5 Pro integrated GPU", memory="64 GB unified memory; no dedicated VRAM",
                 dataset="Stanford IMDB", original_split_seed=42, native_validation_size=result.get("eval_items",2500),
                 common_inference_validation_size=2500, common_calibration_size=400, common_evaluation_size=2100,
                 test_size=25000, test_accessed=False, truncation_strategy="retain head/prefix of review; Laya question/options consume total budget",
                 checkpoint_sha256=export["checkpoint_sha256"], parameters_scope="parameters exclude buffers; trainable denotes requires_grad/optimizer eligibility")
    checkpoint_file=(ROOT/"checkpoints"/run/("best-epoch3.safetensors" if key=="modernbert_base" else "model.safetensors"))
    train["checkpoint_file_bytes"] = checkpoint_file.stat().st_size
    train["fp32_parameter_bytes_theoretical"] = train["total_parameters"] * 4
    if key != "modernbert_base":
        train["classification_loss_connected_parameters"] = train["total_parameters"] - train["action_head_parameters_not_used_by_classification_loss"]
    write_json(folder / "manifest.json", train)
    shutil.copy2(report / "result.json", folder / "source_result.json")
    return train, result


def logs(key, run, info, result, full_metrics):
    folder = OUT / key
    epochs = []
    if key == "modernbert_base":
        for h in result["history"]:
            epochs.append({"epoch": h["epoch"], "train_loss": h["online_train_loss"], "train_loss_type": "online CE",
                           "train_monitor_loss": h["train_monitor_fixed_subset"]["loss"],
                           "train_monitor_accuracy": h["train_monitor_fixed_subset"]["accuracy"],
                           "val_loss": h["validation"]["loss"], "val_loss_type": "uncalibrated CE/NLL",
                           "val_accuracy": h["validation"]["accuracy"], "val_macro_f1": h["validation"]["macro_f1"],
                           "validation_count":2500,"optimizer_updates_cumulative":h["global_step"]})
        steps = []
        note = "Baseline step console log was not persisted; no observed step-loss series is available. Do not fabricate it."
    else:
        epochs = [{"epoch": 1, "train_loss": result["epoch_loss"][0], "train_loss_type": info["loss"],
                   "train_monitor_loss":None, "train_monitor_accuracy":None,
                   "val_loss":full_metrics["uncalibrated"]["nll"], "val_loss_type":"uncalibrated saved-checkpoint NLL; full 2500 re-evaluation",
                   "val_accuracy":full_metrics["uncalibrated"]["accuracy"], "val_macro_f1":full_metrics["uncalibrated"]["macro_f1"],
                   "validation_count":2500, "optimizer_updates_cumulative":info["optimizer_updates"]}]
        steps=[]
        text=(folder/"console.log").read_text()
        for epoch,step,loss in re.findall(r"epoch (\d+)/\d+ step (\d+) loss ([-\d.eE+]+)",text):
            update=(int(step)-1)//info["gradient_accumulation"]
            factor=(1+math.cos(math.pi*update/info["optimizer_updates"]))/2
            steps.append({"epoch":int(epoch),"micro_step":int(step),"loss_logged_4dp":float(loss),
                          "encoder_lr_reconstructed_used":1e-6+(info["encoder_lr"]-1e-6)*factor,
                          "head_lr_reconstructed_used":1e-6+(info["head_lr"]-1e-6)*factor})
        note="Step losses are actual sparse console samples (every 16 micro-steps), rounded to 4 decimals; learning rates reconstructed from the pinned cosine schedule, not recorded measurements."
    csv_write(folder/"epoch_log.csv",epochs)
    csv_write(folder/"train_log.csv",steps)
    if not steps:
        (folder/"train_log.csv").write_text("epoch,step,loss,learning_rate\n")
    write_json(folder/"trainer_state.json",{"epochs":epochs,"step_log_count":len(steps),"note":note,
               "multi_epoch_overfitting_analysis_available":key=="modernbert_base"})
    return epochs


def main():
    rows=list(read_jsonl(ROOT/"data/processed/validation.jsonl")); ids=[r["id"] for r in rows]
    labels=np.asarray([r["label"] for r in rows]); assert len(ids)==len(set(ids))==2500
    seed=20261009; rng=random.Random(seed); calibration_ids=[]
    for label in (0,1):
        pool=[r["id"] for r in rows if r["label"]==label]; rng.shuffle(pool); calibration_ids+=pool[:200]
    calibration_set=set(calibration_ids)
    cal=np.asarray([i in calibration_set for i in ids]); ev=~cal
    assert cal.sum()==400 and ev.sum()==2100 and not np.any(cal&ev)
    frozen=json.loads((OUT/"common_split.json").read_text())
    assert frozen["calibration_ids"]==calibration_ids
    assert frozen["evaluation_ids"]==[i for i in ids if i not in calibration_set]
    write_json(OUT/"analysis_protocol.json",{"dataset":"IMDB frozen development validation", "seed":seed,
               "common_calibration_ids":calibration_ids,"common_evaluation_ids":[i for i in ids if i not in calibration_set],
               "temperature_bounds":[.05,20],"temperature_fit":"minimize stable NLL on calibration-only rows, bisection in inverse temperature",
               "brier_definition":"sum over two classes (p_k-y_k)^2; also export positive-only value exactly half",
               "ece_definition":"15 equal-width bins of top-label confidence; (lo,hi], first bin includes 0",
               "precision_recall":"macro in main table; positive-class also exported", "auroc":"rank raw logit margins, average ranks for ties",
               "evaluation_warning":"development analysis, not fresh test: baseline epoch selection previously used all 2500 rows; pilot development also used a 256-row subset",
               "test_accessed":False,"model_selection":"historical selected checkpoints fixed before this analysis"})
    from tokenizers import Tokenizer
    lock=json.loads((ROOT/"configs/model_lock.json").read_text())
    tokenizer_path=ROOT/".cache/huggingface/hub"/("models--"+lock["model"].replace("/","--"))/"snapshots"/lock["revision"]/"tokenizer.json"
    tok=Tokenizer.from_file(str(tokenizer_path)); tok.no_truncation(); tok.no_padding()
    lengths=np.asarray([len(x.ids) for x in tok.encode_batch([r["text"] for r in rows])])
    all_scores={}; all_metrics={}; infos={}; epoch_logs={}
    for key,run in RUNS.items():
        raw=list(read_jsonl(OUT/key/"raw.jsonl"))
        assert [r["sample_id"] for r in raw]==ids and [r["true_label"] for r in raw]==labels.tolist()
        z=np.asarray([r["logits"] for r in raw],dtype=float); margin=z[:,1]-z[:,0]
        all_scores[key]=margin
        temperature=fit_temperature(margin[cal],labels[cal])
        full={"uncalibrated":metrics(labels,margin)}
        if key!="modernbert_base":
            native=json.loads((OUT/key/"export.json").read_text())["native_temperature"]
            full["native_calibrated"]=metrics(labels,margin,native)
        common={"uncalibrated":metrics(labels[ev],margin[ev]),"calibrated":metrics(labels[ev],margin[ev],temperature),
                "temperature_fit_count":400,"temperature_at_boundary":min(abs(temperature-.05),abs(temperature-20))<1e-6}
        info,result=parameters(key,run); infos[key]=info
        predrows=[]
        for i,row in enumerate(rows):
            predrows.append({"sample_id":ids[i],"true_label":int(labels[i]),"prob_positive":float(sigmoid(margin[i])),
                             "pred_label":int(margin[i]>0),"prob_positive_calibrated":float(sigmoid(margin[i]/temperature)),
                             "logit_negative":float(z[i,0]),"logit_positive":float(z[i,1]),
                             "analysis_role":"calibration" if cal[i] else "evaluation", "review_tokens_with_specials":int(lengths[i])})
        csv_write(OUT/key/"predictions_all2500.csv",predrows)
        csv_write(OUT/key/"predictions.csv",[r for r in predrows if r["analysis_role"]=="evaluation"])
        csv_write(OUT/key/"calibration_predictions.csv",[r for r in predrows if r["analysis_role"]=="calibration"])
        length_metrics={}
        for name,keep in {"base_untruncated_le256":lengths<=256,"base_truncated_gt256":lengths>256,
                          "long_gt512":lengths>512}.items():
            k=keep&ev
            length_metrics[name]=metrics(labels[k],margin[k],temperature)
        combined={"common_evaluation_2100":common,"full_validation_2500":full,"length_analysis_common2100":length_metrics,
                  "scope":"development validation only; common400 calibration is disjoint from common2100 evaluation"}
        write_json(OUT/key/"metrics.json",combined)
        all_metrics[key]=combined
        epoch_logs[key]=logs(key,run,info,result,full)
    pairs=[]
    for a,b in itertools.combinations(RUNS,2):
        pairs.append(dict(model_a=a,model_b=b,**paired(labels[ev],all_scores[a][ev],all_scores[b][ev])))
    previous=0.
    for rank,index in enumerate(sorted(range(len(pairs)),key=lambda i:pairs[i]["mcnemar_exact_two_sided_p"])):
        previous=max(previous,min(1.,(len(pairs)-rank)*pairs[index]["mcnemar_exact_two_sided_p"]))
        pairs[index]["holm_adjusted_p_over_six_exploratory_pairs"]=previous
    write_json(OUT/"paired_tests.json",pairs)
    csv_write(OUT/"paired_tests.csv",pairs)
    # Same train/calibration/validation sources for the pilot loss-function contrast.
    ca=json.loads((OUT/"laya_ce_pilot/source_manifest.json").read_text())
    ra=json.loads((OUT/"laya_rlcd_pilot/source_manifest.json").read_text())
    for k in ("train","validation","gradient_train_ids","calibration_ids","model_lock","versions"):
        assert ca[k]==ra[k],k
    c,r=dict(ca["spec"]["training"]),dict(ra["spec"]["training"]); c.pop("loss"); r.pop("loss"); assert c==r
    pilot_ids=set(ca["validation"]["ids"]); pk=np.asarray([i in pilot_ids for i in ids])
    pilot={key:{"uncalibrated":metrics(labels[pk],all_scores[key][pk]),
                "native_calibrated":metrics(labels[pk],all_scores[key][pk],json.loads((OUT/key/"export.json").read_text())["native_temperature"])}
           for key in ("laya_ce_pilot","laya_rlcd_pilot")}
    write_json(OUT/"matched_pilot_256.json",{"same_training_budget_verified":True,"models":pilot,
               "paired":paired(labels[pk],all_scores["laya_ce_pilot"][pk],all_scores["laya_rlcd_pilot"][pk])})
    write_json(OUT/"all_metrics.json",all_metrics)
    table_fields=("accuracy","macro_f1","precision_macro","recall_macro","precision_positive","recall_positive",
                  "auroc","nll","brier_multiclass","brier_positive","ece_15_equal_width_top_label")
    full_table=[]; common_table=[]
    for key in RUNS:
        full=all_metrics[key]["full_validation_2500"]["uncalibrated"]
        full_table.append(dict(model=key,eval_count=2500,**{f:full[f] for f in table_fields}))
        common=all_metrics[key]["common_evaluation_2100"]
        row=dict(model=key,eval_count=2100,**{f:common["uncalibrated"][f] for f in table_fields})
        row.update({"calibrated_"+f:common["calibrated"][f] for f in ("nll","brier_multiclass","brier_positive","ece_15_equal_width_top_label")})
        row.update(temperature=common["calibrated"]["temperature"],training_time_seconds=infos[key]["training_time_seconds"],
                   training_time_scope=infos[key]["time_scope"],parameters=infos[key]["total_parameters"])
        common_table.append(row)
    csv_write(OUT/"comparison_full2500.csv",full_table)
    csv_write(OUT/"comparison_common2100.csv",common_table)
    make_report(infos,all_metrics,pairs,pilot,epoch_logs)
    make_figures(all_metrics,epoch_logs)
    sources=OUT/"reproduction"; sources.mkdir(exist_ok=True)
    shutil.copy2(ROOT/"tests/test_comparison_metrics.py",sources/"test_comparison_metrics.py")
    for name in ("export_comparison_predictions.py","comparison_metrics.py","build_comparison_report.py"):
        shutil.copy2(ROOT/"scripts"/name,sources/name)
    for path in (ROOT/"pyproject.toml",ROOT/"uv.lock",ROOT/"configs/laya_model_lock.json",ROOT/"configs/model_lock.json"):
        shutil.copy2(path,sources/path.name)
    write_json(sources/"frozen_split_hashes.json",json.loads((ROOT/"reports/split_manifest.json").read_text())["jsonl_sha256"])
    shutil.copy2(ROOT/"reports/split_manifest.json",sources/"frozen_split_manifest.json")
    shutil.copy2(ROOT/"reports/environment.json",sources/"original_environment.json")
    write_json(sources/"analysis_reproduction_notes.json",{
        "python_version":"3.12.14", "weights_included":False, "review_text_included":False,
        "commands_run":["export_comparison_predictions.py --batch-size 8","build_comparison_report.py"],
        "test_results":"10 unittest tests passed; including four analytical metric checks",
        "source_directory":"scripts/ in original project; standalone bundle paths are for audit",
        "source_data_and_model_weights_required_for_inference":True})
    checks={str(p.relative_to(OUT)):file_digest(p) for p in OUT.rglob("*") if p.is_file() and p.name!="checksums.json"}
    write_json(OUT/"checksums.json",checks)
    archive=ROOT/"reports/ModernBERT-Laya-Comparison.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file(): z.write(path,"ModernBERT-Laya-Comparison/"+str(path.relative_to(OUT)))
    print(json.dumps({"archive":str(archive),"bytes":archive.stat().st_size,"models":{
        k:{"full_accuracy":v["full_validation_2500"]["uncalibrated"]["accuracy"],
           "common2100_accuracy":v["common_evaluation_2100"]["uncalibrated"]["accuracy"]} for k,v in all_metrics.items()}},indent=2))


def make_report(infos,results,pairs,pilot,epoch_logs):
    keys=list(RUNS)
    lines=["# ModernBERT 与 Laya-CE / Laya-RLCD 实验分析", "",
           "本报告由冻结配置、训练日志、已保存权重的离线推理结果自动生成；未重新训练，未访问官方测试集。", "",
           "## 结论与证据边界", "",
           "1. **Decision Head 的独立收益尚未验证。** 缺少 ModernBERT-large + 普通分类头这一控制组；当前模型规模、输入长度、头结构、预训练及训练预算均不同。", 
           "2. **CE 与 RLCD 的干净对照是两份 pilot。** 它们模型、训练/校准样本和超参数一致，只有损失不同。完整 2,500 条上 RLCD=93.72%、CE=93.52%，仅相差 5 条；公共 2,100 条配对 McNemar p=1，未观察到准确率差异的统计证据。CE 的 AUROC、未校准 NLL/Brier 更好；统一温度校准后 CE 的 NLL/Brier 仍更低，但 RLCD 的 ECE 更低。因此不能说 CE 在所有校准指标上全面占优。全量 CE 对 pilot RLCD 是不同预算方案比较，不能当作损失函数优劣证据。",
           "3. **额外参数是否值得还没有完整成本证据。** 已有准确率收益和参数量，缺少统一训练计时、统一推理延迟/显存测量及同骨干消融。", "",
           "## 模型与训练配置", "",
           "| 项目 | "+" | ".join(NAMES[k] for k in keys)+" |", "|---|"+"---|"*len(keys)]
    fields=[("Pretrained checkpoint","pretrained_checkpoint"),("总参数","total_parameters"),("可训练参数（声明）","trainable_parameters"),("Encoder 参数","encoder_parameters"),
            ("Head 参数","head_parameters"),("最大总长度","max_sequence_length"),("实际轮数","epochs"),("最佳轮数","best_epoch"),
            ("设备 batch","per_device_batch_size"),("梯度累积","gradient_accumulation"),("有效 batch","effective_batch_size"),
            ("Encoder 初始 LR","encoder_lr"),("Head 初始 LR","head_lr"),("Optimizer","optimizer"),("Weight decay","weight_decay"),
            ("Scheduler","scheduler"),("Warmup 比例","warmup_fraction"),("Seed","seed"),("Loss","loss"),("Freeze Encoder","freeze_encoder"),
            ("训练精度","precision_training"),("保存精度","saved_precision"),("梯度训练样本","gradient_train_size"),
            ("原生校准样本","native_calibration_size"),("原生验证样本","native_validation_size"),("参数更新次数","optimizer_updates"),
            ("耗时记录（秒，口径不同）","training_time_seconds")]
    for title,field in fields: lines.append("| "+title+" | "+" | ".join(str(infos[k].get(field,"未记录")) for k in keys)+" |")
    lines += ["", "硬件均为 Apple M5 Pro、64 GB 统一内存，没有独立显存。ModernBERT 使用 MLX，Laya 使用 PyTorch/MPS。", "",
              "实测参数量：基线 149,015,810，Laya 421,293,827，约 2.83 倍；Laya Encoder 为 394,781,696，Encoder 外为 26,512,131。相对基线的参数增量约 90% 来自升级 Encoder，而非新增 Head。Laya 当前问题模板占 45 token，实际影评预算 466；基线总长 256 还包含特殊 token。", "",
              "ModernBERT 耗时是 forward/backward/update 的计时求和；Laya 是整个 finetune 流程计时，含评估、保存及进程暂停。本轮全量 CE 曾暂停，不能用 8738 秒直接计算相对训练效率。MPS 结束时内存不是峰值，Laya 峰值未记录。", "",
              "Laya 总参数中的 action head 虽 requires_grad=True，但分类损失不使用该分支；其参数数量及损失关联参数另见各 manifest。2 层 Transformer Decision Head、MASK 选项打分、输入模板和原问题 schema 均保留。", "",
              "## 数据、选点与校准协议", "",
              "IMDB 固定划分：22,500 条训练、2,500 条开发验证、25,000 条官方测试，原始划分 seed=42。Head/prefix 截断；Laya 的问题和选项占用 512 总预算，实际影评预算见 manifest。", "",
              "ModernBERT 用完整开发验证 Macro-F1 最大选 epoch 3（并列时 NLL 更低）；Laya 仅训练 1 轮，报告的是唯一最终 checkpoint，不能称其经过多轮最优选择。", "",
              "统一校准分析：从开发验证集用 seed=20261009 按类抽 400 条（每类 200）拟合各模型温度，其余 2,100 条独立计算校准指标。公共校准/评估 ID 严格互斥，所有模型行对齐；温度范围 [0.05,20]。这是一套分析用的统一温度，不是覆盖模型内置温度。", "",
              "由于此前选 checkpoint/做 pilot 已使用开发验证数据，以上不是全新最终测试，也不用于声称无选择偏差。原生 Laya 校准则分别只用原训练划分中留出的 400/128 条样本。", "",
              "![统一公共评估集的校准曲线](plots/reliability.png)", "",
              "低置信度分箱可能只有少量样本，因此局部波动不能单独解释为稳定规律；每箱样本数保存在 metrics.json。", "",
              "## 完整 2,500 条验证：未校准分类成绩", "",
              "| 指标 | "+" | ".join(NAMES[k] for k in keys)+" |", "|---|"+"---|"*len(keys)]
    for title,field in [("Accuracy ↑","accuracy"),("Macro-F1 ↑","macro_f1"),("Precision macro ↑","precision_macro"),("Recall macro ↑","recall_macro"),("AUROC ↑","auroc"),("NLL ↓","nll"),("Brier（双类和）↓","brier_multiclass"),("ECE ↓","ece_15_equal_width_top_label")]:
        lines.append("| "+title+" | "+" | ".join(f'{results[k]["full_validation_2500"]["uncalibrated"][field]:.6f}' for k in keys)+" |")
    lines += ["", "## 公共 2,100 条评估：统一校准前后", "",
              "所有模型使用相同评估 ID；表中 RLCD 仍为 pilot，训练预算差异没有消失。", "",
              "| 指标 | "+" | ".join(NAMES[k] for k in keys)+" |", "|---|"+"---|"*len(keys)]
    for title,stage,field in [("Accuracy ↑","uncalibrated","accuracy"),("Macro-F1 ↑","uncalibrated","macro_f1"),
                             ("Precision macro ↑","uncalibrated","precision_macro"),("Recall macro ↑","uncalibrated","recall_macro"),
                             ("AUROC ↑","uncalibrated","auroc"),("NLL ↓","uncalibrated","nll"),("Brier ↓","uncalibrated","brier_multiclass"),
                             ("ECE ↓","uncalibrated","ece_15_equal_width_top_label"),("Calibrated NLL ↓","calibrated","nll"),
                             ("Calibrated Brier ↓","calibrated","brier_multiclass"),("Calibrated ECE ↓","calibrated","ece_15_equal_width_top_label"),
                             ("Temperature","calibrated","temperature")]:
        lines.append("| "+title+" | "+" | ".join(f'{results[k]["common_evaluation_2100"][stage][field]:.6f}' for k in keys)+" |")
    lines += ["", "Brier 主表统一为双类别平方误差之和；正类平方误差另存，恰为其一半。Precision/Recall 主表为宏平均，JSON 还保存正类指标。AUROC 根据原始 logit margin 计算，避免概率饱和造成额外并列。NLL 使用 logaddexp 稳定计算，不对饱和概率硬裁剪。", "",
              "## 同预算 CE pilot vs RLCD pilot", "",
              "两者各 896 条梯度训练、128 条原生校准、1 轮、112 次参数更新。原 256 条验证上的结果另见 matched_pilot_256.json；两者现在还同时在完整 2,500 条及公共 2,100 条评估。", "",
              "| 原 256 条指标 | CE pilot | RLCD pilot |", "|---|---|---|"]
    for title,stage,field in [("Accuracy","uncalibrated","accuracy"),("原生校准 NLL","native_calibrated","nll"),("原生校准 Brier","native_calibrated","brier_multiclass"),("原生校准 ECE","native_calibrated","ece_15_equal_width_top_label")]:
        lines.append("| "+title+" | "+" | ".join(f'{pilot[k][stage][field]:.6f}' for k in ("laya_ce_pilot","laya_rlcd_pilot"))+" |")
    lines += ["", "## 配对统计、共同错误与长度", "",
              "McNemar 使用同一 2,100 条公共评估、双侧精确二项检验；6 个探索性模型对同时给出 Holm 修正。显著性只针对当前固定权重在该开发子集的配对错误差异，不代表多种子稳健收益或 Decision Head 因果收益。", "",
              "| A vs B | 仅 A 正确 | 仅 B 正确 | 共同错误 | 原始 p | Holm p |", "|---|---:|---:|---:|---:|---:|"]
    for p in pairs:
        lines.append(f'| {p["model_a"]} vs {p["model_b"]} | {p["a_only_correct"]} | {p["b_only_correct"]} | {p["both_wrong"]} | {p["mcnemar_exact_two_sided_p"]:.6g} | {p["holm_adjusted_p_over_six_exploratory_pairs"]:.6g} |')
    lines += ["", "公共 2,100 条上，Laya-CE full 对基线的原始 p=0.01783，六对 Holm 修正后 p=0.07131。因此可以报告该配置的观察准确率收益，但在这组六重探索性检验下未达到 0.05 显著性门槛，不能表述为已经证明稳定优势。"]
    lines += ["", "每份 metrics.json 包括短/长评论分层、错误样本平均置信度、高置信度错误数量、混淆矩阵和可靠性分箱。长度统一使用基线 tokenizer 完整编码（含特殊 token），<=256 与 >256 分组；长度不是因果控制，Laya 有不同文本预算。", "",
              "## 过拟合与日志完整性", "",
              "ModernBERT 第 1/2/3 轮验证 NLL 为 0.219651/0.189119/0.284442，Accuracy 为 94.16%/94.48%/94.64%；固定训练监测子集准确率继续升至 99.51%。第三轮出现概率质量恶化及训练/验证差距扩大的迹象，不能说分类准确率同时下降。", "",
              "Laya 每份只有一轮，不能凭一轮训练 loss 判断多轮过拟合时间。RLCD 的训练目标含奖励项，数值及正负不能与 CE 损失直接比较。图中 RLCD 单独放置。", "",
              "![逐轮训练及验证曲线](plots/epoch_curves.png)", "",
              "![稀疏 step 训练损失记录](plots/step_loss.png)", "",
              "ModernBERT 原逐 step 控制台日志未持久化，缺失如实标注；保留真实逐 epoch 历史。Laya train_log.csv 为稀疏日志采样，LR 是按已锁定调度器重建，列名明确注明，不能当作原始实测。", "",
              "## 下一步如何回答三个问题", "",
              "- 验证 Head：新增 ModernBERT-large + 普通头，尽量统一预训练来源、文本预算、梯度训练样本、优化器预算；若起始预训练不同仍要注明。进一步做 Decision Head、选项语义和顺序增强消融。", 
              "- 验证损失：当前 pilot 已可对照；若要下正式全量结论，还缺同预算全量 RLCD 与多个随机种子。当前包不虚构这一实验，也不替用户启动训练。", 
              "- 验证成本：在同一硬件和同步方式下补测驻留推理延迟、吞吐和峰值内存；训练需分别记录纯参数更新时间与完整流程时间，并排除暂停。", "",
              "## 文件说明与复现", "",
              "每个模型目录含 manifest.json/config.json、metrics.json、trainer_state.json、epoch_log.csv/train_log.csv、predictions.csv（统一 2,100 行）、predictions_all2500.csv、calibration_predictions.csv、原始 logits 与源报告。", "",
              "plots/ 保存可靠性图、epoch 曲线与训练损失图。analysis_protocol.json 保存公共校准/评估 ID；checksums.json 记录包内文件哈希。ZIP 不含模型权重和完整评论文本。", "",
              "在原项目根目录：", "", "```bash",
              "uv run --offline --locked --extra metal --extra laya python scripts/export_comparison_predictions.py --batch-size 8",
              "uv run --offline --locked --extra analysis python scripts/build_comparison_report.py", "```", "",
              "reproduction/ 保存分析代码与依赖锁，仅供审计；重新推理仍需要原项目数据和本地权重。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def make_figures(results,epochs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots=OUT/"plots"; plots.mkdir(exist_ok=True)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    for stage,ax in zip(("uncalibrated","calibrated"),axs):
        ax.plot([0,1],[0,1],"--",color="gray",linewidth=1)
        for key in RUNS:
            bins=results[key]["common_evaluation_2100"][stage]["reliability_bins"]
            bins=[b for b in bins if b["count"]]
            ax.plot([b["confidence"] for b in bins],[b["accuracy"] for b in bins],"o-",markersize=3,label=NAMES[key])
        ax.set(xlabel="Mean confidence",ylabel="Empirical accuracy",title=stage+" / common 2100",xlim=(.45,1),ylim=(0,1.02))
        ax.grid(alpha=.2)
    axs[1].legend(fontsize=8)
    fig.savefig(plots/"reliability.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13,4),layout="constrained")
    for key,rows in epochs.items():
        x=[r["epoch"] for r in rows]
        if key!="laya_rlcd_pilot": axs[0].plot(x,[r["train_loss"] for r in rows],"o-",label=NAMES[key]+" train CE")
        axs[0].plot(x,[r["val_loss"] for r in rows],"s--",label=NAMES[key]+" val NLL")
        axs[1].plot(x,[r["val_accuracy"] for r in rows],"o-",label=NAMES[key])
        axs[2].plot(x,[r["val_macro_f1"] for r in rows],"o-",label=NAMES[key])
    for ax,title in zip(axs,("CE train / raw val NLL", "Validation accuracy", "Validation Macro-F1")):
        ax.set(xlabel="Epoch",title=title,xticks=[1,2,3]); ax.grid(alpha=.2)
    axs[0].legend(fontsize=6); axs[1].legend(fontsize=7)
    fig.savefig(plots/"epoch_curves.png",dpi=180); plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for key in ("laya_ce_full","laya_ce_pilot","laya_rlcd_pilot"):
        with (OUT/key/"train_log.csv").open(encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
        axis=axs[1] if key=="laya_rlcd_pilot" else axs[0]
        axis.plot([int(r["micro_step"]) for r in rows],[float(r["loss_logged_4dp"]) for r in rows],alpha=.6,label=NAMES[key],linewidth=.7)
    for ax,title in zip(axs,("Sparse CE training losses", "Sparse RLCD losses (different objective)")):
        ax.set(xlabel="Micro-step",ylabel="Logged loss",title=title); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(plots/"step_loss.png",dpi=180); plt.close(fig)


if __name__=="__main__":
    main()
