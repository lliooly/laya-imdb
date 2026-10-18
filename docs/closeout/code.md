# 源码与测试索引

路径保持不变；模块说明由 AST 静态读取，不导入或执行训练入口。CLI 列只表示定义了 main，执行前仍需阅读对应协议。局部导入包括函数内部的静态导入，不保证运行时全部经过。

## 全序列查询头

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/run_token_query_study.py](../../scripts/run_token_query_study.py) | Frozen full-token mean/dense/top32 readout experiment with sealed evaluation. | 有 | common, comparison_metrics, export_laya_posthoc, laya_head_control, run_bow_nb_study, run_final_cpu_alternatives_v2, run_laya_frozen_probe, token_query_head, train_laya |
| [scripts/token_query_head.py](../../scripts/token_query_head.py) | Task-conditioned dense/sparse document readout, independent of data access. | — |  |
| [scripts/verify_token_query_study.py](../../scripts/verify_token_query_study.py) | Independent NumPy/SciPy forward and metric checks for token query heads. | 有 | common, run_final_cpu_alternatives_v2, run_token_query_study |
| [tests/test_token_query_head.py](../../tests/test_token_query_head.py) | 无模块说明；按文件名与调用关系定位。 | — | run_token_query_study, token_query_head |

## 共享实现与历史辅助

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/README.md](../../scripts/README.md) |  | — |  |
| [scripts/analyze_accuracy_headroom.py](../../scripts/analyze_accuracy_headroom.py) | Accuracy-headroom quantitative analysis on frozen development artifacts. | 有 |  |
| [scripts/analyze_laya_errors.py](../../scripts/analyze_laya_errors.py) | Audit saved Laya-CE validation errors on CPU, without loading model weights. | 有 | common, comparison_metrics, train_laya |
| [scripts/apply_laya_94_user_label_revision.py](../../scripts/apply_laya_94_user_label_revision.py) | Apply explicit user label decisions to a new training-data version only. | — | common |
| [scripts/apply_laya_label_revision_v2.py](../../scripts/apply_laya_label_revision_v2.py) | Build a separate reviewed data version, retaining original and v1 datasets. | — | common |
| [scripts/audit_bert_bilstm_literature.py](../../scripts/audit_bert_bilstm_literature.py) | Static AST and synthetic-source audit; never import downloaded code or IMDb. | 有 | common |
| [scripts/benchmark_inference.py](../../scripts/benchmark_inference.py) | Warm resident baseline latency on validation reviews; no official test access. | 有 | benchmark_day2, cache_tokens, common, day2_data, day2_runtime, modernbert_mlx |
| [scripts/build_comparison_report.py](../../scripts/build_comparison_report.py) | Build a source-backed comparison bundle from exported validation logits. | 有 | common, comparison_metrics |
| [scripts/check_environment.py](../../scripts/check_environment.py) | MLX GPU arithmetic, autodiff, optimizer and save/reload smoke test. | 有 | common |
| [scripts/common.py](../../scripts/common.py) | Shared paths and deterministic preprocessing; no model dependencies. | — |  |
| [scripts/compare_laya_backends.py](../../scripts/compare_laya_backends.py) | Offline same-checkpoint Laya backend parity and synchronized inference benchmark. | 有 | common, comparison_metrics, day2_runtime, train_laya |
| [scripts/comparison_metrics.py](../../scripts/comparison_metrics.py) | Shared binary metrics, stable temperature fitting and exact paired testing. | — |  |
| [scripts/download_model.py](../../scripts/download_model.py) | Download original pretrained ModernBERT at the Day1 tokenizer revision. | 有 | common |
| [scripts/export_comparison_predictions.py](../../scripts/export_comparison_predictions.py) | Export saved-checkpoint logits on frozen validation data; never loads the test split. | 有 | benchmark_day2, common, day2_data, day2_runtime, modernbert_mlx, train_laya |
| [scripts/export_laya_94_candidate_information.py](../../scripts/export_laya_94_candidate_information.py) | Export only the 94 training candidates with original text and existing review. | 有 | common |
| [scripts/laya_evaluation_completion.py](../../scripts/laya_evaluation_completion.py) | Guards for completing only the preregistered five remaining evaluations. | — | laya_checkpoint_delivery, laya_modal_deferred, laya_modal_recovery |
| [scripts/laya_runtime.py](../../scripts/laya_runtime.py) | Opt-in formal CUDA determinism; configure workspace before importing Laya/Torch. | — |  |
| [scripts/plot_laya_errors.py](../../scripts/plot_laya_errors.py) | Plot saved validation error strata without reading review text or test data. | 有 | common |
| [scripts/predict.py](../../scripts/predict.py) | Offline Mac inference: Laya-CE / MLX FP16 by default; ModernBERT remains selectable. | 有 | common, comparison_metrics, day2_data, day2_runtime, modernbert_mlx, train_laya |
| [scripts/prepare_laya.py](../../scripts/prepare_laya.py) | Download a pinned English Laya checkpoint and record local file hashes. | 有 | common |
| [scripts/probe_laya_embedding_backward.py](../../scripts/probe_laya_embedding_backward.py) | Synthetic ModernBERT embedding reproduction; reads configuration, never weights/data. | 有 | common, laya_determinism, laya_short_trace |
| [scripts/run_laya_strict_pair.py](../../scripts/run_laya_strict_pair.py) | Bounded CUDA smoke + seed43 paired experiment; never reads the official test body. | 有 | common, comparison_metrics |
| [scripts/validate_resume.py](../../scripts/validate_resume.py) | Verify restored optimizer + RNG produce the same next update as uninterrupted training. | 有 | checkpointing, common, day2_runtime |
| [tests/README.md](../../tests/README.md) |  | — |  |
| [tests/test_comparison_metrics.py](../../tests/test_comparison_metrics.py) | 无模块说明；按文件名与调用关系定位。 | — | comparison_metrics |
| [tests/test_laya_evaluation_completion.py](../../tests/test_laya_evaluation_completion.py) | Budget regression guards for a narrowly scoped inference-only completion. | — | laya_evaluation_completion |
| [tests/test_laya_preflight.py](../../tests/test_laya_preflight.py) | CPU-only checks prevent a missing CUDA device from passing training preflight. | — | train_laya |
| [tests/test_laya_runtime.py](../../tests/test_laya_runtime.py) | Guard strict admission and reject changing workspace after CUDA initialization. | — | laya_runtime |

## 后端与数值

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/check_cuda_mlx_parity.py](../../scripts/check_cuda_mlx_parity.py) | Verify CUDA smoke checkpoint inputs and predictions on Mac MLX FP16. | 有 | common, comparison_metrics, day2_runtime, train_laya |
| [scripts/cuda_numerics.py](../../scripts/cuda_numerics.py) | Fail-fast CUDA training guards; restore all upstream functions on exit. | — |  |
| [scripts/debug_parity.py](../../scripts/debug_parity.py) | Layerwise diagnostic retained for investigating numerical drift. | 有 | common, day2_data, day2_runtime, modernbert_mlx |
| [scripts/diagnose_laya_mlx_precision.py](../../scripts/diagnose_laya_mlx_precision.py) | Posthoc numerical diagnosis, not a model-selection or deployment experiment. | — | common, export_laya_posthoc, laya_fusion_inference, laya_mlx_fusion_inference, train_laya, validate_laya_fusion_inference, validate_laya_mlx_fusion |
| [scripts/laya_determinism.py](../../scripts/laya_determinism.py) | Explicit diagnostic protocols; never enabled by the formal trainer. | — |  |
| [scripts/laya_mlx_precise_inference.py](../../scripts/laya_mlx_precise_inference.py) | Explicit full-FP32 Metal inference with reduced-precision matrix operations disabled. | — | laya_mlx_fusion_inference |
| [scripts/laya_short_trace.py](../../scripts/laya_short_trace.py) | Diagnostic hooks for existing FP32 training; never changes the loss or RNG. | — |  |
| [scripts/modernbert_mlx.py](../../scripts/modernbert_mlx.py) | Padded ModernBERT encoder in MLX with an independent mean-pool classifier. | — | common, day2_runtime |
| [scripts/reproduce_mlx_matmul_precision.py](../../scripts/reproduce_mlx_matmul_precision.py) | Model/data-free deterministic FP32 matmul reproduction, run in a fresh process. | 有 |  |
| [scripts/run_laya_determinism_matrix.py](../../scripts/run_laya_determinism_matrix.py) | Bounded, fresh-process diagnostics; no full training or post-training eval. | 有 | common, laya_determinism |
| [scripts/run_laya_short_trace.py](../../scripts/run_laya_short_trace.py) | Fresh-process original/control diagnostics using the full historical budget. | 有 | common, laya_determinism, laya_short_trace |
| [scripts/run_laya_strict_trace.py](../../scripts/run_laya_strict_trace.py) | Fresh-process 16-update admission check for the current formal strict entry. | 有 | common, laya_short_trace, train_laya |
| [scripts/trace_laya_mlx_operators.py](../../scripts/trace_laya_mlx_operators.py) | Localize one preselected worst FP32 discrepancy using CPU FP32 layer outputs. | 有 | common, diagnose_laya_mlx_precision, export_laya_posthoc, laya_mlx_fusion_inference, train_laya |
| [scripts/trace_laya_mlx_precision.py](../../scripts/trace_laya_mlx_precision.py) | Localize one preselected worst FP32 discrepancy using CPU FP32 layer outputs. | 有 | common, diagnose_laya_mlx_precision, export_laya_posthoc, laya_mlx_fusion_inference, train_laya |
| [scripts/trace_laya_mlx_qkv.py](../../scripts/trace_laya_mlx_qkv.py) | Localize one preselected worst FP32 discrepancy using CPU FP32 layer outputs. | 有 | common, diagnose_laya_mlx_precision, export_laya_posthoc, laya_mlx_fusion_inference, train_laya |
| [scripts/trace_laya_mlx_qkv_precise.py](../../scripts/trace_laya_mlx_qkv_precise.py) | Localize one preselected worst FP32 discrepancy using CPU FP32 layer outputs. | 有 | common, diagnose_laya_mlx_precision, export_laya_posthoc, laya_mlx_fusion_inference, train_laya |
| [scripts/validate_laya_mlx_precise.py](../../scripts/validate_laya_mlx_precise.py) | Frozen Mac precise MLX FP32 fusion validation; no fitting, training or official test access. | 有 | common, export_laya_posthoc, laya_decision_views, laya_fusion_inference, laya_mlx_fusion_inference, laya_mlx_precise_inference, run_laya_probability_fusion, train_laya, validate_laya_fusion_inference |
| [tests/test_cuda_numerics.py](../../tests/test_cuda_numerics.py) | Verify nonfinite training cannot advance optimizer/scheduler, and hooks are restored. | — | cuda_numerics |
| [tests/test_laya_mlx_precise.py](../../tests/test_laya_mlx_precise.py) | Guard the explicit precision mode before loading any weights or touching Metal. | — | common, laya_mlx_precise_inference, validate_laya_mlx_precise |
| [tests/test_laya_short_trace.py](../../tests/test_laya_short_trace.py) | Diagnostic instrumentation must preserve parameters and RNG, and restore hooks. | — | common, cuda_numerics, laya_optimization, laya_resampling, laya_short_trace |

## 头部与表示

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/analyze_laya_head_control.py](../../scripts/analyze_laya_head_control.py) | Verify completed L2 receipts and analyze cached logits without new model evaluation. | — | common, comparison_metrics |
| [scripts/laya_frozen_probe.py](../../scripts/laya_frozen_probe.py) | Frozen pre-head representations and convex linear readout, with no data access. | — |  |
| [scripts/laya_head_control.py](../../scripts/laya_head_control.py) | CE-only Laya heads, explicit initialization and lossless FP32 checkpoints. | — | common, comparison_metrics, laya_resampling |
| [scripts/run_laya_frozen_probe.py](../../scripts/run_laya_frozen_probe.py) | Extract frozen Laya features, calibrate linear probes, then evaluate development. | 有 | common, export_laya_posthoc, laya_decision_views, laya_frozen_probe, run_laya_risk_coverage, train_laya |
| [scripts/run_laya_head_control.py](../../scripts/run_laya_head_control.py) | One fixed L2 run; calibration reload and gated development run in new processes. | 有 | common, cuda_numerics, export_laya_posthoc, laya_decision_views, laya_head_control, laya_runtime, train_laya |
| [scripts/run_laya_head_study.py](../../scripts/run_laya_head_study.py) | Bounded sequential CUDA admission, six fixed runs, then gated development. | 有 | common, laya_frozen_probe, run_laya_head_control, run_laya_risk_coverage |
| [tests/test_laya_frozen_probe.py](../../tests/test_laya_frozen_probe.py) | 无模块说明；按文件名与调用关系定位。 | — | laya_frozen_probe |
| [tests/test_laya_head_control.py](../../tests/test_laya_head_control.py) | 无模块说明；按文件名与调用关系定位。 | — | common, cuda_numerics, export_laya_posthoc, laya_head_control, run_laya_head_control |

## 异构模型

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/deberta_bf16_study.py](../../scripts/deberta_bf16_study.py) | Isolated BF16 adapter; all historical FP32 modules and bundles stay unchanged. | 有 | common, deberta_bf16_training, deberta_inputs, laya_checkpoint_delivery |
| [scripts/deberta_bf16_training.py](../../scripts/deberta_bf16_training.py) | BF16 forward only, with the frozen FP32 optimizer, master weights and RNG. | — | deberta_training |
| [scripts/deberta_budget_study.py](../../scripts/deberta_budget_study.py) | Budget-bound BF16 training with durable intermediate checkpoints and exact resume. | 有 | common, deberta_bf16_study, deberta_bf16_training, deberta_budget_training, deberta_inputs, laya_checkpoint_delivery, laya_head_control |
| [scripts/deberta_budget_training.py](../../scripts/deberta_budget_training.py) | Exact positions for budget-limited training; no scientific recipe changes. | — |  |
| [scripts/deberta_final_analysis.py](../../scripts/deberta_final_analysis.py) | Offline development risk/coverage and exact three-pipeline common-error export. | 有 | common, comparison_metrics, run_laya_risk_coverage |
| [scripts/deberta_final_evaluation.py](../../scripts/deberta_final_evaluation.py) | Frozen local CPU test evaluation. No training, cloud compute, fitting or test selection. | 有 | common, comparison_metrics, deberta_inputs, deberta_training, laya_checkpoint_delivery, run_laya_risk_coverage |
| [scripts/deberta_image_build.py](../../scripts/deberta_image_build.py) | Importable, dependency-free image layer function; no training code imports. | — |  |
| [scripts/deberta_inputs.py](../../scripts/deberta_inputs.py) | Frozen partitions and tokenizer windows; preprocessing does not require torch. | — | common |
| [scripts/deberta_study.py](../../scripts/deberta_study.py) | Frozen single DeBERTa study, full epoch bundles, admission and evaluation. | 有 | analyze_laya_length_router, common, comparison_metrics, deberta_inputs, deberta_training, laya_checkpoint_delivery, laya_head_control, laya_robust_training |
| [scripts/deberta_training.py](../../scripts/deberta_training.py) | FP32 sequence classification with sample-weighted accumulation and exact resume. | — | common, deberta_inputs, laya_head_control, laya_robust_training |
| [scripts/modal_deberta_bf16.py](../../scripts/modal_deberta_bf16.py) | One bounded DeBERTa admission and optional two-epoch run, never auto-retried. | 有 | common, deberta_bf16_study, deberta_image_build, deberta_inputs, laya_checkpoint_delivery |
| [scripts/modal_deberta_budget7.py](../../scripts/modal_deberta_budget7.py) | One user-authorized BF16 attempt; shared absolute compute window, no retries. | 有 | common, deberta_budget_study, deberta_inputs, laya_checkpoint_delivery |
| [scripts/modal_deberta_study.py](../../scripts/modal_deberta_study.py) | One bounded DeBERTa admission and optional two-epoch run, never auto-retried. | 有 | common, deberta_image_build, deberta_inputs, deberta_study, laya_checkpoint_delivery |
| [scripts/prepare_deberta_bf16.py](../../scripts/prepare_deberta_bf16.py) | Reuse byte-identical inputs and persistent assets; reserve a new bounded BF16 run. | 有 | common, deberta_bf16_study, deberta_inputs |
| [scripts/prepare_deberta_budget7.py](../../scripts/prepare_deberta_budget7.py) | Freeze new sources and latest-balance reservation, preserving all old experiments. | 有 | common, deberta_budget_study, deberta_inputs |
| [scripts/prepare_deberta_study.py](../../scripts/prepare_deberta_study.py) | Freeze source/input hashes, preflight evidence and a single $5 reservation. | 有 | common, deberta_inputs, deberta_training |
| [scripts/receive_deberta_bf16.py](../../scripts/receive_deberta_bf16.py) | CPU-only background delivery; all Modal operations use one native async loop. | 有 | common, deberta_bf16_study, laya_checkpoint_delivery |
| [scripts/receive_deberta_budget7.py](../../scripts/receive_deberta_budget7.py) | CPU-only background delivery; all Modal operations use one native async loop. | 有 | common, deberta_budget_study, laya_checkpoint_delivery |
| [scripts/receive_deberta_study.py](../../scripts/receive_deberta_study.py) | CPU-only background delivery; all Modal operations use one native async loop. | 有 | common, deberta_study, laya_checkpoint_delivery |
| [scripts/run_laya_deberta_calibration_weight.py](../../scripts/run_laya_deberta_calibration_weight.py) | One CPU calibration-only probability weight; separate frozen evaluation. | — | common, comparison_metrics, laya_probability_fusion, run_bow_nb_study |
| [tests/test_deberta_bf16.py](../../tests/test_deberta_bf16.py) | 无模块说明；按文件名与调用关系定位。 | — | deberta_bf16_study, deberta_bf16_training, deberta_training |
| [tests/test_deberta_budget_training.py](../../tests/test_deberta_budget_training.py) | 无模块说明；按文件名与调用关系定位。 | — | common, deberta_bf16_training, deberta_budget_study, deberta_budget_training, deberta_training |
| [tests/test_deberta_final_evaluation.py](../../tests/test_deberta_final_evaluation.py) | 无模块说明；按文件名与调用关系定位。 | — | common, comparison_metrics, deberta_final_analysis, deberta_final_evaluation |
| [tests/test_deberta_training.py](../../tests/test_deberta_training.py) | 无模块说明；按文件名与调用关系定位。 | — | deberta_inputs, deberta_study, deberta_training |
| [tests/test_laya_deberta_calibration_weight.py](../../tests/test_laya_deberta_calibration_weight.py) | 无模块说明；按文件名与调用关系定位。 | — | run_laya_deberta_calibration_weight |

## 损失与调度

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/analyze_laya_robust_loss.py](../../scripts/analyze_laya_robust_loss.py) | Analyze the fixed three paired seeds only after six local evaluation receipts. | — | common, comparison_metrics, laya_checkpoint_delivery, laya_modal_deferred |
| [scripts/laya_optimization.py](../../scripts/laya_optimization.py) | Opt-in warmup and update diagnostics for the pinned, single-process trainer. | — | common, laya_rdrop |
| [scripts/laya_rdrop.py](../../scripts/laya_rdrop.py) | Training-only, paired dropout forwards for the pinned single-process loop. | — | common |
| [scripts/laya_robust_audit.py](../../scripts/laya_robust_audit.py) | Input and update journals with verified epoch-prefix recovery. | — | laya_checkpoint_delivery, laya_head_control |
| [scripts/laya_robust_training.py](../../scripts/laya_robust_training.py) | Fixed CE/GCE experiment; explicit epoch-boundary optimizer and RNG recovery. | — | laya_head_control, laya_resampling |
| [scripts/prepare_laya_robust_study.py](../../scripts/prepare_laya_robust_study.py) | Prepare a new immutable protocol; does not start CUDA or connect to a host. | — | common, laya_checkpoint_delivery, run_laya_head_control, run_laya_robust_loss |
| [scripts/run_laya_robust_loss.py](../../scripts/run_laya_robust_loss.py) | Fixed robust-loss runs; epoch and final delivery are mandatory progress gates. | 有 | common, export_laya_posthoc, laya_checkpoint_delivery, laya_head_control, laya_robust_audit, laya_robust_training, laya_runtime, run_laya_head_control, run_laya_robust_study, train_laya |
| [scripts/run_laya_robust_study.py](../../scripts/run_laya_robust_study.py) | Admit, train six paired models, and gate development on verified local delivery. | 有 | common, laya_checkpoint_delivery, run_laya_robust_loss |
| [tests/test_analyze_laya_robust_loss.py](../../tests/test_analyze_laya_robust_loss.py) | 无模块说明；按文件名与调用关系定位。 | — | analyze_laya_robust_loss |
| [tests/test_laya_optimization.py](../../tests/test_laya_optimization.py) | CPU regressions for the pinned training loop and opt-in experiment adapter. | — | cuda_numerics, laya_optimization |
| [tests/test_laya_rdrop.py](../../tests/test_laya_rdrop.py) | Numerical and isolation checks for the training-only consistency loss. | — | laya_optimization, laya_rdrop |
| [tests/test_laya_robust_training.py](../../tests/test_laya_robust_training.py) | 无模块说明；按文件名与调用关系定位。 | — | common, export_laya_posthoc, laya_head_control, laya_robust_audit, laya_robust_training, run_laya_head_control |

## 收尾与汇总

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/build_all_experiment_metrics.py](../../scripts/build_all_experiment_metrics.py) | Consolidate saved experiment metrics; never imports models or reads review text. | 有 |  |
| [scripts/build_closeout_catalog.py](../../scripts/build_closeout_catalog.py) | Build reproducible source/docs/report catalogs without model or dataset access. | — |  |
| [tests/test_closeout_catalog.py](../../tests/test_closeout_catalog.py) | Verify catalog boundaries, static analysis, and link checking on fixtures. | — | build_closeout_catalog |

## 数据与分词

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/analyze_tokenizers.py](../../scripts/analyze_tokenizers.py) | Analyze original Rust-backed tokenizers on train/validation only. | 有 | common |
| [scripts/cache_tokens.py](../../scripts/cache_tokens.py) | Verified Rust tokenizer caches for frozen train/validation data only. | 有 | common |
| [scripts/day2_data.py](../../scripts/day2_data.py) | Verified train/validation input access; no test path is accepted. | — | common, day2_runtime, modernbert_mlx |
| [scripts/prepare_data.py](../../scripts/prepare_data.py) | Download the official archive, audit it and freeze duplicate-aware splits. | 有 | common |
| [scripts/validate_rust_tokenizer.py](../../scripts/validate_rust_tokenizer.py) | Exercise character offsets, padding, truncation and batch remainders. | 有 | cache_tokens, common |
| [scripts/verify_data.py](../../scripts/verify_data.py) | Verify frozen artifacts before training without changing any files. | 有 | common, prepare_data |
| [tests/test_data.py](../../tests/test_data.py) | Guard against split leakage and accidental preprocessing damage. | — | common, prepare_data |

## 早期模型与课程

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/analyze_baseline_errors.py](../../scripts/analyze_baseline_errors.py) | Validation-only length strata and errors from the selected baseline. | 有 | cache_tokens, common, training_utils |
| [scripts/benchmark_day2.py](../../scripts/benchmark_day2.py) | Fresh-process full-finetuning probes on real train reviews, fixed padded shapes. | 有 | common, day2_data, day2_runtime, modernbert_mlx |
| [scripts/day2_runtime.py](../../scripts/day2_runtime.py) | Set precision policy before any MLX import (including transitive imports). | — |  |
| [scripts/pilot_day2.py](../../scripts/pilot_day2.py) | Short real IMDB full-encoder finetuning; validation is diagnostic only. | 有 | benchmark_day2, common, day2_data, day2_runtime, modernbert_mlx |
| [scripts/summarize_day1.py](../../scripts/summarize_day1.py) | Build the readable Day1 readout from actual JSON results. | 有 | common |
| [scripts/summarize_day2.py](../../scripts/summarize_day2.py) | Generate the Day2 readout from measured parity, training and benchmark results. | 有 | common |
| [scripts/summarize_day3.py](../../scripts/summarize_day3.py) | Generate a source-backed stage report after full baseline completion. | 有 | common |
| [scripts/train_baseline.py](../../scripts/train_baseline.py) | Full train/validation baseline with warmup, early stopping and exact resume. | 有 | benchmark_day2, checkpointing, common, day2_data, day2_runtime, modernbert_mlx, training_utils |
| [scripts/validate_modernbert.py](../../scripts/validate_modernbert.py) | Compare MLX with a pinned Transformers CPU float32 reference. | 有 | common, day2_data, day2_runtime, modernbert_mlx |

## 融合、评估与拒答

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/bow_nb_branch.py](../../scripts/bow_nb_branch.py) | Train-only sparse sentiment features and calibration-only convex fusion. | — | common, comparison_metrics |
| [scripts/evaluate_laya_frozen_test.py](../../scripts/evaluate_laya_frozen_test.py) | One-shot official-test evaluation of the already selected, frozen Laya candidate. | 有 | common, export_laya_posthoc, laya_decision_views, laya_fusion_inference, laya_mlx_precise_inference, run_laya_probability_fusion, train_laya, validate_laya_fusion_inference |
| [scripts/laya_fusion_inference.py](../../scripts/laya_fusion_inference.py) | Real label-free FP32 inference with a previously frozen probability mixture. | — | common, export_laya_posthoc, laya_probability_fusion |
| [scripts/laya_mlx_fusion_inference.py](../../scripts/laya_mlx_fusion_inference.py) | MLX FP16 execution of the frozen fusion; shared, byte-identical CPU input preparation. | — | export_laya_posthoc, laya_fusion_inference |
| [scripts/laya_probability_fusion.py](../../scripts/laya_probability_fusion.py) | One calibration-only mixing coefficient; no model, data or network access. | — | laya_decision_views |
| [scripts/predict_laya_fusion.py](../../scripts/predict_laya_fusion.py) | Offline real-model inference with the frozen two-model Laya fusion. | 有 | common, laya_fusion_inference, laya_mlx_fusion_inference, laya_mlx_precise_inference, train_laya, validate_laya_fusion_inference, validate_laya_mlx_fusion, validate_laya_mlx_precise |
| [scripts/run_bow_nb_study.py](../../scripts/run_bow_nb_study.py) | Two-process CPU study; no official-test access or development-time fitting. | — | bow_nb_branch, common, comparison_metrics, laya_probability_fusion |
| [scripts/run_final_cpu_alternatives.py](../../scripts/run_final_cpu_alternatives.py) | Bounded four-candidate CPU study; frozen feature heads and held-out fusion. | — | bow_nb_branch, common, comparison_metrics, run_bow_nb_study |
| [scripts/run_final_cpu_alternatives_v2.py](../../scripts/run_final_cpu_alternatives_v2.py) | Bounded four-candidate CPU study; frozen feature heads and held-out fusion. | — | bow_nb_branch, common, comparison_metrics, run_bow_nb_study |
| [scripts/run_laya_probability_fusion.py](../../scripts/run_laya_probability_fusion.py) | Replay fixed cached views: fit on calibration, freeze, then evaluate development. | 有 | common, laya_decision_views, laya_probability_fusion |
| [scripts/run_laya_risk_coverage.py](../../scripts/run_laya_risk_coverage.py) | Describe frozen development predictions; no fitting, inference or test access. | 有 | laya_probability_fusion |
| [scripts/validate_laya_fusion_inference.py](../../scripts/validate_laya_fusion_inference.py) | Verify frozen fusion against real FP32 forwards and measure inference cost. | 有 | common, export_laya_posthoc, laya_decision_views, laya_fusion_inference, run_laya_probability_fusion, train_laya |
| [scripts/validate_laya_mlx_fusion.py](../../scripts/validate_laya_mlx_fusion.py) | Frozen Mac MLX FP16 fusion validation; no fitting, training or official test access. | 有 | common, export_laya_posthoc, laya_decision_views, laya_fusion_inference, laya_mlx_fusion_inference, run_laya_probability_fusion, train_laya, validate_laya_fusion_inference |
| [scripts/verify_bow_nb_study.py](../../scripts/verify_bow_nb_study.py) | Independent saved-output checks; never fit or access official-test files. | 有 | common, run_bow_nb_study |
| [scripts/verify_final_cpu_alternatives.py](../../scripts/verify_final_cpu_alternatives.py) | Independent optimization and saved-metric verification of the bounded study. | 有 | common, run_final_cpu_alternatives_v2 |
| [tests/test_bow_nb_branch.py](../../tests/test_bow_nb_branch.py) | Scientific-boundary and sparse-transform checks on synthetic examples. | — | bow_nb_branch |
| [tests/test_final_cpu_alternatives.py](../../tests/test_final_cpu_alternatives.py) | 无模块说明；按文件名与调用关系定位。 | — | run_final_cpu_alternatives |
| [tests/test_final_cpu_alternatives_v2.py](../../tests/test_final_cpu_alternatives_v2.py) | 无模块说明；按文件名与调用关系定位。 | — | run_final_cpu_alternatives_v2 |
| [tests/test_laya_frozen_test.py](../../tests/test_laya_frozen_test.py) | 无模块说明；按文件名与调用关系定位。 | — | common, evaluate_laya_frozen_test |
| [tests/test_laya_fusion_inference.py](../../tests/test_laya_fusion_inference.py) | 无模块说明；按文件名与调用关系定位。 | — | common, export_laya_posthoc, laya_fusion_inference, laya_probability_fusion, validate_laya_fusion_inference |
| [tests/test_laya_mlx_fusion_protocol.py](../../tests/test_laya_mlx_fusion_protocol.py) | Reject accidental precision/protocol drift before any GPU execution. | — | common, validate_laya_fusion_inference, validate_laya_mlx_fusion |
| [tests/test_laya_probability_fusion.py](../../tests/test_laya_probability_fusion.py) | 无模块说明；按文件名与调用关系定位。 | — | common, laya_decision_views, laya_probability_fusion, run_laya_probability_fusion |

## 训练基础设施

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/audit_cuda_training.py](../../scripts/audit_cuda_training.py) | Observe a CUDA smoke run without changing its loss, optimizer or scheduler. | 有 | common, train_laya |
| [scripts/average_laya_checkpoints.py](../../scripts/average_laya_checkpoints.py) | N8 fixed 50:50 parameter average; calibration and evaluation remain separate. | 有 | common |
| [scripts/benchmark_laya_cuda.py](../../scripts/benchmark_laya_cuda.py) | Offline FP32 full-length training-step timing; no research checkpoint is saved. | 有 | common, train_laya |
| [scripts/check_cuda_environment.py](../../scripts/check_cuda_environment.py) | Read-only checks of frozen local inputs and CUDA availability; never trains. | 有 | common, train_laya |
| [scripts/checkpointing.py](../../scripts/checkpointing.py) | Atomic resumable generations; pointer switches only after all files exist. | — | common, day2_runtime |
| [scripts/export_cuda_laya_predictions.py](../../scripts/export_cuda_laya_predictions.py) | Evaluate saved CUDA Laya weights on the frozen full validation split, offline. | 有 | common, comparison_metrics, train_laya |
| [scripts/laya_checkpoint_delivery.py](../../scripts/laya_checkpoint_delivery.py) | Immutable bundles, verified resumable copies and durable epoch acknowledgments. | 有 | laya_head_control, laya_robust_training |
| [scripts/laya_epoch_checkpoints.py](../../scripts/laya_epoch_checkpoints.py) | Opt-in snapshots of upstream's existing epoch save; no extra model forwards. | — | common |
| [scripts/laya_modal_deferred.py](../../scripts/laya_modal_deferred.py) | User-authorized deferred local delivery; cloud receipts never claim local backup. | 有 | common, laya_checkpoint_delivery, run_laya_robust_loss, run_laya_robust_study |
| [scripts/laya_modal_delivery.py](../../scripts/laya_modal_delivery.py) | Copy committed Modal Volume files locally, independently load, then acknowledge. | — | laya_checkpoint_delivery, laya_modal_runtime |
| [scripts/laya_modal_recovery.py](../../scripts/laya_modal_recovery.py) | Crash recovery for the unchanged deferred-v2 scientific experiment. | — | laya_checkpoint_delivery, laya_modal_deferred |
| [scripts/laya_modal_runtime.py](../../scripts/laya_modal_runtime.py) | Modal transport adapter; the frozen SSH protocol and scientific loop stay intact. | 有 | common, laya_checkpoint_delivery, run_laya_robust_loss, run_laya_robust_study |
| [scripts/modal_laya_deferred.py](../../scripts/modal_laya_deferred.py) | Detached cloud coordinator. Local delivery is explicitly deferred by the user. | 有 | common, laya_checkpoint_delivery, laya_modal_deferred |
| [scripts/modal_laya_evaluation_completion.py](../../scripts/modal_laya_evaluation_completion.py) | Complete five fixed CUDA evaluations; no admission, optimizer, or model selection. | 有 | laya_checkpoint_delivery, laya_evaluation_completion, laya_modal_deferred, laya_modal_recovery, modal_laya_recovery |
| [scripts/modal_laya_recovery.py](../../scripts/modal_laya_recovery.py) | Resume the original six-model study with durable orchestration and deadlines. | 有 | common, laya_checkpoint_delivery, laya_modal_deferred, laya_modal_recovery |
| [scripts/modal_laya_train.py](../../scripts/modal_laya_train.py) | Single-GPU Modal study with persistent checkpoints and synchronous local delivery. | 有 | common, laya_checkpoint_delivery, laya_modal_delivery, laya_modal_runtime |
| [scripts/plot_cuda_benchmark.py](../../scripts/plot_cuda_benchmark.py) | Render measured CUDA FP32 step throughput and allocated memory for the report. | 有 | common |
| [scripts/summarize_cuda_reference.py](../../scripts/summarize_cuda_reference.py) | Compare the completed CUDA reference to frozen historical CE predictions. | 有 | common, comparison_metrics, train_laya |
| [scripts/train_laya.py](../../scripts/train_laya.py) | Offline IMDB adapter for the pinned upstream Laya trainer (PyTorch/MPS/CUDA). | 有 | common, cuda_numerics, laya_decision_views, laya_epoch_checkpoints, laya_optimization, laya_resampling, laya_runtime, laya_semantic_training, laya_window_training |
| [scripts/training_utils.py](../../scripts/training_utils.py) | Deterministic scheduling, metrics and checkpoint ranking. | — |  |
| [tests/test_cuda_audit_indices.py](../../tests/test_cuda_audit_indices.py) | Regression for integer sampling beyond the exact range of float32 indices. | — | audit_cuda_training |
| [tests/test_laya_checkpoint_delivery.py](../../tests/test_laya_checkpoint_delivery.py) | 无模块说明；按文件名与调用关系定位。 | — | laya_checkpoint_delivery, laya_head_control, laya_robust_training, run_laya_robust_loss |
| [tests/test_laya_epoch_checkpoints.py](../../tests/test_laya_epoch_checkpoints.py) | 无模块说明；按文件名与调用关系定位。 | — | laya_epoch_checkpoints |
| [tests/test_laya_modal_deferred.py](../../tests/test_laya_modal_deferred.py) | 无模块说明；按文件名与调用关系定位。 | — | laya_checkpoint_delivery, laya_modal_deferred |
| [tests/test_laya_modal_delivery.py](../../tests/test_laya_modal_delivery.py) | Local byte/strict-load tests of the Modal adapter; not cloud admission. | — | laya_checkpoint_delivery, laya_modal_delivery, laya_modal_runtime |
| [tests/test_laya_modal_recovery.py](../../tests/test_laya_modal_recovery.py) | Fault injection for orchestration, not a substitute for CUDA admission. | — | laya_checkpoint_delivery, laya_modal_recovery |
| [tests/test_training.py](../../tests/test_training.py) | 无模块说明；按文件名与调用关系定位。 | — | training_utils |

## 训练数据与语义

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/analyze_laya_train_candidates.py](../../scripts/analyze_laya_train_candidates.py) | Join training-only model diagnostics with frozen keyword candidates; prepare review. | 有 | common, export_laya_train_candidates |
| [scripts/audit_semantic_pair_duplicates.py](../../scripts/audit_semantic_pair_duplicates.py) | Exact and lexical five-word-shingle overlap audit; official test is never read. | 有 | common, semantic_pair_pilot |
| [scripts/export_laya_train_candidates.py](../../scripts/export_laya_train_candidates.py) | Offline fixed-weight inference on frozen gradient-training IDs only. | 有 | common, export_laya_posthoc, train_laya |
| [scripts/laya_resampling.py](../../scripts/laya_resampling.py) | Audited one-expression adaptation of the pinned epoch input list; no loop rewrite. | — | common |
| [scripts/laya_reviewed_data_study.py](../../scripts/laya_reviewed_data_study.py) | Isolated adapter: reuse frozen CE epochs, change only the reviewed dataset. | 有 | analyze_laya_length_router, common, comparison_metrics, laya_head_control, laya_long_context_pilot, run_laya_head_control, train_laya |
| [scripts/laya_semantic_training.py](../../scripts/laya_semantic_training.py) | Review-gated semantic epoch inputs with audited actual upstream consumption. | — | common, laya_resampling, semantic_pair_pilot |
| [scripts/modal_laya_reviewed_data.py](../../scripts/modal_laya_reviewed_data.py) | One reviewed-data model; frozen CE implementation, durable delivery, $4 cap. | 有 | common, laya_checkpoint_delivery, laya_reviewed_data_study, modal_laya_recovery |
| [scripts/prepare_laya_semantic_training.py](../../scripts/prepare_laya_semantic_training.py) | Bind a human-reviewed semantic pack to strict A/B/C configs and exposure contracts. | 有 | common, laya_semantic_training, semantic_pair_pilot |
| [scripts/probe_semantic_pairs.py](../../scripts/probe_semantic_pairs.py) | Fixed-weight MLX sensitivity probe on AI-drafted seen-training triplets only. | 有 | common, semantic_pair_pilot |
| [scripts/receive_laya_reviewed_data.py](../../scripts/receive_laya_reviewed_data.py) | CPU-only background delivery; all Modal operations use one native async loop. | 有 | common, laya_checkpoint_delivery, laya_reviewed_data_study |
| [scripts/run_laya_semantic_abc.py](../../scripts/run_laya_semantic_abc.py) | Bounded reviewed 30-source exploratory A/B/C; frozen official test remains sealed. | 有 | common, comparison_metrics |
| [scripts/semantic_pair_pilot.py](../../scripts/semantic_pair_pilot.py) | Build auditable, review-gated semantic triplets from frozen gradient sources. | 有 | common |
| [scripts/verify_semantic_pairs.py](../../scripts/verify_semantic_pairs.py) | Independently verify saved MLX inputs and full A/B/C data budgets on CPU. | 有 | common, semantic_pair_pilot |
| [tests/test_laya_resampling.py](../../tests/test_laya_resampling.py) | Exposure budget, source contract and numerical control-path regressions. | — | common, cuda_numerics, laya_optimization, laya_resampling |
| [tests/test_laya_reviewed_data_study.py](../../tests/test_laya_reviewed_data_study.py) | Verify dataset adaptation, module isolation and unchanged CE configuration. | — | common, laya_context_pair, laya_reviewed_data_study |
| [tests/test_laya_semantic_training.py](../../tests/test_laya_semantic_training.py) | Real upstream loop and tokenizer, synthetic reviews; optional CUDA mechanical check. | — | common, cuda_numerics, laya_optimization, laya_runtime, laya_semantic_training, semantic_pair_pilot, train_laya |
| [tests/test_laya_train_candidates.py](../../tests/test_laya_train_candidates.py) | 无模块说明；按文件名与调用关系定位。 | — | export_laya_train_candidates |
| [tests/test_semantic_pair_pilot.py](../../tests/test_semantic_pair_pilot.py) | 无模块说明；按文件名与调用关系定位。 | — | probe_semantic_pairs, semantic_pair_pilot |

## 长文与决策策略

| 文件 | 说明 | CLI | 本地依赖 |
|---|---|---|---|
| [scripts/analyze_laya_length_router.py](../../scripts/analyze_laya_length_router.py) | Reproduce a fixed length router from sealed predictions; no models or GPU. | 有 | common, comparison_metrics, laya_length_router |
| [scripts/analyze_laya_long_fusion_bc.py](../../scripts/analyze_laya_long_fusion_bc.py) | Evaluate only the declared B/C mixtures, using cached development probabilities. | 有 | analyze_laya_length_router, common, comparison_metrics, laya_decision_views, laya_probability_fusion |
| [scripts/export_laya_posthoc.py](../../scripts/export_laya_posthoc.py) | Export fixed-weight head/head-tail predictions on calibration and development only. | 有 | common, train_laya |
| [scripts/laya_context_pair.py](../../scripts/laya_context_pair.py) | Frozen paired context experiment, epoch bundles, admission and gated evaluation. | 有 | common, comparison_metrics, laya_checkpoint_delivery, laya_context_training, laya_decision_views, laya_head_control, laya_long_context_pilot, laya_robust_training, run_laya_head_control, train_laya |
| [scripts/laya_context_training.py](../../scripts/laya_context_training.py) | Length-aware CE epochs with exact optimizer/RNG continuation and input journals. | — | laya_head_control, laya_robust_training |
| [scripts/laya_decision_views.py](../../scripts/laya_decision_views.py) | Fixed decision-view algorithms; no fitting or selection on development labels. | — | comparison_metrics |
| [scripts/laya_length_router.py](../../scripts/laya_length_router.py) | Fixed, label-free dispatch to the frozen 512/2048 context models. | — |  |
| [scripts/laya_long_context_pilot.py](../../scripts/laya_long_context_pilot.py) | Bounded training-only 512/2048 admission; never an accuracy experiment. | — | common, laya_decision_views, laya_head_control, laya_robust_training, laya_runtime, train_laya |
| [scripts/laya_posthoc.py](../../scripts/laya_posthoc.py) | Fit fixed post-hoc calibration on original calibration IDs; evaluate frozen validation. | 有 | common, comparison_metrics |
| [scripts/laya_window_training.py](../../scripts/laya_window_training.py) | Gradient-only window sampling; retain upstream order, losses and RNG streams. | — | common, laya_decision_views, laya_resampling |
| [scripts/modal_laya_context_pair.py](../../scripts/modal_laya_context_pair.py) | Single GPU queue, absolute spend deadline, durable epoch acceptance. | 有 | common, laya_checkpoint_delivery, laya_context_pair, modal_laya_recovery |
| [scripts/modal_laya_long_context_pilot.py](../../scripts/modal_laya_long_context_pilot.py) | One bounded GPU pilot with persistent files and mandatory local weight delivery. | 有 | common, laya_checkpoint_delivery, laya_long_context_pilot, laya_modal_delivery, modal_laya_recovery |
| [scripts/prepare_laya_window_evaluation.py](../../scripts/prepare_laya_window_evaluation.py) | Bind a completed EXP-3 pair to new fixed-weight study IDs and actual hashes. | 有 | common, run_laya_decision_study |
| [scripts/receive_laya_context_pair.py](../../scripts/receive_laya_context_pair.py) | CPU-only background delivery; all Modal operations use one native async loop. | 有 | common, laya_checkpoint_delivery, laya_context_pair |
| [scripts/run_laya_checkpoint_average.py](../../scripts/run_laya_checkpoint_average.py) | N8: fixed epoch 1/2 average, calibration-only temperature, paired head-tail study. | 有 | average_laya_checkpoints, common, comparison_metrics, export_laya_posthoc, laya_decision_views, laya_posthoc, laya_runtime, run_laya_decision_study, train_laya |
| [scripts/run_laya_decision_pipeline.py](../../scripts/run_laya_decision_pipeline.py) | Bounded sequential study -> evidence gate -> strict window pair -> final study. | 有 | common, prepare_laya_window_evaluation, run_laya_decision_study |
| [scripts/run_laya_decision_study.py](../../scripts/run_laya_decision_study.py) | Predeclared EXP-1/2 and N1 zero-training study, calibration first, test sealed. | 有 | common, export_laya_posthoc, laya_decision_views, laya_runtime, train_laya |
| [tests/test_laya_checkpoint_average.py](../../tests/test_laya_checkpoint_average.py) | 无模块说明；按文件名与调用关系定位。 | — | average_laya_checkpoints, common, comparison_metrics, run_laya_checkpoint_average |
| [tests/test_laya_context_pair.py](../../tests/test_laya_context_pair.py) | Scientific invariance, length retention and genuine epoch recovery tests. | — | common, laya_context_pair, laya_context_training, laya_head_control |
| [tests/test_laya_decision_pipeline.py](../../tests/test_laya_decision_pipeline.py) | 无模块说明；按文件名与调用关系定位。 | — | common, run_laya_decision_pipeline |
| [tests/test_laya_decision_views.py](../../tests/test_laya_decision_views.py) | Algorithm checks use synthetic inputs; no real validation or model inference. | — | average_laya_checkpoints, common, cuda_numerics, laya_decision_views, laya_optimization, laya_window_training, prepare_laya_window_evaluation, run_laya_decision_study, train_laya |
| [tests/test_laya_length_router.py](../../tests/test_laya_length_router.py) | Check actual dispatch, no duplicate inference, order and length semantics. | — | laya_length_router |
| [tests/test_laya_long_context_pilot.py](../../tests/test_laya_long_context_pilot.py) | Check token retention beyond the historical 512-token pre-clamp. | — | common, laya_decision_views, laya_long_context_pilot |
| [tests/test_laya_posthoc.py](../../tests/test_laya_posthoc.py) | 无模块说明；按文件名与调用关系定位。 | — | common, laya_posthoc |
| [tests/test_laya_posthoc_inputs.py](../../tests/test_laya_posthoc_inputs.py) | 无模块说明；按文件名与调用关系定位。 | — | common, export_laya_posthoc |
