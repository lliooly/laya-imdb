# 测试与独立验证

从仓库根目录运行：

```bash
uv run --locked --offline --no-sync python scripts/build_closeout_catalog.py test
```

按模块隔离子进程，避免永久 audit hook 相互影响。test_bow_nb_branch 与 test_laya_deberta_calibration_weight 使用 environments/bow-cpu，其余使用主项目锁；首次需安装相应环境。

本轮 48 个模块、234 项测试通过。新增归档工具修改后另重跑 4 项自身测试。日志在 reports/project-closeout-20261018-v1/。这不表示重新训练或重新运行正式测试集，也不替代原 GPU 准入、模型交付与统计核验。
