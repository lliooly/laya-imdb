# 后台编排首次兼容性检查

用户已明确将交付方式改为：云端训练完成后，再唤醒助手回传权重和分析；不要求本机逐轮在线。

首次后台 CPU 编排在调用 SDK 不存在的 `Function.from_id` 时自行失败，未启动 GPU 训练。原协议 / SHA、Volume 和本地 `artifacts/laya-modal-deferred-v1-before-handle-fix/` 源快照保留。已核验调用终态，再按最早日志创建至终态核验的完整墙钟时间加 600 秒，保守重算预算上界；没有修改或假称读取实际账单。

后续 v2 直接将已初始化的 Modal Function 句柄作为参数传给云端编排，不再使用该不存在的 API。
