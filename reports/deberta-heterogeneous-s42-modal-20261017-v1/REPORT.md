# DeBERTa v1：GPU 启动前的镜像构建失败

本次构建在加载镜像安装函数时发生 `ModuleNotFoundError: modal_deberta_study`。本地入口未执行，尚无 submission，未启动 GPU、未训练、未访问开发或官方测试。

原冻结配置与源码副本保存在 `preparation-snapshot/`，保留失败证据，不将其写成成功准入。原本机空等接收器已停止。

修复：将镜像安装函数移到可独立导入的纯标准库模块，并在构建前复制该模块及设置 PYTHONPATH。新运行编号为 `deberta-heterogeneous-s42-modal-20261017-v2`；本轮仍使用一次 5 美元总预算，v1 预留转入 v2，构建失败 CPU 余量也包含其中，不重复预留 5 美元。
