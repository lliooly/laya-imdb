"""Build the readable Day1 readout from actual JSON results."""
import json

from common import ROOT


def main():
    data = json.loads((ROOT / "reports/data_audit.json").read_text())
    env = json.loads((ROOT / "reports/environment.json").read_text())
    tokens = json.loads((ROOT / "reports/tokenizer_audit.json").read_text())
    lines = ["# Day1 实际结果", "", "日期：2026-10-09。由 `scripts/summarize_day1.py` 根据实际运行结果生成。", "",
        "## 环境", "", f"- {env['hardware']['chip_type']}，{env['hardware']['physical_memory']}；Python {env['python']}。",
        f"- MLX {env['packages']['mlx']}，Tokenizers {env['packages']['tokenizers']}，Metal GPU 验证：{env['status']}。",
        f"- 合成任务20次参数更新：loss {env['initial_loss']:.6f} → {env['final_loss']:.6f}；保存重载最大绝对误差 {env['save_reload_max_abs_error']}。",
        "- 此验证不含预训练编码器，不代表 IMDB 准确率，也不能估算正式训练时间。", "",
        "## 数据", "", "| 划分 | 样本 | 负面 | 正面 | 额外重复行 |", "|---|---:|---:|---:|---:|"]
    for split, s in data["splits"].items():
        lines.append(f"| {split} | {s['count']} | {s['labels']['0']} | {s['labels']['1']} | {s['duplicate_extra_rows']} |")
    lines += ["", "seed=42，按预处理后文本哈希分组，再按类别选取验证样本。训练/验证相同文本交集为0。重复行保留在各自组内。",
        f"官方训练/测试存在 **{data['official_train_test_overlap_unique_texts']} 组**相同预处理文本；保留官方划分并披露，不将其描述为完全无重复的测试集。",
        "空文本、UTF-8、标签/评分范围与唯一ID检查通过。长度统计中的词数是空白分词数，不是模型token数。",
        "归档SHA-256（本地记录，非官方发布的校验值）：", "", f"`{data['archive_sha256']}`", "",
        "## 原配 tokenizer 实测", "", "只读取训练和验证划分；包含特殊token，关闭padding和truncation。", "",
        "| 分词器 | 划分 | token均值 | p95 | >256 | >512 | >1024 | 评论/秒 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for model, result in tokens["models"].items():
        for split, s in result["splits"].items():
            lengths = s["tokens_including_specials"]
            rates = s["truncated_fraction"]
            lines.append(f"| {model} | {split} | {lengths['mean']:.1f} | {lengths['p95']} | {rates['256']:.2%} | {rates['512']:.2%} | {rates['1024']:.2%} | {s['reviews_per_second']:.0f} |")
    lines += ["", "吞吐量是单遍探索性测量，已排除下载、读盘和预热，仅计encode_batch。不同分词器产生不同序列，不能据此声称模型端到端更快。",
        "BERT结构限制下，1024列仅作文本长度观察，不意味着可以用1024输入训练。",
        "tokenizer版本和文件哈希见 `configs/tokenizer_lock.json`。", "",
        "## 阶段判断与 Day2", "", "Day1 的环境、数据、原配分词参照与实验协议已建立；未训练 IMDB 分类模型。",
        "Day2 优先完成预训练权重加载、参考实现前向对齐、小批量反向与真实长度/batch测速。",
        "从256/512两档开始，按验证表现和计算预算选择；ModernBERT的1024档留到基础适配通过后。",
        "在获得真实编码器训练耗时前，不作是否需要租服务器的结论。", ""]
    (ROOT / "reports/day1.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote reports/day1.md")


if __name__ == "__main__":
    main()
