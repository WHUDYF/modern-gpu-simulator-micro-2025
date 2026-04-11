# 组会幻灯片使用说明

这套材料同时提供两种格式：

- `ai-group-meeting-2026-04-05.tex`
  - `LaTeX Beamer` 版本
  - 适合已经有 TeX 环境的机器
- `ai-group-meeting-2026-04-05.marp.md`
  - `Marp Markdown` 版本
  - 适合快速预览和导出 HTML / PDF / PPTX

## 1. Beamer 版本

文件：

- `ai-group-meeting-2026-04-05.tex`

如果机器上已经安装了 TeX Live，可执行：

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/docs/slides
pdflatex ai-group-meeting-2026-04-05.tex
pdflatex ai-group-meeting-2026-04-05.tex
```

输出文件通常是：

- `ai-group-meeting-2026-04-05.pdf`

## 2. Marp 版本

文件：

- `ai-group-meeting-2026-04-05.marp.md`

如果机器上有 `marp-cli`，可执行：

```bash
cd /home/dyf/modern-gpu-simulator-micro-2025/docs/slides
marp ai-group-meeting-2026-04-05.marp.md --pdf
marp ai-group-meeting-2026-04-05.marp.md --pptx
```

如果没有 `marp-cli`，可先安装：

```bash
npm install -g @marp-team/marp-cli
```

## 3. 两种格式怎么选

- 如果你只是想最快导出成 `PDF` 或 `PPTX`，优先用 `Marp`
- 如果你更熟悉学术汇报的排版，或者后面还想细调样式，用 `Beamer`

## 4. 当前页数

这两份稿件都按完整组会版本组织，约 `8` 页：

1. 标题页
2. 背景与问题重定义
3. 研究主线
4. 当前采用的压缩手段
5. 当前已完成工作与测试结果
6. 第一轮 AI trace 实验设计
7. 4 个月计划与目标
8. 总结页
