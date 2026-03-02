---
name: pdf-to-md-docling
description: Docling（docling）を使ってPDFを精度重視でMarkdownに変換する。論文PDF（docs/ref/*.pdf など）をPDFの隣に .md として生成し、LLM/Codexが参照しやすい形に整える時に使う。
---

# PDF → Markdown (Docling)

## Quick start

```bash
uvx --torch-backend cpu --from docling==2.76.0 \
  python .agents/skills/pdf-to-md-docling/scripts/convert_pdf_to_md.py \
  docs/ref/1994-benter.pdf docs/ref/2406.16507v3.pdf
```

生成物:
- `docs/ref/1994-benter.md`
- `docs/ref/2406.16507v3.md`

## Notes
- 依存はプロジェクトに追加せず、`uvx` の隔離環境で実行する（初回はモデル/依存のDLで時間がかかる）。
- `--torch-backend cpu` を指定してCUDA依存を避ける。
- 画像は埋め込まず、Markdown上では `<!-- image -->` のプレースホルダ運用。
- ページ境界は `<!-- page break -->` を挿入して残す。

