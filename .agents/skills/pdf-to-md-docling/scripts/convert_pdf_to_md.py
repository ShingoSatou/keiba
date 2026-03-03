from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    if not text.endswith("\n"):
        text += "\n"
    return text


def _build_meta_header(
    *,
    source: Path,
    sha256: str,
    converted_at: dt.datetime,
    docling_version: str,
    docling_core_version: str,
    options: dict[str, object],
) -> str:
    lines: list[str] = ["<!--"]
    lines.append(f"source: {source.as_posix()}")
    lines.append(f"sha256: {sha256}")
    lines.append(f"converted_at: {converted_at.isoformat()}")
    lines.append(f"docling: {docling_version}")
    lines.append(f"docling_core: {docling_core_version}")
    lines.append("options:")
    for k in sorted(options.keys()):
        v = options[k]
        lines.append(f"  {k}: {v}")
    lines.append("-->")
    return "\n".join(lines) + "\n\n"


def _convert_one_pdf(pdf_path: Path) -> Path:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.document import DoclingVersion
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import ImageRefMode

    pdf_path = pdf_path.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_path = pdf_path.with_suffix(".md")

    # CPU-only, OCR enabled (bitmap-only, not forced), table structure enabled.
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
        do_ocr=True,
        do_table_structure=True,
    )
    pipeline_options.ocr_options.force_full_page_ocr = False
    pipeline_options.enable_remote_services = False
    pipeline_options.allow_external_plugins = False
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        },
    )

    conv_res = converter.convert(pdf_path, raises_on_error=False)
    if conv_res.status not in {ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS}:
        errs = "; ".join(e.error_message for e in (conv_res.errors or []))
        raise RuntimeError(f"Docling conversion failed: {pdf_path} ({conv_res.status}). {errs}".strip())

    if conv_res.status == ConversionStatus.PARTIAL_SUCCESS and conv_res.errors:
        print(
            f"Warning: partial success converting {pdf_path}: "
            + "; ".join(e.error_message for e in conv_res.errors),
            file=sys.stderr,
        )

    doc = conv_res.document
    if doc is None:
        raise RuntimeError(f"Docling produced no document for: {pdf_path}")

    page_break_placeholder = "<!-- page break -->"
    md = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        page_break_placeholder=page_break_placeholder,
    )

    if "data:image/" in md:
        raise RuntimeError(
            "Unexpected embedded image data (data:image/...). "
            "This script is configured to use placeholders only."
        )

    versions = DoclingVersion()
    converted_at = dt.datetime.now().astimezone()
    meta_header = _build_meta_header(
        source=pdf_path.relative_to(Path.cwd()) if pdf_path.is_relative_to(Path.cwd()) else pdf_path,
        sha256=_sha256_file(pdf_path),
        converted_at=converted_at,
        docling_version=str(versions.docling_version),
        docling_core_version=str(versions.docling_core_version),
        options={
            "accelerator_device": "cpu",
            "ocr": True,
            "force_full_page_ocr": False,
            "table_structure": True,
            "image_mode": "placeholder",
            "page_break_placeholder": page_break_placeholder,
        },
    )

    output_text = _normalize_markdown(meta_header + md)
    output_path.write_text(output_text, encoding="utf-8", newline="\n")
    return output_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert one or more PDFs to Markdown using Docling.")
    parser.add_argument("pdf", nargs="+", type=Path, help="Input PDF path(s).")
    args = parser.parse_args(argv)

    failures: list[str] = []
    for pdf in args.pdf:
        try:
            out = _convert_one_pdf(pdf)
            print(f"Wrote: {out}")
        except Exception as e:
            failures.append(f"{pdf}: {e}")

    if failures:
        for msg in failures:
            print(f"Error: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

