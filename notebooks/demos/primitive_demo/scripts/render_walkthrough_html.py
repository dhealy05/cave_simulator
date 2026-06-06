"""Render ``walkthrough.md`` as a standalone local HTML page.

The page deliberately stays close to the Markdown source. It only adds enough
structure and CSS to make the checked-in Cave Walker animation the lead visual.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


DEMO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = DEMO_ROOT / "docs"
DEFAULT_SOURCE = DOCS_DIR / "walkthrough.md"
DEFAULT_OUTPUT = DOCS_DIR / "walkthrough.html"


def render_walkthrough_html(
    source: str | Path = DEFAULT_SOURCE,
    output: str | Path = DEFAULT_OUTPUT,
) -> str:
    """Render the primitive walkthrough Markdown to HTML and return the path."""

    source = Path(source)
    output = Path(output)
    markdown = source.read_text(encoding="utf-8")
    title = _first_heading(markdown) or "Primitive Demo Walkthrough"
    body = _render_markdown(markdown)
    document = _document(title, body)
    output.write_text(document, encoding="utf-8")
    return str(output)


def _first_heading(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _render_markdown(markdown: str) -> str:
    blocks: list[str] = []
    paragraph: list[str] = []
    code: list[str] = []
    table: list[str] = []
    list_items: list[str] = []
    in_code = False
    code_lang = ""

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            text = " ".join(part.strip() for part in paragraph)
            blocks.append(f"<p>{_inline(text)}</p>")
            paragraph = []

    def flush_table() -> None:
        nonlocal table
        if table:
            blocks.append(_table(table))
            table = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            items = "".join(f"<li>{_inline(item)}</li>" for item in list_items)
            blocks.append(f"<ul>{items}</ul>")
            list_items = []

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            flush_table()
            flush_list()
            if in_code:
                language = f' class="language-{html.escape(code_lang)}"' if code_lang else ""
                blocks.append(f"<pre><code{language}>{html.escape(chr(10).join(code))}</code></pre>")
                code = []
                code_lang = ""
                in_code = False
            else:
                code_lang = line.strip("`").strip()
                in_code = True
            continue

        if in_code:
            code.append(raw)
            continue

        if not line.strip():
            flush_paragraph()
            flush_table()
            flush_list()
            continue

        if line.startswith("#"):
            flush_paragraph()
            flush_table()
            flush_list()
            level = min(len(line) - len(line.lstrip("#")), 3)
            text = line[level:].strip()
            blocks.append(f"<h{level}>{_inline(text)}</h{level}>")
            continue

        if _image_match(line):
            flush_paragraph()
            flush_table()
            flush_list()
            alt, src = _image_match(line).groups()
            blocks.append(_figure(alt, src))
            continue

        if _is_table_line(line):
            flush_paragraph()
            flush_list()
            table.append(line)
            continue

        if line.startswith("- "):
            flush_paragraph()
            flush_table()
            list_items.append(line[2:].strip())
            continue

        flush_table()
        flush_list()
        paragraph.append(line)

    flush_paragraph()
    flush_table()
    flush_list()
    return "\n".join(blocks)


def _image_match(line: str) -> re.Match[str] | None:
    return re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _table(lines: list[str]) -> str:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) >= 2 and all(set(cell) <= {"-", ":", " "} for cell in rows[1]):
        head, body = rows[0], rows[2:]
    else:
        head, body = (), rows
    parts = ["<table>"]
    if head:
        cells = "".join(f"<th>{_inline(cell)}</th>" for cell in head)
        parts.append(f"<thead><tr>{cells}</tr></thead>")
    parts.append("<tbody>")
    for row in body:
        cells = "".join(f"<td>{_inline(cell)}</td>" for cell in row)
        parts.append(f"<tr>{cells}</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _figure(alt: str, src: str) -> str:
    klass = "figure money-shot" if src.endswith("cave_walker.gif") else "figure"
    return (
        f'<figure class="{klass}">'
        f'<img src="{html.escape(src, quote=True)}" alt="{html.escape(alt, quote=True)}">'
        f"<figcaption>{_inline(alt)}</figcaption>"
        "</figure>"
    )


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
    return escaped


def _document(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17201b;
      --muted: #66736d;
      --line: #d7ded9;
      --paper: #f8f8f4;
      --panel: #ffffff;
      --code: #eef3ef;
      --accent: #16885f;
      --error: #c93e6d;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-size: 17px;
      line-height: 1.62;
    }}

    .page {{
      width: min(100%, 1120px);
      margin: 0 auto;
      padding: 24px clamp(16px, 3vw, 36px) 56px;
    }}

    .hero {{
      display: grid;
      gap: 18px;
      margin-bottom: 32px;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--line);
    }}

    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 4.6vw, 4.1rem);
      line-height: 1.03;
      letter-spacing: 0;
    }}

    .hero p {{
      max-width: 760px;
      margin: 0;
      color: var(--muted);
      font-size: clamp(1rem, 1.7vw, 1.2rem);
    }}

    article {{
      max-width: 880px;
      margin: 0 auto;
    }}

    h1, h2, h3 {{
      line-height: 1.16;
      letter-spacing: 0;
    }}

    article > h1:first-child {{
      display: none;
    }}

    h2 {{
      margin: 42px 0 12px;
      font-size: clamp(1.45rem, 2vw, 2rem);
    }}

    h3 {{
      margin: 30px 0 10px;
      font-size: 1.2rem;
    }}

    p, ul, table, pre, figure {{
      margin: 16px 0;
    }}

    ul {{
      padding-left: 1.35rem;
    }}

    code {{
      padding: 0.12em 0.32em;
      background: var(--code);
      border: 1px solid var(--line);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.9em;
    }}

    pre {{
      overflow-x: auto;
      padding: 16px;
      background: var(--code);
      border: 1px solid var(--line);
    }}

    pre code {{
      padding: 0;
      border: 0;
      background: transparent;
      font-size: 0.88rem;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
    }}

    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      background: var(--code);
      font-size: 0.83rem;
      text-transform: uppercase;
      color: var(--muted);
      letter-spacing: 0.08em;
    }}

    .figure {{
      width: min(100%, 960px);
      margin: 24px auto;
    }}

    .figure img {{
      display: block;
      width: 100%;
      height: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 12px 34px rgba(23, 32, 27, 0.12);
    }}

    .figure figcaption {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.88rem;
      text-align: center;
    }}

    .money-shot {{
      width: min(100%, 1080px);
      margin: 28px 50% 34px;
      transform: translateX(-50%);
    }}

    .money-shot img {{
      image-rendering: auto;
      border: 2px solid var(--ink);
    }}

    @media (max-width: 620px) {{
      body {{ font-size: 16px; }}
      .page {{ padding-top: 18px; }}
      pre {{ padding: 12px; }}
      th, td {{ padding: 8px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <h1>{html.escape(title)}</h1>
      <p>A direct HTML rendering of <code>walkthrough.md</code>, with the Cave Walker animation left as the central visual.</p>
    </header>
    <article>
{body}
    </article>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(render_walkthrough_html(args.source, args.output))


if __name__ == "__main__":
    main()
