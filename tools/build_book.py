#!/usr/bin/env python
"""把每课 README.md 编译成 HTML 教材，注入 pages/ 对应互动页。

- 源头是各课 README.md（唯一事实来源），本脚本可反复运行（幂等）
- 数学公式: markdown → Arithmatex (\\(\\) / $..$) → 本地 MathJax (assets/mathjax/tex-svg.js)
- 相对链接自动改写为从 pages/ 出发的路径
- 用法: python tools/build_book.py
"""
import markdown
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
LESSONS = [
    ("01_numpy", "01_numpy.html"),
    ("02_gradient_descent", "02_gd.html"),
    ("03_logistic_regression", "03_sigmoid.html"),
    ("04_neural_network", "04_mlp.html"),
    ("05_autograd", "05_autograd.html"),
    ("06_pytorch", "06_pytorch.html"),
    ("07_cnn", "07_cnn.html"),
    ("08_sequence_models", "08_lm.html"),
    ("09_attention", "09_attention.html"),
    ("10_transformer", "10_transformer.html"),
]
START, END = "<!--BOOK-START-->", "<!--BOOK-END-->"

MJ_CONFIG = (
    '<script>window.MathJax = { tex: { inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]], '
    'displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]] }, '
    'svg: { fontCache: "global" }, '
    'options: { skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"] } };</script>\n'
    '<script defer src="../assets/mathjax/tex-svg.js"></script>\n'
)

BOOK_CSS = """
  <style>
  .book{font-size:14px;line-height:1.85}
  .book h2{font-size:16.5px;margin:26px 0 8px;padding-top:10px;border-top:1px dashed var(--line)}
  .book h3{font-size:14.5px;margin:18px 0 6px}
  .book p{margin:9px 0}
  .book code{background:#eef1f6;border-radius:5px;padding:1px 6px;font-size:12.5px}
  .book pre{background:#0d1220;color:#d8e0f0;border-radius:10px;padding:13px 16px;overflow:auto;
            font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.7}
  .book pre code{background:none;padding:0;color:inherit}
  .book table{border-collapse:collapse;font-size:13px;width:100%;margin:10px 0}
  .book td,.book th{border:1px solid var(--line);padding:6px 10px;text-align:left}
  .book th{background:#f4f7fc}
  .book ul,.book ol{line-height:1.8;padding-left:22px}
  .book blockquote{border-left:4px solid var(--orange);background:#fff8ef;margin:10px 0;
                   padding:8px 14px;border-radius:0 8px 8px 0}
  .book a{color:var(--blue)}
  .book mjx-container{overflow-x:auto;overflow-y:hidden;max-width:100%}
  .gen{font-size:11.5px;color:var(--muted);font-weight:400;margin-left:8px}
  </style>
"""


def convert(md_text: str, lesson_dir: str) -> str:
    html = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "pymdownx.arithmatex"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    # 去掉第一个 h1 (页面已有标题), 其余标题降一级
    html = re.sub(r"<h1>.*?</h1>\s*", "", html, count=1, flags=re.S)
    for n in (5, 4, 3, 2):
        html = html.replace(f"<h{n}>", f"<h{n+1}>").replace(f"</h{n}>", f"</h{n+1}>")
    # 相对链接改写: 从 pages/ 出发需要 ../<lesson_dir>/ 前缀
    def fix(m):
        attr, url = m.group(1), m.group(2)
        if (url.startswith(("http://", "https://", "#", "mailto:", "../", "pages/"))
                or "://" in url):
            return m.group(0)
        return f'{attr}="../{lesson_dir}/{url}"'
    html = re.sub(r'(href|src)="([^"]+)"', fix, html)
    return html


def build() -> None:
    for lesson_dir, page_name in LESSONS:
        md_path = ROOT / lesson_dir / "README.md"
        page_path = ROOT / "pages" / page_name
        body = convert(md_path.read_text(encoding="utf-8"), lesson_dir)
        block = (f"{START}\n"
                 f'<div class="card" id="book">{BOOK_CSS}\n'
                 f'<h2>📖 教材全文<span class="gen">由 {lesson_dir}/README.md 自动生成 — '
                 f'修改后运行 <code>python tools/build_book.py</code> 重新编译</span></h2>\n'
                 f'<div class="book">\n{body}\n</div>\n</div>\n'
                 f"{END}\n")
        page = page_path.read_text(encoding="utf-8")
        if START in page:                                   # 幂等: 替换旧块
            page = re.sub(START + r".*?" + END + r"\n?", block, page, flags=re.S)
        else:
            anchor = '<div class="card bridge">'
            assert anchor in page, f"{page_name} 缺少 bridge 卡片"
            page = page.replace(anchor, block + anchor, 1)
        if "assets/mathjax" not in page:                    # 注入 MathJax
            page = page.replace("</head>", MJ_CONFIG + "</head>", 1)
        page_path.write_text(page, encoding="utf-8")
        print(f"✓ {page_name} ← {lesson_dir}/README.md ({len(body)} 字符)")


if __name__ == "__main__":
    build()
