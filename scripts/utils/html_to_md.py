#!/usr/bin/env python3
"""
HTML to Markdown Converter

Converts HTML files to Markdown format with support for:
- Headers (h1-h6)
- Paragraphs (p)
- Images (img) - including base64 embedded images
- Tables (table)
- Horizontal rules (hr)
- Links (a)
- Bold/Italic (strong, b, em, i)

Usage:
    python html_to_md.py -i input.html -o output.md
    python html_to_md.py -i input.html  # outputs to input.md
"""

import argparse
import re
from pathlib import Path

try:
    from bs4 import BeautifulSoup, NavigableString

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


def convert_header(element):
    """Convert h1-h6 to # markdown headers."""
    level = int(element.name[1])  # h1 -> 1, h2 -> 2, etc.
    text = element.get_text().strip()
    return f"{'#' * level} {text}\n\n"


def convert_paragraph(element):
    """Convert <p> to markdown paragraph with inline formatting."""
    text = process_inline_elements(element)
    if text.strip():
        return f"{text.strip()}\n\n"
    return ""


def process_inline_elements(element):
    """Process inline elements like bold, italic, links within a parent element."""
    result = ""
    for child in element.children:
        if isinstance(child, NavigableString):
            result += str(child)
        elif child.name in ["strong", "b"]:
            result += f"**{child.get_text()}**"
        elif child.name in ["em", "i"]:
            result += f"*{child.get_text()}*"
        elif child.name == "a":
            href = child.get("href", "")
            text = child.get_text()
            result += f"[{text}]({href})"
        elif child.name == "code":
            result += f"`{child.get_text()}`"
        elif child.name == "br":
            result += "  \n"  # Markdown line break
        elif child.name == "img":
            result += convert_image(child)
        else:
            # Recursively process other elements
            result += process_inline_elements(child)
    return result


def convert_image(element):
    """Convert <img> to markdown image or preserve base64 as HTML."""
    src = element.get("src", "")
    alt = element.get("alt", "image")
    width = element.get("width", "800")

    if src.startswith("data:image"):
        # Base64 image - keep as HTML for GitHub compatibility
        return f'\n<p align="center">\n<img src="{src}" alt="{alt}" width="{width}">\n</p>\n\n'
    else:
        # Regular image - use markdown syntax
        return f"![{alt}]({src})\n\n"


def convert_table(element):
    """Convert <table> to markdown table."""
    rows = element.find_all("tr")
    if not rows:
        return ""

    md_table = "\n"
    header_processed = False

    for row in rows:
        # Check for header cells (th) or data cells (td)
        header_cells = row.find_all("th")
        data_cells = row.find_all("td")

        if header_cells:
            # This is a header row
            cell_texts = [cell.get_text().strip() for cell in header_cells]
            md_table += "| " + " | ".join(cell_texts) + " |\n"
            md_table += "| " + " | ".join(["---"] * len(header_cells)) + " |\n"
            header_processed = True
        elif data_cells:
            # This is a data row
            cell_texts = [cell.get_text().strip() for cell in data_cells]

            # If no header was processed, treat first row as header
            if not header_processed:
                md_table += "| " + " | ".join(cell_texts) + " |\n"
                md_table += "| " + " | ".join(["---"] * len(data_cells)) + " |\n"
                header_processed = True
            else:
                md_table += "| " + " | ".join(cell_texts) + " |\n"

    return md_table + "\n"


def convert_list(element, ordered=False):
    """Convert <ul> or <ol> to markdown list."""
    items = element.find_all("li", recursive=False)
    md_list = "\n"

    for i, item in enumerate(items, 1):
        text = process_inline_elements(item).strip()
        if ordered:
            md_list += f"{i}. {text}\n"
        else:
            md_list += f"- {text}\n"

    return md_list + "\n"


def convert_blockquote(element):
    """Convert <blockquote> to markdown blockquote."""
    text = element.get_text().strip()
    lines = text.split("\n")
    quoted_lines = [f"> {line.strip()}" for line in lines if line.strip()]
    return "\n".join(quoted_lines) + "\n\n"


def convert_code_block(element):
    """Convert <pre> or <code> block to markdown code block."""
    code = element.get_text()
    # Try to detect language from class
    code_elem = element.find("code")
    lang = ""
    if code_elem and code_elem.get("class"):
        classes = code_elem.get("class")
        for cls in classes:
            if cls.startswith("language-"):
                lang = cls.replace("language-", "")
                break

    return f"```{lang}\n{code}\n```\n\n"


def html_to_markdown(html_content):
    """
    Convert HTML content to Markdown.

    Args:
        html_content: HTML string to convert

    Returns:
        Markdown string
    """
    if not BS4_AVAILABLE:
        raise ImportError(
            "BeautifulSoup4 is required for HTML to Markdown conversion. "
            "Install it with: pip install beautifulsoup4"
        )

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for element in soup.find_all(["script", "style", "head"]):
        element.decompose()

    # Get the body or use the whole document
    body = soup.find("body") or soup

    markdown = ""
    processed_elements = set()

    # Define element handlers
    handlers = {
        "h1": convert_header,
        "h2": convert_header,
        "h3": convert_header,
        "h4": convert_header,
        "h5": convert_header,
        "h6": convert_header,
        "p": convert_paragraph,
        "img": convert_image,
        "table": convert_table,
        "ul": lambda e: convert_list(e, ordered=False),
        "ol": lambda e: convert_list(e, ordered=True),
        "blockquote": convert_blockquote,
        "pre": convert_code_block,
    }

    # Process elements in document order
    for element in body.find_all(list(handlers.keys()) + ["hr"]):
        # Skip if already processed (nested elements)
        if id(element) in processed_elements:
            continue

        if element.name == "hr":
            markdown += "---\n\n"
        elif element.name in handlers:
            markdown += handlers[element.name](element)

        processed_elements.add(id(element))

    # Clean up extra whitespace
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown.strip() + "\n"


def convert_file(input_path, output_path=None):
    """
    Convert an HTML file to Markdown.

    Args:
        input_path: Path to input HTML file
        output_path: Path to output Markdown file (optional)

    Returns:
        Path to the output file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".md")
    else:
        output_path = Path(output_path)

    html_content = input_path.read_text(encoding="utf-8")
    markdown = html_to_markdown(html_content)
    output_path.write_text(markdown, encoding="utf-8")

    print(f"Converted: {input_path} -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python html_to_md.py -i report.html
    python html_to_md.py -i report.html -o custom_output.md
    python html_to_md.py --input report.html --output report.md
        """,
    )
    parser.add_argument("-i", "--input", type=Path, required=True, help="Input HTML file path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Markdown file path (default: same as input with .md extension)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    try:
        convert_file(args.input, args.output)
        return 0
    except ImportError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error converting file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
