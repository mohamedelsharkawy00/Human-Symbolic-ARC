"""MkDocs hook: make every sidebar/page title the RAW FILENAME, uniformly.

Without this, MkDocs uses each file's first Markdown heading as the title when one
exists (often a shouty ALL-CAPS line) and the filename otherwise - so the nav is a
mix of styles. Using the bare filename (underscores kept, no capitalization) makes
the outline consistent and exactly matches the file you edit. Home page left alone.
"""
import os


def on_page_markdown(markdown, page, config, files):
    src = page.file.src_path.replace("\\", "/")
    page.meta["title"] = os.path.splitext(os.path.basename(src))[0]
    return markdown


def on_nav(nav, config, files):
    """Show the RAW directory name for each folder section in the sidebar.

    MkDocs auto-titles section (folder) entries by stripping separators and
    capitalizing (e.g. '1 - Overall Design' -> '1 Overall Design', dropping the
    dash). This sets each section's title back to its real directory name, found
    from the path of the first page underneath it - so the sidebar matches disk.
    """
    for section in _sections(nav):
        dirname = _section_dirname(section)
        if dirname:
            section.title = dirname
    return nav


def _sections(items):
    for item in items:
        if getattr(item, "is_section", False):
            yield item
            yield from _sections(item.children)


def _depth(section):
    """How many section levels deep this section is (top-level = 0)."""
    d, p = 0, getattr(section, "parent", None)
    while p is not None:
        d += 1
        p = getattr(p, "parent", None)
    return d


def _section_dirname(section):
    """The directory segment that names this section, from a descendant page path."""
    for child in _walk(section.children):
        src = getattr(getattr(child, "file", None), "src_path", None)
        if src:
            parts = src.replace("\\", "/").split("/")[:-1]  # drop filename
            idx = _depth(section)
            if idx < len(parts):
                return parts[idx]
    return None


def _walk(items):
    for item in items:
        yield item
        if getattr(item, "children", None):
            yield from _walk(item.children)
