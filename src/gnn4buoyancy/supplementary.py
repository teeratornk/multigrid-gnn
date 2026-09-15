"""Hydra app: `gnn4buoyancy-supplementary`, the material moved out of the paper's Supplementary
Information.

The Supplementary Information was shortened before submission. What left it is kept under
`supplementary/`, exported there from the paper's source repository:

  tables/<name>/published.tex   each removed table, as printed before the cut
  tables/<name>/data.csv        its values, one row per printed row, written by mode=refresh
  notes/<name>/published.tex    a paragraph or caption whose results were printed only there
  notes/<name>/data.csv         its values, kept by hand and checked against it
  figures/<name>/published.pdf  each removed figure, as printed
  figures/<name>/data/          the inputs it was drawn from
  si_before_cut.pdf             the Supplementary pages before the cut, the record of the prose
  manifest.yaml                 what each exported file is, and a digest of its content

mode=render (default) writes every table as Markdown and LaTeX, and every figure as PDF, into the
run directory, and recomputes the figure numbers listed under `printed:` in the configuration, which
come from the pre-cut captions and text. mode=check exits nonzero unless each data.csv is what its
published.tex parses to, the numbers of every printed row are the numbers of its data row, every
file matches its digest and every listed figure number matches its printed digits. mode=refresh
rewrites each data.csv from its published.tex. Every mode exits nonzero on any disagreement.

These are results of the production solvers, which are not part of this package. They are kept as
data, with the code that turns the data into the printed tables and figures.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import re
from collections import Counter
from pathlib import Path

import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

from . import figures, summary

FIGURES = {"cht-fields": figures.cht_fields, "cht-transient": figures.cht_transient,
           "engine-geom": figures.engine_geom, "rbc-stats": figures.rbc_stats,
           "size-sweep": figures.size_sweep}
# Files under supplementary/ that the export does not write: this README and the data.csv files.
PACKAGE_OWNED = re.compile(r"^(README\.md|(tables|notes)/[^/]+/data\.csv)$")

# --- published LaTeX tables --------------------------------------------------------------------

_ROW_END = re.compile(r"\\\\(?:\[[^\]]*\])?")
_RULES = re.compile(r"\\(?:toprule|bottomrule|hline)(?![A-Za-z])|\\cmidrule(?:\([^)]*\))?\{[^}]*\}"
                    r"|\\addlinespace(?:\[[^\]]*\])?")
_MIDRULE = re.compile(r"\\midrule(?![A-Za-z])")
_AMP = re.compile(r"(?<!\\)&")
# A printed number keeps its sign, and a number in scientific notation keeps its exponent, so a
# dropped minus or a flipped exponent sign changes the token. The LaTeX form ("$-1.40\\times
# 10^{-3}$", "3.07\\!\\times\\!10^{-4}") and the plain-text form ("-1.40×10^-3") tokenise alike.
# A minus is a sign only when it does not follow a word character, a digit, a dot or another
# minus, so "1.9--2.5", "h_{64}" and "1e-5" keep their unsigned digits.
_SIGNED = r"(?:(?<![\w.\-])-)?\d+(?:\.\d+)?"
# The exponent is braced on both sides or on neither; an unbalanced brace is not an exponent.
_TOKEN = re.compile(rf"(?P<m>{_SIGNED})(?:\\[!,;]|\s)*(?:\\times|×)(?:\\[!,;]|\s)*10\^(?:\{{(?P<e>-?\d+)\}}|(?P<e2>-?\d+))"
                    rf"|(?P<p>{_SIGNED})")


def number_tokens(text: str) -> list[str]:
    """The printed numbers of `text`, in order: "-0.08", "2.5", and "3.07e-4" for 3.07 x 10^-4."""
    return [f"{m.group('m')}e{int(m.group('e') or m.group('e2'))}" if m.group("m") is not None
            else m.group("p") for m in _TOKEN.finditer(text)]


# The argument prefixes of a merged cell, which are layout rather than printed numbers.
_SPAN_ARGS = re.compile(r"\\multirow\{[^}]*\}\{[^}]*\}|\\multicolumn\{[^}]*\}\{[^}]*\}")
# Commands whose argument is the text: (name, arguments, which one to keep).
_UNWRAP = (("multirow", 3, 2), ("multicolumn", 3, 2), ("emph", 1, 0), ("textbf", 1, 0),
           ("text", 1, 0), ("mathrm", 1, 0), ("texttt", 1, 0), ("bm", 1, 0), ("mathbf", 1, 0),
           ("Cref", 1, 0), ("cref", 1, 0), ("ref", 1, 0), ("caption", 1, 0), ("label", 1, 0),
           ("paragraph", 1, 0), ("citet", 1, 0), ("citep", 1, 0), ("mathcal", 1, 0))
# Symbols and their plain-text form. Longer names come first, so \times is not read as \t, and
# accents and ^\dagger come before the characters they contain.
_SYMBOLS = (("\\'e", "é"), ("\\'E", "É"), ("\\'a", "á"), ("\\'o", "ó"), ("\\`e", "è"),
            ('\\"a', "ä"), ('\\"o', "ö"), ('\\"u', "ü"), ("^\\dagger", "†"),
            ("\\times", "×"), ("\\pm", "±"), ("\\dagger", "†"), ("\\approx", "≈"),
            ("\\langle", "⟨"), ("\\rangle", "⟩"), ("\\nabla", "∇"), ("\\alpha", "α"),
            ("\\Gamma", "Γ"), ("\\theta", "θ"), ("\\cdot", "·"), ("\\sim", "∼"), ("\\propto", "∝"),
            ("\\sqrt", "√"), ("\\ldots", "…"), ("\\lesssim", "≲"), ("\\cos", "cos"), ("\\pi", "π"),
            ("\\to ", "→"), ("\\to", "→"),
            ("\\in", "∈"), ("\\|", "‖"), ("\\{", "{"), ("\\}", "}"), ("\\_", "_"), ("\\%", "%"),
            ("\\&", "&"), ("\\#", "#"), ("\\,", " "), ("\\ ", " "), ("\\!", ""), ("~", " "),
            ("--", "–"), ("$", ""))


def _group_end(s: str, i: int) -> int:
    """Index just past the brace group that opens at s[i]."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError(f"unbalanced braces in {s[i:i + 60]!r}")


def _args(s: str, i: int, count: int) -> tuple[list[str], int]:
    """`count` brace arguments from s[i] on -> (their contents, index past the last)."""
    out = []
    for _ in range(count):
        while s[i].isspace():
            i += 1
        end = _group_end(s, i)
        out.append(s[i + 1:end - 1])
        i = end
    return out, i


def _unwrap(s: str, name: str, count: int, keep: int, prefix: str = "") -> str:
    command = re.compile(r"\\%s(?![A-Za-z])" % name)
    while (m := command.search(s)):
        args, end = _args(s, m.end(), count)
        s = s[:m.start()] + prefix + args[keep] + s[end:]
    return s


# A sub- or superscript that is one number, word or command loses its braces (h_{64} is h_64); a
# longer one keeps them (S_{2\to N} is S_{2→N}). \x01 and \x02 stand for those braces meanwhile.
_SIMPLE_SCRIPT = re.compile(r"-?[\d.]+|[A-Za-z0-9]+|\\[A-Za-z]+")


def _script(m: re.Match) -> str:
    body = m.group(2)
    return m.group(1) + (body if _SIMPLE_SCRIPT.fullmatch(body) else "\x01" + body + "\x02")


def cell_text(raw: str) -> str:
    """A published LaTeX cell as plain text: h_{64} is h_64, 1292$\\pm$3 is 1292±3."""
    s = " ".join(raw.split())          # a control space can end a source line: "vs.\" + newline
    for name, count, keep in _UNWRAP:
        s = _unwrap(s, name, count, keep)
    s = _unwrap(s, "textsubscript", 1, 0, prefix="_")
    while (t := re.sub(r"([_^])\{([^{}]*)\}", _script, s)) != s:
        s = t
    s = re.sub(r"(?<!\\)[{}]", "", s)
    for tex, text in _SYMBOLS:
        s = s.replace(tex, text)
    return " ".join(s.replace("\x01", "{").replace("\x02", "}").split())


def tabular(published: str) -> str:
    """The body of the tabular environment in a published table float."""
    start = published.index("\\begin{tabular}") + len("\\begin{tabular}")
    _, start = _args(published, start, 1)                   # the column specification
    return published[start:published.index("\\end{tabular}")]


def _rows(text: str) -> list[str]:
    return [r for r in _ROW_END.split(text) if r.strip()]


def _header_cells(raw_row: str) -> list[str]:
    cells = []
    for cell in _AMP.split(raw_row):
        m = re.match(r"\s*\\multicolumn(?![A-Za-z])", cell)
        if m:
            args, _ = _args(cell, m.end(), 3)
            cells += [cell_text(args[2])] * int(args[0])
        else:
            cells.append(cell_text(cell))
    return cells


def parse_table(published: str) -> dict:
    """columns, rows, printed and raw rows of a published table float.

    Rows above the first \\midrule are the header; a column's name joins its header cells, with a
    \\multicolumn label repeated over the columns it spans. A body row that is one \\multicolumn
    names a group, kept as a leading `group` column. A blank cell repeats the cell above it in the
    same group, which is how the published tables print a merged cell. `printed` keeps the cells
    as printed, blanks included, and `raw` the LaTeX of each body row, for the number check.
    """
    body = _RULES.sub(" ", tabular(published))
    parts = _MIDRULE.split(body, maxsplit=1)
    if len(parts) != 2:
        raise ValueError("no \\midrule under the header")
    header = [_header_cells(r) for r in _rows(parts[0])]
    width = max(len(r) for r in header)
    columns = []
    for c in range(width):
        name = " ".join(r[c] for r in header if c < len(r) and r[c]) or f"column {c + 1}"
        columns.append(name if name not in columns else f"{name} ({c + 1})")
    group, groups, rows, printed, raw = None, [], [], [], []
    for raw_row in _rows(_MIDRULE.sub(" ", parts[1])):
        cells = _AMP.split(raw_row)
        if len(cells) == 1 and re.match(r"\s*\\multicolumn(?![A-Za-z])", cells[0]):
            group = cell_text(cells[0])
            continue
        texts = [cell_text(c) for c in cells] + [""] * (width - len(cells))
        above = rows[-1] if rows and groups[-1] == group else None
        rows.append([t if t or above is None else above[i] for i, t in enumerate(texts)])
        printed.append(texts)
        groups.append(group)
        raw.append(raw_row)
    if group is not None:
        columns = ["group"] + columns
        rows = [[g or ""] + r for g, r in zip(groups, rows)]
        printed = [[g or ""] + r for g, r in zip(groups, printed)]
    return dict(columns=columns, rows=rows, printed=printed, raw=raw)


_LAYOUT = re.compile(r"\\(?:footnotesize|small|smallskip|medskip|bigskip|centering|noindent|par)"
                     r"(?![A-Za-z])")


def _prose(tex: str) -> str:
    return cell_text(_LAYOUT.sub(" ", tex))


def table_caption(published: str) -> tuple[str, str]:
    """(bold title, rest of the caption) of a published table, as plain text."""
    m = re.search(r"\\caption\{", published)
    (body,), _ = _args(published, m.end() - 1, 1)
    bold = re.match(r"\s*\\textbf", body)
    (title,), end = _args(body, bold.end(), 1)
    return cell_text(title).rstrip("."), _prose(body[end:])


def table_note(published: str) -> str:
    """What a published table prints under its tabular, such as a dagger note, as plain text."""
    after = published[published.index("\\end{tabular}") + len("\\end{tabular}"):]
    return _prose(after.split("\\end{table")[0])


def read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def write_csv(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def check_table(name: str, published: str, data: list[list[str]]) -> list[str]:
    """data.csv must be what published.tex parses to, and every printed row's numbers must be the
    numbers of its data row, read straight from the LaTeX so no text conversion can hide a digit."""
    t = parse_table(published)
    problems = []
    if [t["columns"]] + t["rows"] != data:
        problems.append(f"table {name}: data.csv is not what published.tex parses to; "
                        "run mode=refresh")
    first = 1 if t["columns"][0] == "group" else 0
    for i, (raw_row, cells) in enumerate(zip(t["raw"], t["printed"]), 1):
        want = number_tokens(_SPAN_ARGS.sub("", raw_row))
        got = number_tokens(" ".join(cells[first:]))
        if want != got:
            problems.append(f"table {name} row {i}: printed numbers {want}, data {got}")
    return problems


def check_note(name: str, published: str, data: list[list[str]]) -> list[str]:
    """Every value in a note's data.csv is printed in its paragraph: each number of each value cell,
    counted with multiplicity over the whole file, is a number the paragraph's LaTeX prints."""
    if not data or "value" not in data[0]:
        return [f"note {name}: data.csv has no value column"]
    col = data[0].index("value")
    problems = [f"note {name} row {i}: no number in the value {row[col]!r}"
                for i, row in enumerate(data[1:], 1) if not number_tokens(row[col])]
    missing = (Counter(n for row in data[1:] for n in number_tokens(row[col]))
               - Counter(number_tokens(published)))
    if missing:
        problems.append(f"note {name}: values not printed in published.tex: "
                        f"{sorted(missing.elements())}")
    return problems


def render_note(title: str, published: str, data: list[list[str]]) -> str:
    """A note as Markdown: its title, the printed paragraph as plain text, and its values."""
    cell = lambda s: s.replace("|", r"\|")
    lines = [f"**{title}**", "", _prose(published), "",
             "| " + " | ".join(map(cell, data[0])) + " |", "|" + "---|" * len(data[0])]
    lines += ["| " + " | ".join(map(cell, r)) + " |" for r in data[1:]]
    return "\n".join(lines) + "\n"


def _latex_escape(s: str) -> str:
    special = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
               "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
               "^": r"\textasciicircum{}"}
    return "".join(special.get(ch, ch) for ch in s)


def render_latex(columns: list[str], rows: list[list[str]]) -> str:
    """A booktabs tabular with every cell written out. UTF-8: it keeps ±, × and † as characters."""
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(map(_latex_escape, columns)) + r" \\", r"\midrule"]
    lines += [" & ".join(map(_latex_escape, r)) + r" \\" for r in rows]
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def render_markdown(title: str, caption: str, columns: list[str], rows: list[list[str]],
                    note: str = "") -> str:
    """A Markdown table under its title and caption, with its note below. A leading `group` column
    becomes a bold row wherever the group changes, so each label is printed once, as published."""
    cell = lambda s: s.replace("|", r"\|")
    grouped = columns[0] == "group"
    shown = columns[1:] if grouped else columns
    lines = [f"**{title}**", ""] + ([caption, ""] if caption else [])
    lines += ["| " + " | ".join(map(cell, shown)) + " |", "|" + "---|" * len(shown)]
    group = None
    for r in rows:
        if grouped and r[0] and r[0] != group:
            group = r[0]
            lines.append("| " + " | ".join([f"**{cell(group)}**"] + [""] * (len(shown) - 1)) + " |")
        lines.append("| " + " | ".join(map(cell, r[1:] if grouped else r)) + " |")
    lines += ["", note] if note else []
    return "\n".join(lines) + "\n"


# --- exported files ------------------------------------------------------------------------------

def array_digest(arrays) -> str:
    """sha256 over each array's name, dtype, shape and bytes, in name order: the digest the export
    records for a .npz, independent of the zip metadata around the arrays."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        for part in (name, a.dtype.str, repr(a.shape)):
            h.update(part.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def digest(path: Path) -> str:
    if path.suffix == ".npz":
        with np.load(path) as d:
            return array_digest({k: d[k] for k in d.files})
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_files(root: Path) -> list[str]:
    """Every file in manifest.yaml is present with its digest, and nothing else is there."""
    listed = yaml.safe_load((root / "manifest.yaml").read_text(encoding="utf-8"))["files"]
    problems = []
    for rel, entry in listed.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"{rel}: listed in manifest.yaml but missing")
        elif digest(path) != entry["digest"]:
            problems.append(f"{rel}: content differs from its digest in manifest.yaml")
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and rel != "manifest.yaml" and rel not in listed \
                and not PACKAGE_OWNED.match(rel):
            problems.append(f"{rel}: not listed in manifest.yaml")
    return problems


# --- printed figure numbers ----------------------------------------------------------------------

def agrees(value, printed) -> bool:
    """Whether a recomputed value matches what was printed, at the printed digits.

    "14.04" compares at two decimals and "4.0e3" at two significant figures. "2.42/1.23" is a
    ratio of printed values and must be that ratio exactly. A dict or list compares entry by
    entry; a dict may hold more entries than were printed.
    """
    if isinstance(printed, dict):
        return isinstance(value, dict) and all(k in value and agrees(value[k], w)
                                               for k, w in printed.items())
    if isinstance(printed, list):
        return isinstance(value, list) and len(value) == len(printed) \
            and all(agrees(v, w) for v, w in zip(value, printed))
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    s = str(printed)
    if "/" in s:
        a, b = s.split("/")
        return value == float(a) / float(b)
    if "e" in s.lower():
        mantissa = s.lower().split("e")[0]
        figures_printed = len(mantissa.replace(".", "").lstrip("0")) or 1
        return float(f"{value:.{figures_printed - 1}e}") == float(s)
    decimals = len(s.split(".")[1]) if "." in s else 0
    return f"{value:.{decimals}f}" == s


def run(cfg: DictConfig, root: Path, out: Path | None) -> tuple[dict, list[str]]:
    """Check the tables and files, and recompute each figure's numbers; draw into `out` if given."""
    problems = check_files(root)
    for name in cfg.tables:
        folder = root / "tables" / name
        problems += check_table(name, (folder / "published.tex").read_text(encoding="utf-8"),
                                read_csv(folder / "data.csv"))
    for name in cfg.notes:
        folder = root / "notes" / name
        problems += check_note(name, (folder / "published.tex").read_text(encoding="utf-8"),
                               read_csv(folder / "data.csv"))
    numbers = {}
    for name in cfg.figures:
        fcfg = cfg.figures[name]
        pdf = None if out is None else out / "figures" / f"{name}.pdf"
        if pdf is not None:
            pdf.parent.mkdir(parents=True, exist_ok=True)
        numbers[name] = FIGURES[name](fcfg, root, pdf)
        printed = OmegaConf.to_container(fcfg.printed, resolve=True) if "printed" in fcfg else {}
        for key, want in printed.items():
            if not agrees(numbers[name].get(key), want):
                problems.append(f"figure {name} {key}: recomputed {numbers[name].get(key)!r}, "
                                f"printed {want!r}")
    return numbers, problems


@hydra.main(version_base=None, config_path="conf", config_name="supplementary")
def main(cfg: DictConfig) -> None:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import to_absolute_path

    if cfg.mode not in ("render", "check", "refresh"):
        raise ValueError(f"mode must be render, check or refresh, not {cfg.mode!r}")
    root = Path(to_absolute_path(cfg.root))
    if not (root / "manifest.yaml").is_file():
        raise FileNotFoundError(f"no supplementary material at {root}; set root, or run from the "
                                "repository root")
    out = Path(HydraConfig.get().runtime.output_dir)
    if cfg.mode == "refresh":
        for name in cfg.tables:
            folder = root / "tables" / name
            t = parse_table((folder / "published.tex").read_text(encoding="utf-8"))
            write_csv(folder / "data.csv", t["columns"], t["rows"])
            print(f"refreshed tables/{name}/data.csv: {len(t['rows'])} rows")

    drawing = cfg.mode == "render"
    if drawing:
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for name in cfg.tables:
            folder = root / "tables" / name
            published = (folder / "published.tex").read_text(encoding="utf-8")
            data = read_csv(folder / "data.csv")
            title, caption = table_caption(published)
            (out / "tables" / f"{name}.md").write_text(
                render_markdown(title, caption, data[0], data[1:], table_note(published)),
                encoding="utf-8")
            (out / "tables" / f"{name}.tex").write_text(render_latex(data[0], data[1:]),
                                                        encoding="utf-8")
        (out / "notes").mkdir(parents=True, exist_ok=True)
        for name in cfg.notes:
            folder = root / "notes" / name
            (out / "notes" / f"{name}.md").write_text(
                render_note(cfg.notes[name].title,
                            (folder / "published.tex").read_text(encoding="utf-8"),
                            read_csv(folder / "data.csv")), encoding="utf-8")
        print(f"wrote {len(cfg.tables)} tables and {len(cfg.notes)} notes under {out}")
        if importlib.util.find_spec("matplotlib") is None:
            drawing = False
            print("  matplotlib is not installed, so no figure is drawn; install the plots extra")
    numbers, problems = run(cfg, root, out if drawing else None)
    if drawing:
        print(f"wrote {len(cfg.figures)} figures under {out / 'figures'}")
    elif cfg.mode == "render":
        problems.append("figures not drawn: matplotlib is not installed")
    summary.write(dict(mode=cfg.mode, figures=numbers, problems=problems),
                  out / "supplementary.summary.json")
    for name in cfg.figures:
        for key in sorted(cfg.figures[name].get("printed", {})):
            print(f"  {name} {key}: {numbers[name].get(key, 'not computed')}")
    for p in problems:
        print(f"  MISMATCH {p}")
    print(f"supplementary {cfg.mode}: {len(cfg.tables)} tables, {len(cfg.notes)} notes, "
          f"{len(cfg.figures)} figures, "
          f"{len(problems)} mismatch(es)")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
