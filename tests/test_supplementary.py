"""The material moved out of the paper's Supplementary Information, held to what was printed.

The tables, figure inputs and pre-cut pages under supplementary/ are production results kept as
data. Each table's data is what its published LaTeX parses to, and its numbers are the printed
numbers, read from the LaTeX itself. Every file matches the digest the export recorded, and nothing
unlisted sits beside them. The figure numbers listed under `printed:` in the app's configuration,
from the pre-cut captions and text, are recomputed from each figure's inputs. The mutation tests
break each guarantee, in a copy of the files, in the configuration or in the LaTeX conversion, and
show the check then fails.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir

from gnn4buoyancy import supplementary as sup

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "supplementary"
CONF = str(ROOT / "src" / "gnn4buoyancy" / "conf")
TABLES = sorted(p.name for p in (SUP / "tables").iterdir()) if (SUP / "tables").is_dir() else []
NOTES = sorted(p.name for p in (SUP / "notes").iterdir()) if (SUP / "notes").is_dir() else []

pytestmark = pytest.mark.skipif(not (SUP / "manifest.yaml").is_file(),
                                reason="supplementary/ is not in this checkout")


def _cfg():
    with initialize_config_dir(version_base=None, config_dir=CONF):
        return compose(config_name="supplementary")


@pytest.fixture(scope="module")
def cfg():
    return _cfg()


@pytest.fixture(scope="module")
def checked(cfg):
    return sup.run(cfg, SUP, None)


@pytest.fixture
def copy(tmp_path):
    dest = tmp_path / "supplementary"
    shutil.copytree(SUP, dest)
    return dest


# --- the committed material --------------------------------------------------------------------

def test_config_names_every_exported_item(cfg):
    assert sorted(cfg.tables) == TABLES
    assert sorted(cfg.notes) == NOTES
    assert sorted(cfg.figures) == sorted(p.name for p in (SUP / "figures").iterdir())
    assert sorted(cfg.figures) == sorted(sup.FIGURES)


def test_every_file_matches_its_digest_and_nothing_else_is_there():
    assert sup.check_files(SUP) == []


@pytest.mark.parametrize("name", TABLES)
def test_table_data_is_the_published_table(name):
    folder = SUP / "tables" / name
    assert sup.check_table(name, (folder / "published.tex").read_text(encoding="utf-8"),
                           sup.read_csv(folder / "data.csv")) == []


@pytest.mark.parametrize("name", NOTES)
def test_note_values_are_printed_in_the_published_paragraph(name):
    folder = SUP / "notes" / name
    assert sup.check_note(name, (folder / "published.tex").read_text(encoding="utf-8"),
                          sup.read_csv(folder / "data.csv")) == []


def test_every_printed_figure_number_is_recomputed(checked):
    numbers, problems = checked
    assert problems == []
    assert set(numbers) == set(sup.FIGURES)


def test_readme_lists_every_item(cfg):
    readme = (SUP / "README.md").read_text(encoding="utf-8")
    for name in list(cfg.tables) + list(cfg.notes) + list(cfg.figures):
        assert f"`{name}`" in readme, name


def test_the_pre_cut_pages_are_a_pdf():
    assert (SUP / "si_before_cut.pdf").read_bytes()[:5] == b"%PDF-"


def test_check_mode_passes_end_to_end(tmp_path):
    r = subprocess.run([sys.executable, "-m", "gnn4buoyancy.supplementary", "mode=check",
                        f"hydra.run.dir={tmp_path}"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 mismatch(es)" in r.stdout


# --- the checks fail when the material is broken ---------------------------------------------------

def test_a_changed_table_digit_fails(copy, cfg):
    data = copy / "tables" / "msiter" / "data.csv"
    text = data.read_text(encoding="utf-8")
    assert "157.4" in text
    data.write_text(text.replace("157.4", "157.5", 1), encoding="utf-8")
    _, problems = sup.run(cfg, copy, None)
    assert any("table msiter" in p and "parses to" in p for p in problems)


def test_a_changed_published_table_fails_its_digest(copy, cfg):
    tex = copy / "tables" / "weak" / "published.tex"
    text = tex.read_text(encoding="utf-8")
    assert "973" in text
    tex.write_text(text.replace("973", "974", 1), encoding="utf-8")
    _, problems = sup.run(cfg, copy, None)
    assert any(p.startswith("tables/weak/published.tex") for p in problems)


def test_a_changed_input_sample_fails_its_digest(copy, cfg):
    path = copy / "figures" / "rbc-stats" / "data" / "nu_trace.npz"
    with np.load(path) as d:
        arrays = {k: d[k].copy() for k in d.files}
    arrays["nu_bot"][0] += 1e-12
    np.savez_compressed(path, **arrays)
    _, problems = sup.run(cfg, copy, None)
    assert any(p.startswith("figures/rbc-stats/data/nu_trace.npz") for p in problems)


def test_an_unlisted_file_fails(copy, cfg):
    (copy / "figures" / "cht-fields" / "notes.txt").write_text("not exported\n")
    _, problems = sup.run(cfg, copy, None)
    assert any(p.startswith("figures/cht-fields/notes.txt") for p in problems)


def test_a_changed_note_value_fails(copy, cfg):
    data = copy / "notes" / "coarse-space-ablation" / "data.csv"
    text = data.read_text(encoding="utf-8")
    assert "283.5" in text
    data.write_text(text.replace("283.5", "283.6", 1), encoding="utf-8")
    _, problems = sup.run(cfg, copy, None)
    assert any("note coarse-space-ablation" in p and "283.6" in p for p in problems)


def test_a_wrong_printed_figure_number_fails():
    cfg = _cfg()
    cfg.figures["rbc-stats"].printed.mean_nu = "14.05"
    _, problems = sup.run(cfg, SUP, None)
    assert any("rbc-stats mean_nu" in p for p in problems)


def test_the_digit_check_does_not_trust_the_text_conversion(monkeypatch):
    original = sup.cell_text
    monkeypatch.setattr(sup, "cell_text", lambda raw: original(raw).replace("7", ""))
    folder = SUP / "tables" / "msiter"
    problems = sup.check_table("msiter", (folder / "published.tex").read_text(encoding="utf-8"),
                               sup.read_csv(folder / "data.csv"))
    assert any("printed numbers" in p for p in problems)


# --- the comparison and the renderers ----------------------------------------------------------------

@pytest.mark.parametrize("raw,text", [
    ("$h_{64}$", "h_64"), ("1292$\\pm$3", "1292±3"), ("{1.00$\\times$}", "1.00×"),
    ("\\multirow{3}{*}{$10^{4}$}", "10^4"), ("900$^\\dagger$", "900†"), ("{--}", "–"),
    ("Rayleigh--B\\'enard", "Rayleigh–Bénard"), ("(vs.\\\n$1.62\\times10^{-3}$", "(vs. 1.62×10^-3"),
    ("$\\|e_{u}\\|_{L^{2}}\\sim10^{-14}$", "‖e_u‖_{L^2}∼10^-14"),
    ("$S_{2\\to N}=t_2/t_N$", "S_{2→N}=t_2/t_N"), ("$h_{\\Gamma}/R$", "h_Γ/R"),
    ("$n\\propto\\sqrt{\\text{NPROC}}$", "n∝√NPROC"),
    ("\\emph{Two-dimensional benchmark}", "Two-dimensional benchmark"), ("P\\textsubscript{2}", "P_2"),
])
def test_cell_text_turns_published_latex_into_plain_text(raw, text):
    assert sup.cell_text(raw) == text


@pytest.mark.parametrize("value,printed,ok", [
    (14.036, "14.04", True), (14.046, "14.04", False),
    (4400.0, "4e3", True), (4600.0, "4e3", False),
    (3960.0, "4.0e3", True), (3940.0, "4.0e3", False),
    (2.42 / 1.23, "2.42/1.23", True), (1.97, "2.42/1.23", False),
    ([1.0, 0.71], ["1.00", "0.71"], True), ([1.0], ["1.00", "0.71"], False),
    ({"a": 1.0, "b": 2.0}, {"a": "1.00"}, True), ({"b": 2.0}, {"a": "1.00"}, False),
    (None, "1.00", False), (True, "1", False),
])
def test_agrees_compares_at_the_printed_digits(value, printed, ok):
    assert sup.agrees(value, printed) is ok


def test_tables_render_every_cell():
    for name in TABLES:
        published = (SUP / "tables" / name / "published.tex").read_text(encoding="utf-8")
        data = sup.read_csv(SUP / "tables" / name / "data.csv")
        title, caption = sup.table_caption(published)
        assert title and caption, name
        markdown = sup.render_markdown(title, caption, data[0], data[1:], sup.table_note(published))
        latex = sup.render_latex(data[0], data[1:])
        first = 1 if data[0][0] == "group" else 0
        for row in data:
            for cell in row[first:]:
                assert cell.replace("|", r"\|") in markdown, (name, cell)
            for cell in row:
                assert sup._latex_escape(cell) in latex, (name, cell)
        for row in data[1:] if first else []:
            assert f"**{row[0]}**" in markdown, (name, row[0])


def test_markdown_prints_each_group_once_and_keeps_caption_and_note():
    msiter = (SUP / "tables" / "msiter" / "published.tex").read_text(encoding="utf-8")
    data = sup.read_csv(SUP / "tables" / "msiter" / "data.csv")
    title, caption = sup.table_caption(msiter)
    markdown = sup.render_markdown(title, caption, data[0], data[1:], sup.table_note(msiter))
    assert markdown.count("**Two-dimensional benchmark**") == 1
    assert "| group |" not in markdown
    assert caption.startswith("Cost of one SIMPLE outer iteration")
    weak = (SUP / "tables" / "weak-transient" / "published.tex").read_text(encoding="utf-8")
    assert sup.table_note(weak).startswith("† At the pressure cap")


@pytest.mark.parametrize("name", NOTES)
def test_note_markdown_carries_every_value(cfg, name):
    folder = SUP / "notes" / name
    data = sup.read_csv(folder / "data.csv")
    markdown = sup.render_note(cfg.notes[name].title,
                               (folder / "published.tex").read_text(encoding="utf-8"), data)
    for row in data:
        for cell in row:
            assert cell.replace("|", r"\|") in markdown, (name, cell)


@pytest.mark.parametrize("name", ["cht-transient", "engine-geom", "rbc-stats", "size-sweep",
                                  pytest.param("cht-fields", marks=pytest.mark.slow)])
def test_figures_draw(cfg, name, tmp_path):
    pytest.importorskip("matplotlib")
    pdf = tmp_path / f"{name}.pdf"
    sup.FIGURES[name](cfg.figures[name], SUP, pdf)
    assert pdf.read_bytes()[:5] == b"%PDF-"
