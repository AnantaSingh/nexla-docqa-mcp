"""Unit tests for the layout-reconstruction internals — the most bug-prone module.

These run on synthetic word boxes (no real PDFs, no API keys). The headline test is a
regression for the real bug that once split a financial-statement row's label away from its
values; the row must reconstruct as ONE line.

A PyMuPDF "word" is a tuple (x0, y0, x1, y1, text, ...); the parser uses indices 0,1,2,4.
"""

from docqa.pdf_parser import (
    _MIN_WORDS_FOR_COLUMNS,
    _clean_line,
    _detect_gutter_x,
    _reconstruct_lines,
    _strip_running_headers_footers,
    _word_numeric_ratio,
)


def w(x0, y, text, width=20):
    return (x0, y, x0 + width, y + 8, text)


# --- _reconstruct_lines -------------------------------------------------------

def test_financial_row_stays_one_line_regression():
    # "Net sales  $ 222,730  $ 192,052  $ 163,220" all on the same visual row (same y).
    words = [
        w(50, 100, "Net"), w(95, 100, "sales"),
        w(300, 100, "$"), w(330, 100, "222,730"),
        w(420, 100, "$"), w(450, 100, "192,052"),
        w(520, 100, "$"), w(550, 100, "163,220"),
        w(50, 120, "Membership"), w(330, 120, "4,224"),  # next row
    ]
    lines = _reconstruct_lines(words)
    assert lines[0] == "Net sales $ 222,730 $ 192,052 $ 163,220"
    assert lines[1] == "Membership 4,224"


def test_words_are_ordered_left_to_right_within_a_row():
    words = [w(300, 50, "third"), w(50, 50, "first"), w(175, 50, "second")]
    assert _reconstruct_lines(words) == ["first second third"]


# --- _clean_line --------------------------------------------------------------

def test_clean_line_collapses_dotted_leaders_but_keeps_decimals():
    assert _clean_line("Net sales . . . . . 222,730") == "Net sales 222,730"
    assert _clean_line("Diluted .......... 13.14") == "Diluted 13.14"  # decimal preserved
    assert _clean_line("under­stand") == "understand"  # soft hyphen removed


# --- _detect_gutter_x ---------------------------------------------------------

def test_gutter_detected_for_clean_two_columns():
    width = 700  # central band searched is [0.40,0.60]*700 = [280, 420]
    words = []
    for i in range(45):
        y = 50 + i * 10
        words.append(w(60 + (i % 4) * 30, y, "L"))    # left column, x well below 280
        words.append(w(460 + (i % 4) * 30, y, "R"))   # right column, x well above 420
    assert len(words) >= _MIN_WORDS_FOR_COLUMNS
    gx = _detect_gutter_x(words, width)
    assert gx is not None and 280 <= gx <= 420


def test_no_gutter_when_too_few_words():
    words = [w(60, 50 + i * 10, "x") for i in range(10)]
    assert _detect_gutter_x(words, 700) is None


def test_no_gutter_when_content_spans_the_centre():
    # single-column-ish: every word straddles the whole central band -> no clean gutter
    words = [w(270, 50 + i * 5, "wide", width=160) for i in range(90)]  # x0=270,x1=430
    assert _detect_gutter_x(words, 700) is None


# --- _word_numeric_ratio ------------------------------------------------------

def test_word_numeric_ratio():
    words = [w(0, 0, "Net"), w(0, 0, "sales"), w(0, 0, "222,730"),
             w(0, 0, "$163,220"), w(0, 0, "(158)")]
    assert abs(_word_numeric_ratio(words) - 0.6) < 1e-9  # 3 of 5 look numeric
    assert _word_numeric_ratio([]) == 0.0


# --- _strip_running_headers_footers ------------------------------------------

def test_strip_running_headers_and_footers():
    # bodies must be digit-free: lines differing only by a number normalise together (that's how
    # "Page 1"/"Page 2" footers collapse), so we use distinct alphabetic bodies here.
    bodies = ["alpha section", "bravo section", "charlie section", "delta section", "echo section"]
    pages = [["ACME ANNUAL REPORT", bodies[i], f"Page {i}"] for i in range(5)]
    cleaned = _strip_running_headers_footers(pages)
    # repeated header + (page-number-insensitive) footer removed; unique body kept
    assert all("ACME ANNUAL REPORT" not in page for page in cleaned)
    assert all(not any(line.startswith("Page ") for line in page) for page in cleaned)
    assert cleaned[2] == ["charlie section"]


def test_strip_is_noop_for_few_pages():
    pages = [["HEADER", "body"], ["HEADER", "body2"]]  # < 4 pages -> unchanged
    assert _strip_running_headers_footers(pages) == pages
