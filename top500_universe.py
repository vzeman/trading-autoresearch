"""Fetch the current S&P 500 symbol universe with stdlib-only parsing."""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.request import Request, urlopen

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class _SP500TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_constituents = False
        self.in_row = False
        self.in_cell = False
        self.current_cell: list[str] = []
        self.current_row: list[str] = []
        self.rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_d = {k: v for k, v in attrs}
        if tag == "table" and attrs_d.get("id") == "constituents":
            self.in_constituents = True
        if not self.in_constituents:
            return
        if tag == "tr":
            self.in_row = True
            self.current_row = []
        elif tag in {"td", "th"} and self.in_row:
            self.in_cell = True
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.in_constituents and self.in_cell:
            self.current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.in_constituents:
            return
        if tag in {"td", "th"} and self.in_cell:
            cell = " ".join("".join(self.current_cell).split())
            self.current_row.append(cell)
            self.current_cell = []
            self.in_cell = False
        elif tag == "tr" and self.in_row:
            if self.current_row:
                self.rows.append(self.current_row)
            self.current_row = []
            self.in_row = False
        elif tag == "table":
            self.in_constituents = False


def load_top500_symbols() -> list[str]:
    """Return current S&P 500 constituent tickers in page order.

    The index has slightly more than 500 tradable symbols because a few
    companies have multiple listed share classes.
    """
    req = Request(SP500_URL, headers={"User-Agent": "trading-autoresearch/0.1"})
    html = urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
    parser = _SP500TableParser()
    parser.feed(html)
    symbols: list[str] = []
    for row in parser.rows:
        if not row or row[0].lower() == "symbol":
            continue
        symbol = row[0].strip().replace("\n", "")
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    if len(symbols) < 450:
        raise RuntimeError(f"expected S&P 500 table, parsed only {len(symbols)} symbols")
    return symbols
