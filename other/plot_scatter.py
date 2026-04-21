#!/usr/bin/env python3
"""Simple scatter plotter for 2-column CSV/TXT files.

Examples:
  python plot_scatter.py data.csv
  python plot_scatter.py data.txt --delimiter "\t" --out plot.png
  python plot_scatter.py data.csv --xcol 1 --ycol 0 --title "y vs x"

- Accepts comma/tab/whitespace delimited files.
- Skips blank lines and lines starting with '#'.
- If a header is present, it is auto-detected (unless --no-header).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def _parse_col_spec(spec: str) -> Union[int, str]:
    """Parse column spec as either int index (0-based) or string name."""
    spec = spec.strip()
    try:
        return int(spec)
    except ValueError:
        return spec


def _iter_clean_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            yield line


def _sniff_dialect(sample: str, forced_delimiter: Optional[str]) -> csv.Dialect:
    if forced_delimiter is not None:
        class _D(csv.Dialect):
            delimiter = forced_delimiter
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _D()

    # Try csv.Sniffer; if it fails, fall back to comma.
    sniffer = csv.Sniffer()
    try:
        return sniffer.sniff(sample, delimiters=[",", "\t", ";", "|"])
    except Exception:
        class _D(csv.Dialect):
            delimiter = ","
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        return _D()


def _split_row_fallback(line: str) -> List[str]:
    # Fallback for whitespace-delimited files.
    return line.strip().split()


def read_two_columns(
    path: Path,
    delimiter: Optional[str],
    header_mode: str,
    xcol: Union[int, str],
    ycol: Union[int, str],
) -> Tuple[List[float], List[float], Optional[Sequence[str]]]:
    """Read x/y arrays from a delimited text file.

    header_mode: 'auto' | 'yes' | 'no'
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Collect a small sample for sniffing.
    cleaned = list(_iter_clean_lines(path))
    if not cleaned:
        raise ValueError("No non-empty, non-comment lines found.")

    sample = "".join(cleaned[:20])
    dialect = _sniff_dialect(sample, delimiter)

    def parse_line(line: str) -> List[str]:
        # Try CSV parsing first; if it yields a single field but line has whitespace,
        # fall back to whitespace split.
        row = next(csv.reader([line], dialect=dialect))
        if len(row) <= 1 and ("\t" in line or " " in line.strip()):
            return _split_row_fallback(line)
        return [c.strip() for c in row]

    rows: List[List[str]] = [parse_line(line) for line in cleaned]

    # Header handling.
    header: Optional[Sequence[str]] = None
    if header_mode not in {"auto", "yes", "no"}:
        raise ValueError("header_mode must be one of: auto, yes, no")

    if header_mode == "yes":
        header = rows[0]
        data_rows = rows[1:]
    elif header_mode == "no":
        data_rows = rows
    else:  # auto
        first = rows[0]
        # If first row isn't numeric in both columns, treat it as header.
        # Use first two columns for detection.
        if len(first) >= 2 and (not _is_float(first[0]) or not _is_float(first[1])):
            header = first
            data_rows = rows[1:]
        else:
            data_rows = rows

    if not data_rows:
        raise ValueError("No data rows found after header handling.")

    # Resolve column indices.
    def resolve(col: Union[int, str]) -> int:
        if isinstance(col, int):
            return col
        if header is None:
            raise ValueError(f"Column '{col}' requires a header. Use --header or pass indices.")
        try:
            return list(header).index(col)
        except ValueError:
            raise ValueError(f"Column name not found in header: {col!r}. Header: {list(header)!r}")

    xi = resolve(xcol)
    yi = resolve(ycol)

    xs: List[float] = []
    ys: List[float] = []

    for r in data_rows:
        if max(xi, yi) >= len(r):
            continue
        a, b = r[xi], r[yi]
        if not (_is_float(a) and _is_float(b)):
            continue
        xs.append(float(a))
        ys.append(float(b))

    if not xs:
        raise ValueError("No numeric x/y pairs parsed. Check delimiter/columns/header.")

    return xs, ys, header


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Scatter plot from a 2-column CSV/TXT file.")
    ap.add_argument("input", type=Path, help="Path to input CSV/TXT")
    ap.add_argument("--delimiter", default=None, help="Delimiter override, e.g. ',' or '\\t'")
    ap.add_argument(
        "--header",
        choices=["auto", "yes", "no"],
        default="auto",
        help="Header handling (default: auto)",
    )
    ap.add_argument("--xcol", default="0", help="X column index (0-based) or name (default: 0)")
    ap.add_argument("--ycol", default="1", help="Y column index (0-based) or name (default: 1)")

    ap.add_argument("--title", default=None, help="Plot title")
    ap.add_argument("--xlabel", default=None, help="X label")
    ap.add_argument("--ylabel", default=None, help="Y label")
    ap.add_argument("--alpha", type=float, default=0.7, help="Marker alpha")
    ap.add_argument("--s", type=float, default=12.0, help="Marker size")
    ap.add_argument("--grid", action="store_true", help="Enable grid")
    ap.add_argument("--out", default=None, help="Output image file (png/pdf/svg). If omitted, shows window.")

    args = ap.parse_args(argv)

    xcol = _parse_col_spec(args.xcol)
    ycol = _parse_col_spec(args.ycol)

    xs, ys, header = read_two_columns(
        path=args.input,
        delimiter=args.delimiter,
        header_mode=args.header,
        xcol=xcol,
        ycol=ycol,
    )

    # Lazy import so reading works even if matplotlib isn't installed.
    import matplotlib.pyplot as plt  # type: ignore

    if args.xlabel is None:
        args.xlabel = str(xcol) if header is None or isinstance(xcol, int) else str(xcol)
    if args.ylabel is None:
        args.ylabel = str(ycol) if header is None or isinstance(ycol, int) else str(ycol)

    plt.figure(figsize=(12, 8))
    plt.scatter(xs, ys, s=1, alpha=args.alpha)
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    if args.title:
        plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    if args.grid:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

