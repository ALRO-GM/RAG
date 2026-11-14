"""The join link and its terminal QR code."""

import math

import segno

from transcriber.adapters.inbound.join import join_link, qr_text

CODE = "Abc_-123" * 6


def test_through_the_tunnel_the_link_is_public_https() -> None:
    assert join_link(CODE, "rag.peer4u.app", 8770, tunnel=True) == (
        f"https://rag.peer4u.app/#k={CODE}"
    )


def test_without_the_tunnel_the_link_is_local() -> None:
    assert join_link(CODE, "rag.peer4u.app", 8771, tunnel=False) == (
        f"http://127.0.0.1:8771/#k={CODE}"
    )


def test_the_code_is_only_in_the_fragment() -> None:
    link = join_link(CODE, "example.org", 1, tunnel=True)
    before, _, fragment = link.partition("#")
    assert CODE not in before and fragment == f"k={CODE}"


def test_the_qr_code_draws_the_link() -> None:
    link = join_link(CODE, "rag.peer4u.app", 8770, tunnel=True)
    drawn = qr_text(link)
    width, height = segno.make(link, error="m").symbol_size()
    rows = drawn.rstrip("\n").split("\n")
    assert len(rows) == math.ceil(height / 2)  # two modules per line
    assert {len(row) for row in rows} == {width}
    assert set("".join(rows)) <= {" ", "█", "▀", "▄"}
    other = qr_text(join_link("x" * 43, "rag.peer4u.app", 8770, True))
    assert other != drawn
