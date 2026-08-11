"""Tests for the shared FormulaFinder cache (find_mfs.get_finder).

There used to be three independent finder caches (find_chnops', the MS2
assignment stage's, and mist-cf's decomp stage's). They are now one, so the
properties that make sharing safe are worth pinning:

* different spellings of the same element set must resolve to one instance
* different element sets must not collide
* symbol order must stay canonical, because callers index into `_symbols`
"""

import time

import pytest

from find_mfs import FormulaFinder, get_finder


def test_same_set_different_spelling_is_one_instance():
    assert get_finder("CHNOPS") is get_finder(["C", "H", "N", "O", "P", "S"])
    # order must not matter either
    assert get_finder("CHNOPS") is get_finder(["S", "P", "O", "N", "H", "C"])
    assert get_finder("CHNOPSFClBrI") is get_finder(
        ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"]
    )


def test_different_sets_are_different_instances():
    assert get_finder("CHNOPS") is not get_finder("CHNOPSFClBrI")
    assert get_finder("CHNOPS") is not get_finder("CHNO")


def test_multi_character_elements_parse_correctly():
    """'Cl' must not be read as 'C' + 'l'."""
    assert get_finder("CHNOPSFClBrI").element_set == {
        "C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I",
    }


@pytest.mark.parametrize("elements", ["CHNOPS", "CHNOPSFClBrI"])
def test_symbol_order_matches_direct_construction(elements):
    """Callers align count vectors to `_symbols`; the cache must not reorder it."""
    direct = FormulaFinder(elements)
    assert [str(s) for s in get_finder(elements)._symbols] == \
           [str(s) for s in direct._symbols]


def test_cached_lookup_is_cheap():
    """It is documented as safe to call in a loop, so it must actually be."""
    get_finder("CHNOPS")  # warm
    start = time.perf_counter()
    for _ in range(10_000):
        get_finder("CHNOPS")
    per_call_us = (time.perf_counter() - start) / 10_000 * 1e6
    assert per_call_us < 20, f"{per_call_us:.1f} us/call -- element parsing not cached?"


def test_find_chnops_uses_the_shared_finder():
    """find_chnops must not keep a finder of its own alongside the cache."""
    from find_mfs import find_chnops

    get_finder.cache_clear()
    find_chnops(mass=180.0634, charge=0, error_ppm=5.0)

    assert get_finder.cache_info().currsize == 1
    # A second query must reuse the cached entry rather than adding one.
    find_chnops(mass=200.0470, charge=0, error_ppm=5.0)
    assert get_finder.cache_info().currsize == 1


def test_cache_clear_forces_a_rebuild():
    first = get_finder("CHNOPS")
    get_finder.cache_clear()
    assert get_finder("CHNOPS") is not first
