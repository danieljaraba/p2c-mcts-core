"""Root conftest.py – expose all shared fixtures to the test suite.

Pytest auto-discovers fixtures defined in conftest.py files, so tests can use
them without explicit imports, avoiding F811 re-definition warnings.
"""

from tests.fixtures.crafting_fixtures import (  # noqa: F401
    base_item,
    bench_craft_action,
    chaos_orb_action,
    empty_state,
    exalted_orb_action,
    partial_goal,
    partial_state,
    single_mod_goal,
)
