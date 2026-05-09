"""Lightweight import checks for modules that do not require running experiments."""


def test_constants_importable():
    from escher_poker.constants import KUHN_GAME_VALUE_PLAYER_0

    assert KUHN_GAME_VALUE_PLAYER_0 == -1.0 / 18.0
