"""Core engine: additivity, invariance, and refusal to guess at zeros."""

from __future__ import annotations

import math

import pytest

from climate_attribution import Decomposition, NonPositiveDriver, lmdi_decompose
from climate_attribution.core import log_mean, log_mean_array


# ── Logarithmic mean ─────────────────────────────────────────────────────────


def test_log_mean_lies_between_its_arguments():
    assert 2.0 < log_mean(2.0, 8.0) < 8.0


def test_log_mean_is_symmetric():
    assert log_mean(3.0, 11.0) == pytest.approx(log_mean(11.0, 3.0))


def test_log_mean_of_equal_values_is_the_value():
    assert log_mean(7.0, 7.0) == pytest.approx(7.0)


def test_log_mean_is_continuous_at_equality():
    """The arithmetic-mean fallback must not introduce a step."""
    near = log_mean(1.0, 1.0 + 1e-11)
    far = log_mean(1.0, 1.0 + 1e-6)
    assert near == pytest.approx(far, abs=1e-6)


def test_log_mean_is_zero_when_either_argument_is_non_positive():
    assert log_mean(0.0, 5.0) == 0.0
    assert log_mean(5.0, 0.0) == 0.0
    assert log_mean(-1.0, 5.0) == 0.0


def test_log_mean_array_matches_scalar():
    xs, ys = [1.0, 4.0, 9.0, 0.0], [2.0, 4.0, 3.0, 5.0]
    expected = [log_mean(x, y) for x, y in zip(xs, ys)]
    assert list(log_mean_array(xs, ys)) == pytest.approx(expected)


# ── Additivity ───────────────────────────────────────────────────────────────


def two_driver_case() -> dict:
    return {
        "weight": ([0.4, 0.35, 0.25], [0.2, 0.5, 0.3]),
        "intensity": ([120.0, 300.0, 45.0], [95.0, 310.0, 40.0]),
    }


def test_effects_sum_exactly_to_the_change():
    result = lmdi_decompose(two_driver_case())
    assert result.residual == pytest.approx(0.0, abs=1e-12)
    assert result.explained == pytest.approx(result.delta)


def test_additivity_holds_for_many_drivers():
    """Additivity is what lets the number of multiples stay flexible."""
    drivers = {
        "weight": ([0.4, 0.35, 0.25], [0.2, 0.5, 0.3]),
        "scope1_intensity": ([80.0, 200.0, 30.0], [60.0, 210.0, 25.0]),
        "scope2_ratio": ([1.5, 1.5, 1.5], [1.4, 1.45, 1.6]),
        "inflation": ([1.0, 1.0, 1.0], [1.03, 1.03, 1.03]),
        "coverage": ([1.0, 0.9, 1.0], [1.0, 1.0, 0.95]),
    }
    result = lmdi_decompose(drivers)
    assert len(result.effects) == 5
    result.check_additivity()


def test_single_driver_effect_is_the_change_itself():
    result = lmdi_decompose({"contribution": ([10.0, 20.0], [12.0, 15.0])})
    assert result.effects["contribution"] == pytest.approx(-3.0)
    result.check_additivity()


def test_single_driver_accepts_zeros():
    """The escape hatch for entrants and leavers."""
    result = lmdi_decompose({"contribution": ([10.0, 0.0], [0.0, 7.0])})
    assert result.effects["contribution"] == pytest.approx(-3.0)
    result.check_additivity()


# ── Invariance properties ────────────────────────────────────────────────────


def test_driver_order_does_not_change_effects():
    """LMDI's advantage over Laspeyres: no sensitivity to decomposition order."""
    forward = lmdi_decompose(two_driver_case())
    drivers = two_driver_case()
    reversed_order = lmdi_decompose(dict(reversed(list(drivers.items()))))
    assert forward.effects == pytest.approx(reversed_order.effects)


def test_reversing_the_period_negates_every_effect():
    """Symmetry: decomposing t1 -> t0 mirrors t0 -> t1."""
    forward = lmdi_decompose(two_driver_case())
    backward = lmdi_decompose(
        {name: (t1, t0) for name, (t0, t1) in two_driver_case().items()}
    )
    for name, effect in forward.effects.items():
        assert backward.effects[name] == pytest.approx(-effect)


def test_merging_drivers_sums_their_effects():
    """
    Splitting a driver into multiplicative factors splits its effect additively.
    This is what makes grouping drivers for reporting exact rather than a
    convenient approximation.
    """
    split = lmdi_decompose(
        {
            "quantity": ([100.0, 200.0], [150.0, 180.0]),
            "price": ([10.0, 20.0], [12.0, 19.0]),
            "intensity": ([50.0, 30.0], [45.0, 33.0]),
        }
    )
    merged = lmdi_decompose(
        {
            "value": ([100.0 * 10.0, 200.0 * 20.0], [150.0 * 12.0, 180.0 * 19.0]),
            "intensity": ([50.0, 30.0], [45.0, 33.0]),
        }
    )
    assert merged.effects["value"] == pytest.approx(
        split.effects["quantity"] + split.effects["price"]
    )
    assert merged.effects["intensity"] == pytest.approx(split.effects["intensity"])


def test_unchanged_driver_has_no_effect():
    result = lmdi_decompose(
        {
            "weight": ([0.5, 0.5], [0.3, 0.7]),
            "intensity": ([100.0, 200.0], [100.0, 200.0]),
        }
    )
    assert result.effects["intensity"] == pytest.approx(0.0)


# ── Refusing to guess ────────────────────────────────────────────────────────


def test_zero_driver_raises_rather_than_silently_dropping_the_instrument():
    with pytest.raises(NonPositiveDriver) as excinfo:
        lmdi_decompose(
            {
                "weight": ([0.5, 0.5], [1.0, 0.0]),
                "intensity": ([100.0, 200.0], [90.0, 180.0]),
            },
            labels=["KEPT", "SOLD"],
        )
    message = str(excinfo.value)
    assert "SOLD" in message
    assert "partition" in message


def test_negative_driver_is_rejected():
    """A negative denominator (loss-making revenue) has no log."""
    with pytest.raises(NonPositiveDriver):
        lmdi_decompose(
            {
                "weight": ([0.5, 0.5], [0.5, 0.5]),
                "intensity": ([100.0, -20.0], [90.0, -25.0]),
            }
        )


def test_mismatched_driver_lengths_are_rejected():
    with pytest.raises(ValueError, match="instruments"):
        lmdi_decompose(
            {
                "weight": ([0.5, 0.5], [0.5, 0.5]),
                "intensity": ([100.0], [90.0]),
            }
        )


def test_no_drivers_is_rejected():
    with pytest.raises(ValueError, match="at least one driver"):
        lmdi_decompose({})


def test_label_count_must_match():
    with pytest.raises(ValueError, match="labels"):
        lmdi_decompose({"x": ([1.0, 2.0], [1.0, 2.0])}, labels=["only-one"])


# ── Result object ────────────────────────────────────────────────────────────


def test_check_additivity_detects_a_broken_decomposition():
    broken = Decomposition(
        effects={"weight": 1.0}, total_t0=10.0, total_t1=20.0, labels=("a",)
    )
    with pytest.raises(AssertionError, match="additivity"):
        broken.check_additivity()


def test_effects_match_the_closed_form():
    """Spot-check one effect against the formula computed by hand."""
    weights_t0, weights_t1 = [0.4, 0.6], [0.55, 0.45]
    intensity_t0, intensity_t1 = [100.0, 50.0], [80.0, 70.0]
    result = lmdi_decompose(
        {
            "weight": (weights_t0, weights_t1),
            "intensity": (intensity_t0, intensity_t1),
        }
    )

    expected = 0.0
    for w0, w1, i0, i1 in zip(weights_t0, weights_t1, intensity_t0, intensity_t1):
        m0, m1 = w0 * i0, w1 * i1
        expected += ((m1 - m0) / (math.log(m1) - math.log(m0))) * math.log(i1 / i0)
    assert result.effects["intensity"] == pytest.approx(expected)
