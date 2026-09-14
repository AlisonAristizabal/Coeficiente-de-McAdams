from mcadams_core.session.coefficient import generate_coefficient
from mcadams_core.config.mcadams_defaults import (
    MIN_MCADAMS_COEFFICIENT,
    MAX_MCADAMS_COEFFICIENT,
)

def test_generate_coefficient_deterministic():
    session_id = "test_session"
    coefficient1, seed1 = generate_coefficient(session_id)
    coefficient2, seed2 = generate_coefficient(session_id)

    assert coefficient1 == coefficient2
    assert seed1 == seed2

def test_generate_coefficient_range():
    session_id = [f"test_session_{i}" for i in range(100)]
    coefficients = [generate_coefficient(sid)[0] for sid in session_id]

    for coefficient in coefficients:
        assert MIN_MCADAMS_COEFFICIENT <= coefficient <= MAX_MCADAMS_COEFFICIENT

