import numpy as np
from mcadams_core.signal_transform.mcadams import (
    transform_angle,
    transform_poles,
    process_frame,
)

def test_process_frame_coefficient_changes_output():
    frame_length = 320
    t = np.arange(frame_length)
    frame = np.sin(2 * np.pi * t / 50)  # Señal de prueba
    order = 20

    result_identity = process_frame(frame, order, mcadams_coefficient=1.0)
    result_anonymized = process_frame(frame, order, mcadams_coefficient=0.8)
    assert not np.allclose(result_identity, result_anonymized)


def test_transform_angle():
    angle = 0.5
    mcadams_coefficient = 0.8
    result = transform_angle(angle, mcadams_coefficient)
    answer = angle ** mcadams_coefficient
    assert np.isclose(result, answer)

def test_transform_poles_real_unchanged():
    poles = np.array([0.5+0j, 0.8+0j, 0.9+0j])
    mcadams_coefficient = 0.8
    transformed = transform_poles(poles, mcadams_coefficient)
    assert np.allclose(transformed, poles)

def test_transform_poles_preserves_conjugate_pairs():
    radius = 0.8
    angle = 0.5
    pole_positive = radius * np.exp(1j * angle)
    pole_negative = np.conj(pole_positive)
    poles = np.array([pole_positive, pole_negative])
    mcadams_coefficient = 0.8

    transformed = transform_poles(poles, mcadams_coefficient)

    assert np.isclose(transformed[0], np.conj(transformed[1]))

def test_transform_poles_keeps_radius():
    radius = 0.8
    angle = 0.5
    pole_positive = radius * np.exp(1j * angle)
    pole_negative = np.conj(pole_positive)
    poles = np.array([pole_positive, pole_negative])
    mcadams_coefficient = 0.8

    transformed = transform_poles(poles, mcadams_coefficient)

    assert np.allclose(np.abs(transformed), radius)