from mcadams_core.config.lpc_defaults import compute_lpc_order

def test_compute_lpc_order_16kHz():
    sample_rate = 16000
    expected_order = round(sample_rate / 1000) + 4  # offset por defecto es 4
    computed_order = compute_lpc_order(sample_rate)
    assert computed_order == expected_order

def test_compute_lpc_order_8kHz():
    sample_rate = 8000
    expected_order = round(sample_rate / 1000) + 4  # offset por defecto es 4
    computed_order = compute_lpc_order(sample_rate)
    assert computed_order == expected_order

def test_compute_lpc_order_custom_offset():
    sample_rate = 16000
    custom_offset = 5
    expected_order = round(sample_rate / 1000) + custom_offset
    computed_order = compute_lpc_order(sample_rate, offset=custom_offset)
    assert computed_order == expected_order