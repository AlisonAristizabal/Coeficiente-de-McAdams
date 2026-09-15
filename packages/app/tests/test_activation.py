import numpy as np
from mcadams_app.session_manager.activation import (
    start_anonymized_call
)
from mcadams_core.session.coefficient import (
    generate_coefficient
)

def test_start_anonymized_call():
    sample_rate = 16000
    stream, id_call = start_anonymized_call(sample_rate)
    coefficient, seed = generate_coefficient(id_call)

    assert np.isclose(stream.mcadams_coefficient, coefficient, atol=1e-6)