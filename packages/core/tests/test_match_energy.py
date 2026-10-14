import numpy as np
import pytest
import scipy.signal as signal

from mcadams_core.batch.utterance import anonymize_utterance
from mcadams_core.config.energy_defaults import MIN_FRAME_ENERGY
from mcadams_core.config.lpc_defaults import compute_lpc_order
from mcadams_core.signal_transform import mcadams
from mcadams_core.signal_transform.framing import (
    apply_window,
    compute_frame_params,
    hann_window,
)
from mcadams_core.signal_transform.mcadams import process_frame
from mcadams_core.signal_transform.resynth import match_energy

SAMPLE_RATE = 16000
ORDER = compute_lpc_order(SAMPLE_RATE)
FRAME_LENGTH, _ = compute_frame_params(SAMPLE_RATE)


def energia(x):
    return float(np.sum(np.asarray(x) ** 2))


def vocal_sintetica(f0, formantes, segundos=0.5, semilla=0):
    """Tren de pulsos filtrado por resonadores: una vocal artificial."""
    rng = np.random.default_rng(semilla)
    n = int(SAMPLE_RATE * segundos)
    x = np.zeros(n)
    x[:: int(SAMPLE_RATE / f0)] = 1.0
    for frecuencia, ancho in formantes:
        r = np.exp(-np.pi * ancho / SAMPLE_RATE)
        theta = 2 * np.pi * frecuencia / SAMPLE_RATE
        x = signal.lfilter([1], [1, -2 * r * np.cos(theta), r * r], x)
    x = x / np.max(np.abs(x))
    return 0.3 * x + 0.002 * rng.standard_normal(n)


def fragmento_enventanado(semilla=0, f0=120, formantes=((700, 90), (1200, 110), (2600, 160))):
    voz = vocal_sintetica(f0, formantes, semilla=semilla)
    inicio = 1600
    marco = voz[inicio:inicio + FRAME_LENGTH]
    return apply_window(marco, hann_window(FRAME_LENGTH))


# --- constante -------------------------------------------------------------

def test_min_frame_energy_vale_1e_menos_12():
    assert MIN_FRAME_ENERGY == 1e-12


# --- match_energy ----------------------------------------------------------

def test_match_energy_iguala_la_energia_de_la_referencia():
    rng = np.random.default_rng(0)
    reconstruido = rng.normal(size=320) * 3.0
    referencia = rng.normal(size=320) * 0.5
    resultado = match_energy(reconstruido, referencia)
    assert energia(resultado) == pytest.approx(energia(referencia), rel=1e-12)


def test_match_energy_aplica_un_unico_factor_positivo():
    rng = np.random.default_rng(1)
    reconstruido = rng.normal(size=320)
    referencia = rng.normal(size=320) * 2.0
    resultado = match_energy(reconstruido, referencia)
    factores = resultado[reconstruido != 0] / reconstruido[reconstruido != 0]
    assert np.allclose(factores, factores[0])
    assert factores[0] > 0


def test_match_energy_amplifica_un_fragmento_mas_debil():
    referencia = np.ones(100)
    reconstruido = np.full(100, 0.1)
    resultado = match_energy(reconstruido, referencia)
    assert np.allclose(resultado, 1.0)


def test_match_energy_atenua_un_fragmento_mas_fuerte():
    referencia = np.full(100, 0.5)
    reconstruido = np.full(100, 2.0)
    resultado = match_energy(reconstruido, referencia)
    assert np.allclose(resultado, 0.5)


def test_match_energy_con_energias_iguales_no_cambia_nada():
    rng = np.random.default_rng(2)
    x = rng.normal(size=320)
    assert np.allclose(match_energy(x, x.copy()), x)


def test_match_energy_conserva_forma_y_signo():
    reconstruido = np.array([1.0, -2.0, 3.0, -4.0])
    referencia = np.array([10.0, 10.0, 10.0, 10.0])
    resultado = match_energy(reconstruido, referencia)
    assert resultado.shape == reconstruido.shape
    assert np.array_equal(np.sign(resultado), np.sign(reconstruido))


def test_match_energy_no_modifica_las_entradas():
    reconstruido = np.array([1.0, -2.0, 3.0])
    referencia = np.array([0.5, 0.5, 0.5])
    copia_reconstruido, copia_referencia = reconstruido.copy(), referencia.copy()
    match_energy(reconstruido, referencia)
    assert np.array_equal(reconstruido, copia_reconstruido)
    assert np.array_equal(referencia, copia_referencia)


def test_match_energy_referencia_en_silencio_deja_el_fragmento_igual():
    reconstruido = np.array([1.0, -2.0, 3.0])
    resultado = match_energy(reconstruido, np.zeros(3))
    assert np.array_equal(resultado, reconstruido)


def test_match_energy_fragmento_en_silencio_sale_sin_nan():
    resultado = match_energy(np.zeros(5), np.ones(5))
    assert np.array_equal(resultado, np.zeros(5))
    assert np.isfinite(resultado).all()


def test_match_energy_dos_silencios_dan_silencio():
    resultado = match_energy(np.zeros(5), np.zeros(5))
    assert np.array_equal(resultado, np.zeros(5))


def test_match_energy_escala_fragmentos_muy_debiles_pero_sobre_el_umbral():
    referencia = np.full(100, 1e-5)      # energía 1e-8 > 1e-12
    reconstruido = np.full(100, 1e-6)
    resultado = match_energy(reconstruido, referencia)
    assert energia(resultado) == pytest.approx(energia(referencia), rel=1e-9)


def test_match_energy_la_energia_igual_al_umbral_no_se_reescala():
    reconstruido = np.array([2.0, 0.0])          # energía 4
    referencia = np.array([1.0, 1.0])            # energía 2
    resultado = match_energy(reconstruido, referencia, min_energy=2.0)
    assert np.array_equal(resultado, reconstruido)


def test_match_energy_justo_sobre_el_umbral_si_se_reescala():
    reconstruido = np.array([2.0, 0.0])
    referencia = np.array([1.0, 1.0])
    resultado = match_energy(reconstruido, referencia, min_energy=1.99)
    assert energia(resultado) == pytest.approx(2.0)


def test_match_energy_umbral_de_la_reconstruccion_tambien_cuenta():
    reconstruido = np.array([0.1, 0.0])          # energía 0.01
    referencia = np.array([5.0, 5.0])
    resultado = match_energy(reconstruido, referencia, min_energy=0.5)
    assert np.array_equal(resultado, reconstruido)


def test_match_energy_usa_el_umbral_por_defecto_de_la_configuracion():
    reconstruido = np.full(4, 1e-7)              # energía 4e-14 < 1e-12
    referencia = np.full(4, 1.0)
    resultado = match_energy(reconstruido, referencia)
    assert np.array_equal(resultado, reconstruido)


# --- process_frame ---------------------------------------------------------

@pytest.mark.parametrize("coeficiente", [0.7, 0.75, 0.8, 0.85, 0.9])
@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_process_frame_conserva_la_energia_del_fragmento(coeficiente, semilla):
    fragmento = fragmento_enventanado(semilla=semilla)
    resultado = process_frame(fragmento, ORDER, coeficiente)
    assert energia(resultado) == pytest.approx(energia(fragmento), rel=1e-9)


@pytest.mark.parametrize(
    "formantes", [((300, 70), (2300, 150), (3000, 200)), ((350, 80), (800, 100), (2400, 150))]
)
def test_process_frame_conserva_la_energia_en_otras_vocales(formantes):
    fragmento = fragmento_enventanado(f0=200, formantes=formantes)
    resultado = process_frame(fragmento, ORDER, 0.72)
    assert energia(resultado) == pytest.approx(energia(fragmento), rel=1e-9)


def test_process_frame_con_coeficiente_uno_no_cambia_el_fragmento():
    fragmento = fragmento_enventanado()
    resultado = process_frame(fragmento, ORDER, 1.0)
    assert np.allclose(resultado, fragmento, atol=1e-6)


def test_process_frame_sigue_modificando_el_fragmento_con_coeficiente_menor_que_uno():
    fragmento = fragmento_enventanado()
    resultado = process_frame(fragmento, ORDER, 0.75)
    assert not np.allclose(resultado, fragmento, atol=1e-3)


def test_process_frame_fragmento_en_silencio_sale_en_silencio():
    resultado = process_frame(np.zeros(FRAME_LENGTH), ORDER, 0.8)
    assert np.isfinite(resultado).all()
    assert np.array_equal(resultado, np.zeros(FRAME_LENGTH))


def test_process_frame_compara_contra_el_fragmento_de_entrada(monkeypatch):
    fragmento = fragmento_enventanado()
    llamadas = []

    def espia(reconstruido, referencia, *args, **kwargs):
        llamadas.append((reconstruido, referencia))
        return reconstruido

    monkeypatch.setattr(mcadams, "match_energy", espia)
    mcadams.process_frame(fragmento, ORDER, 0.8)
    assert len(llamadas) == 1
    reconstruido, referencia = llamadas[0]
    assert referencia is fragmento
    assert reconstruido is not fragmento


# --- nivel global del audio completo ---------------------------------------

def enunciado_sintetico():
    """Cuatro vocales con pausas de ruido de fondo."""
    vocales = [
        (120, ((700, 90), (1200, 110), (2600, 160)), 0.30),
        (200, ((300, 70), (2300, 150), (3000, 200)), 0.15),
        (110, ((350, 80), (800, 100), (2400, 150)), 0.40),
        (150, ((700, 90), (1200, 110), (2600, 160)), 0.05),
    ]
    rng = np.random.default_rng(0)
    partes = []
    for f0, formantes, amplitud in vocales:
        voz = vocal_sintetica(f0, formantes, segundos=0.5, semilla=3)
        partes.append(amplitud * voz / np.max(np.abs(voz)))
        partes.append(0.002 * rng.standard_normal(int(SAMPLE_RATE * 0.15)))
    return np.concatenate(partes)


@pytest.mark.parametrize("sesion", [f"sesion-{i}" for i in range(12)])
def test_el_nivel_del_enunciado_anonimizado_queda_cerca_del_original(sesion):
    original = enunciado_sintetico()
    anonimizado, _ = anonymize_utterance(original, SAMPLE_RATE, sesion)
    razon_rms = np.sqrt(np.mean(anonimizado ** 2) / np.mean(original ** 2))
    razon_pico = np.max(np.abs(anonimizado)) / np.max(np.abs(original))
    assert 0.9 <= razon_rms <= 1.1
    assert razon_pico <= 2.0