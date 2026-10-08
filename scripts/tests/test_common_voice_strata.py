from common_voice_strata import assign_stratum
import pytest

import itertools

import pytest

from common_voice_strata import assign_stratum

ANDINO = "Andino-Pacífico: Colombia, Perú, Ecuador, oeste de Bolivia y Venezuela andina"
NORTE = "España: Norte peninsular (Asturias, Castilla y León, Cantabria, País Vasco, Navarra, Aragón, La Rioja, Guadalajara, Cuenca)"
RIOPLATENSE = "Rioplatense: Argentina, Uruguay, este de Bolivia, Paraguay"
CARIBE = "Caribe: Cuba, Venezuela, Puerto Rico, República Dominicana, Panamá, Colombia caribeña, México caribeño, Costa del golfo de México"
CENTRO_SUR = "España: Centro-Sur peninsular (Madrid, Toledo, Castilla-La Mancha)"
CHILENO = "Chileno: Chile, Cuyo"
SUR = "España: Sur peninsular (Andalucia, Extremadura, Murcia)"
CANARIAS = "España: Islas Canarias"
COMBINADO = ANDINO + "|Acento costeño de Lima Perú"


@pytest.mark.parametrize("accents, expected", [
    # Sin etiqueta
    pytest.param([], "sin_dato", id="lista_vacia"),
    pytest.param([""], "sin_dato", id="un_vacio"),
    pytest.param([" ", ""], "sin_dato", id="solo_espacios_y_vacios"),
    pytest.param([None], "sin_dato", id="solo_none"),
    pytest.param([None, "", " "], "sin_dato", id="none_vacio_y_espacios"),

    # Una etiqueta de cada uno de los nueve estratos
    pytest.param(["México"], "mexico", id="mexico"),
    pytest.param([ANDINO], "andino_pacifico", id="andino_pacifico"),
    pytest.param([NORTE], "espana_norte", id="espana_norte"),
    pytest.param([RIOPLATENSE], "rioplatense", id="rioplatense"),
    pytest.param([CARIBE], "caribe", id="caribe"),
    pytest.param([CENTRO_SUR], "espana_centro_sur", id="espana_centro_sur"),
    pytest.param(["América central"], "america_central", id="america_central"),
    pytest.param([CHILENO], "chileno", id="chileno"),
    pytest.param([SUR], "espana_sur", id="espana_sur"),

    # Limpieza y duplicados
    pytest.param(["  México  "], "mexico", id="espacios_alrededor"),
    pytest.param(["México", "México"], "mexico", id="duplicados_colapsan"),

    # Etiqueta mezclada con clips sin etiqueta
    pytest.param(["México", ""], "mexico", id="etiqueta_mas_vacio"),
    pytest.param(["", "México", " "], "mexico", id="vacios_alrededor"),
    pytest.param(["México", None], "mexico", id="etiqueta_mas_none"),

    # Conflicto entre etiquetas distintas
    pytest.param(["México", CHILENO], "otro", id="dos_validas_distintas"),
    pytest.param(["México", "México", ANDINO], "otro", id="conflicto_con_duplicados"),
    pytest.param(["México", CANARIAS], "otro", id="valida_mas_desconocida"),

    # Una sola etiqueta que no está entre los nueve
    pytest.param([CANARIAS], "otro", id="desconocida_canarias"),
    pytest.param(["Costa rica"], "otro", id="texto_libre"),
    pytest.param([COMBINADO], "otro", id="valor_combinado_con_barra"),
])
def test_assign_stratum(accents, expected):
    assert assign_stratum(accents) == expected


def test_assign_stratum_no_depende_del_orden():
    clips = ["México", "", None, "México"]
    resultados = {assign_stratum(list(p)) for p in itertools.permutations(clips)}
    assert resultados == {"mexico"}


def test_assign_stratum_acepta_generador():
    assert assign_stratum(a for a in ["México", ""]) == "mexico"