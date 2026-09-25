"""HTTP 层测试：三个接口、结构化错误信封、示范工况。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


INLET_RH = {"t_db_c": 35.0, "p_pa": 101325.0, "rh": 0.5}
INLET_DP = {"t_db_c": 35.0, "p_pa": 101325.0, "t_dp_c": 23.020373458966787}


def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ---- /state ---------------------------------------------------------------

def test_state_expand_rh():
    r = client.post("/state", json=INLET_RH)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["w"] == pytest.approx(0.0177655, rel=1e-5)
    assert body["rh"] == pytest.approx(0.5, abs=1e-12)
    assert body["t_dp_c"] == pytest.approx(23.02, abs=0.05)
    assert body["t_wb_c"] == pytest.approx(26.07, abs=0.05)
    assert body["w_g_per_kg"] == pytest.approx(body["w"] * 1000.0, abs=1e-12)


def test_state_rh_and_dewpoint_agree():
    a = client.post("/state", json=INLET_RH).json()
    b = client.post("/state", json=INLET_DP).json()
    assert a["w"] == pytest.approx(b["w"], abs=1e-10)


def test_state_supersaturated_structured_error():
    r = client.post("/state", json={"t_db_c": 30.0, "p_pa": 101325.0, "w": 0.05})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_state"
    assert "饱和" in err["message"]


def test_state_duplicate_humidity_fields_422():
    r = client.post("/state", json={
        "t_db_c": 30.0, "rh": 0.5, "w": 0.01,
    })
    assert r.status_code == 422


def test_state_unknown_field_rejected():
    r = client.post("/state", json={"t_db_c": 30.0, "rh": 0.5, "bogus": 1})
    assert r.status_code == 422


# ---- /coil ----------------------------------------------------------------

def test_coil_direct_demo_shape():
    r = client.post("/coil", json={"inlet": INLET_RH, "t_adp_c": 7.0, "bf": 0.2})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "direct"
    assert b["bf"] == pytest.approx(0.2)
    assert b["adp"]["t_db_c"] == pytest.approx(7.0, abs=1e-9)
    assert b["adp"]["rh"] == pytest.approx(1.0, abs=1e-9)
    # 手算量级：出风 ~12.7 °C、W ~8.5 g/kg、总冷量 ~46.5 kJ/kg
    assert b["outlet"]["t_db_c"] == pytest.approx(12.7, abs=0.2)
    assert b["outlet"]["w_g_per_kg"] == pytest.approx(8.52, abs=0.1)
    assert b["loads"]["q_total"] == pytest.approx(46.5, abs=0.3)
    # 显热 + 潜热 = 总冷量
    assert b["loads"]["q_sensible"] + b["loads"]["q_latent"] == \
        pytest.approx(b["loads"]["q_total"], abs=1e-9)
    assert 0.0 < b["loads"]["shr"] <= 1.0


def test_coil_bf_endpoints():
    r0 = client.post("/coil", json={"inlet": INLET_RH, "t_adp_c": 7.0, "bf": 0.0})
    assert r0.status_code == 200
    b0 = r0.json()
    assert b0["outlet"]["t_db_c"] == pytest.approx(7.0, abs=1e-10)
    assert b0["outlet"]["w"] == pytest.approx(b0["adp"]["w"], abs=1e-12)
    assert b0["outlet"]["rh"] == pytest.approx(1.0, abs=1e-9)

    r1 = client.post("/coil", json={"inlet": INLET_RH, "t_adp_c": 7.0, "bf": 1.0})
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["outlet"]["t_db_c"] == pytest.approx(35.0, abs=1e-10)
    assert b1["outlet"]["w"] == pytest.approx(b1["inlet"]["w"], abs=1e-12)
    assert b1["loads"]["q_total"] == pytest.approx(0.0, abs=1e-12)
    assert b1["loads"]["shr"] is None


def test_coil_inverse_modes_via_http():
    # 模式2：完整出风
    direct = client.post("/coil", json={
        "inlet": INLET_RH, "t_adp_c": 7.0, "bf": 0.2}).json()
    out = direct["outlet"]
    r2 = client.post("/coil", json={"inlet": INLET_RH, "outlet": {
        "t_db_c": out["t_db_c"], "p_pa": 101325.0, "w": out["w"]}})
    assert r2.status_code == 200, r2.text
    b2 = r2.json()
    assert b2["mode"] == "from_outlet"
    assert b2["adp"]["t_db_c"] == pytest.approx(7.0, abs=1e-5)
    assert b2["bf"] == pytest.approx(0.2, abs=1e-5)

    # 模式3：目标温度 + SHR（取物理可行的一对）
    r3 = client.post("/coil", json={
        "inlet": INLET_RH, "target_t_out_c": 20.0, "target_shr": 0.60})
    assert r3.status_code == 200, r3.text
    b3 = r3.json()
    assert b3["mode"] == "target_t_shr"
    assert b3["outlet"]["t_db_c"] == pytest.approx(20.0, abs=1e-7)
    assert b3["loads"]["shr"] == pytest.approx(0.60, abs=1e-7)

    # 模式4
    r4 = client.post("/coil", json={
        "inlet": INLET_RH, "bf": 0.25, "target_t_out_c": 14.0})
    assert r4.status_code == 200
    assert r4.json()["outlet"]["t_db_c"] == pytest.approx(14.0, abs=1e-7)

    # 模式5
    r5 = client.post("/coil", json={
        "inlet": INLET_RH, "bf": 0.25, "target_shr": 0.6})
    assert r5.status_code == 200
    assert r5.json()["loads"]["shr"] == pytest.approx(0.6, abs=1e-7)


def test_coil_underdetermined_422():
    for payload in (
        {"inlet": INLET_RH, "target_t_out_c": 12.0},
        {"inlet": INLET_RH, "target_shr": 0.5},
        {"inlet": INLET_RH, "bf": 0.2},
        {"inlet": INLET_RH},
    ):
        r = client.post("/coil", json=payload)
        assert r.status_code == 422, payload
        assert r.json()["error"]["code"] == "invalid_request"


def test_coil_invalid_bf_422():
    r = client.post("/coil", json={
        "inlet": INLET_RH, "t_adp_c": 7.0, "bf": 1.3})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_bypass_factor"


def test_coil_adp_above_wetbulb_422():
    r = client.post("/coil", json={
        "inlet": INLET_RH, "t_adp_c": 28.0, "bf": 0.2})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "not_dehumidifying"


def test_coil_inlet_representation_does_not_change_outlet():
    a = client.post("/coil", json={
        "inlet": INLET_RH, "t_adp_c": 7.0, "bf": 0.2}).json()
    b = client.post("/coil", json={
        "inlet": INLET_DP, "t_adp_c": 7.0, "bf": 0.2}).json()
    assert a["outlet"]["t_db_c"] == pytest.approx(b["outlet"]["t_db_c"], abs=1e-8)
    assert a["outlet"]["w"] == pytest.approx(b["outlet"]["w"], abs=1e-10)
    assert a["loads"]["q_total"] == pytest.approx(b["loads"]["q_total"], abs=1e-8)


def test_coil_mass_flow_param():
    r = client.post("/coil", json={
        "inlet": INLET_RH, "t_adp_c": 7.0, "bf": 0.2, "m_da": 2.5})
    assert r.status_code == 200
    b = r.json()
    assert b["loads"]["m_da"] == 2.5
    assert b["loads"]["q_total"] == pytest.approx(46.512 * 2.5, rel=1e-3)


# ---- /loads ---------------------------------------------------------------

def test_loads_endpoint_matches_coil_breakdown():
    coil = client.post("/coil", json={
        "inlet": INLET_RH, "t_adp_c": 7.0, "bf": 0.2, "m_da": 3.0}).json()
    r = client.post("/loads", json={
        "inlet": INLET_RH,
        "outlet": {"t_db_c": coil["outlet"]["t_db_c"],
                   "p_pa": 101325.0, "w": coil["outlet"]["w"]},
        "m_da": 3.0,
    })
    assert r.status_code == 200, r.text
    lb = r.json()
    assert lb["q_total"] == pytest.approx(coil["loads"]["q_total"], rel=1e-9)
    assert lb["q_sensible"] == pytest.approx(coil["loads"]["q_sensible"], rel=1e-9)
    assert lb["q_latent"] == pytest.approx(coil["loads"]["q_latent"], rel=1e-9)
    assert lb["shr"] == pytest.approx(coil["loads"]["shr"], abs=1e-9)


def test_loads_rejects_non_cooling():
    r = client.post("/loads", json={
        "inlet": {"t_db_c": 30.0, "rh": 0.5},
        "outlet": {"t_db_c": 32.0, "rh": 0.5},
    })
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_shr"


# ---- /demo ----------------------------------------------------------------

def test_demo_endpoint():
    r = client.get("/demo")
    assert r.status_code == 200
    b = r.json()
    assert b["inlet"]["t_db_c"] == 35.0
    assert b["inlet"]["rh"] == pytest.approx(0.5, abs=1e-12)
    assert b["adp"]["t_db_c"] == pytest.approx(7.0)
    assert b["bf"] == pytest.approx(0.2)
    assert 0.0 < b["loads"]["shr"] <= 1.0
    assert b["loads"]["q_sensible"] + b["loads"]["q_latent"] == \
        pytest.approx(b["loads"]["q_total"], abs=1e-9)


def test_demo_hand_check_ballpark():
    """示范工况必须能手算大致对上。"""
    b = client.get("/demo").json()
    # 进口 ~17.8 g/kg，~80.8 kJ/kg；出口 ~8.5 g/kg，~34.3 kJ/kg
    assert b["inlet"]["w_g_per_kg"] == pytest.approx(17.8, abs=0.3)
    assert b["inlet"]["enthalpy"] == pytest.approx(80.8, abs=0.3)
    assert b["outlet"]["w_g_per_kg"] == pytest.approx(8.5, abs=0.3)
    assert b["outlet"]["enthalpy"] == pytest.approx(34.3, abs=0.3)
    assert b["loads"]["q_total"] == pytest.approx(46.5, abs=0.5)
