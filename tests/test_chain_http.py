"""多级串联链 HTTP 接口测试：/chain 路由、结构化错误信封、级次定位。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

INLET = {"t_db_c": 35.0, "p_pa": 101325.0, "rh": 0.5}

#: 温和工况三级链（各级出口均远离饱和线）
THREE_STAGE_DIRECT = [
    {"name": "预冷", "t_adp_c": 10.0, "bf": 0.5},
    {"name": "深除湿", "t_adp_c": 8.0, "bf": 0.5},
    {"name": "调温", "t_adp_c": 6.5, "bf": 0.5},
]


def _forward_chain():
    r = client.post("/chain", json={"inlet": INLET, "stages": THREE_STAGE_DIRECT})
    assert r.status_code == 200, r.text
    return r.json()


# ---- 正向链 -----------------------------------------------------------------

def test_chain_forward_three_stage_shape():
    b = _forward_chain()
    assert b["mode"] == "forward"
    assert len(b["stages"]) == 3
    # 级名透传、级次从 1 起计
    assert [s["index"] for s in b["stages"]] == [1, 2, 3]
    assert [s["name"] for s in b["stages"]] == ["预冷", "深除湿", "调温"]
    # 链出风 == 最后一级出口；链进口 == 第一级进口
    last = b["stages"][-1]
    assert b["outlet"]["t_db_c"] == pytest.approx(last["outlet"]["t_db_c"])
    assert b["outlet"]["w"] == pytest.approx(last["outlet"]["w"])
    assert b["inlet"]["t_db_c"] == pytest.approx(35.0)
    # 级间传递：后一级进口逐位等于前一级出口
    for prev, nxt in zip(b["stages"], b["stages"][1:]):
        assert nxt["inlet"]["t_db_c"] == prev["outlet"]["t_db_c"]
        assert nxt["inlet"]["w"] == prev["outlet"]["w"]
        assert nxt["inlet"]["enthalpy"] == prev["outlet"]["enthalpy"]
        assert nxt["inlet"]["p_pa"] == prev["outlet"]["p_pa"]
    # 沿链温度、含湿量单调
    ts = [b["inlet"]["t_db_c"]] + [s["outlet"]["t_db_c"] for s in b["stages"]]
    ws = [b["inlet"]["w"]] + [s["outlet"]["w"] for s in b["stages"]]
    assert all(a >= b2 for a, b2 in zip(ts, ts[1:]))
    assert all(a >= b2 for a, b2 in zip(ws, ws[1:]))
    # 全链合计 == 各级之和；份额归一
    sum_q = sum(s["loads"]["q_total"] for s in b["stages"])
    assert b["totals"]["q_total"] == pytest.approx(sum_q, abs=1e-9)
    assert sum(s["load_share"] for s in b["stages"]) == pytest.approx(1.0)
    assert b["totals"]["q_sensible"] + b["totals"]["q_latent"] == \
        pytest.approx(b["totals"]["q_total"], abs=1e-9)
    assert 0.0 < b["totals"]["shr"] <= 1.0
    # 总去湿量 == m_da·(首 W − 末 W)
    assert b["totals"]["dehumidification"] == pytest.approx(
        b["totals"]["m_da"] * (b["inlet"]["w"] - b["outlet"]["w"]), abs=1e-12
    )


def test_chain_single_stage_matches_coil_endpoint():
    """一级链的每一项都要和 /coil 对同样输入的回答一致。"""
    payload = {"inlet": INLET, "t_adp_c": 7.0, "bf": 0.2, "m_da": 2.0}
    coil = client.post("/coil", json=payload).json()
    r = client.post("/chain", json={
        "inlet": INLET, "m_da": 2.0,
        "stages": [{"t_adp_c": 7.0, "bf": 0.2}],
    })
    assert r.status_code == 200, r.text
    b = r.json()
    st = b["stages"][0]
    assert st["outlet"]["t_db_c"] == pytest.approx(coil["outlet"]["t_db_c"])
    assert st["outlet"]["w"] == pytest.approx(coil["outlet"]["w"])
    assert st["adp"]["t_db_c"] == pytest.approx(coil["adp"]["t_db_c"])
    assert st["bf"] == pytest.approx(coil["bf"])
    assert st["loads"]["q_total"] == pytest.approx(coil["loads"]["q_total"])
    assert b["totals"]["q_total"] == pytest.approx(coil["loads"]["q_total"])
    assert b["totals"]["shr"] == pytest.approx(coil["loads"]["shr"])


def test_chain_mixed_modes_via_http():
    fwd = _forward_chain()
    mixed = [
        {"t_adp_c": 10.0, "bf": 0.5},
        {"outlet": {"t_db_c": fwd["stages"][1]["outlet"]["t_db_c"],
                    "p_pa": 101325.0, "w": fwd["stages"][1]["outlet"]["w"]}},
        {"bf": 0.5, "target_t_out_c": fwd["stages"][2]["outlet"]["t_db_c"]},
    ]
    r = client.post("/chain", json={"inlet": INLET, "stages": mixed})
    assert r.status_code == 200, r.text
    b = r.json()
    assert [s["mode"] for s in b["stages"]] == ["direct", "from_outlet",
                                                "bf_target_t"]
    for got, want in zip(b["stages"], fwd["stages"]):
        assert got["outlet"]["t_db_c"] == pytest.approx(
            want["outlet"]["t_db_c"], abs=1e-6)


def test_chain_passthrough_stage_via_http():
    r = client.post("/chain", json={"inlet": INLET, "stages": [
        {"t_adp_c": 10.0, "bf": 0.5},
        {"t_adp_c": 8.0, "bf": 1.0},
        {"t_adp_c": 6.5, "bf": 0.5},
    ]})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["stages"][1]["passthrough"] is True
    assert b["stages"][1]["loads"]["q_total"] == pytest.approx(0.0, abs=1e-12)
    assert b["stages"][1]["load_share"] == pytest.approx(0.0, abs=1e-12)


# ---- 全链目标反推 -------------------------------------------------------------

def test_chain_inverse_roundtrip_via_http():
    fwd = _forward_chain()
    target_state = {"t_db_c": fwd["outlet"]["t_db_c"], "p_pa": 101325.0,
                    "w": fwd["outlet"]["w"]}
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": [{"t_adp_c": 10.0, "bf": 0.5},
                   {},
                   {"t_adp_c": 6.5, "bf": 0.5}],
        "target": {"outlet": target_state},
    })
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "inverse"
    assert b["stages"][1]["adp"]["t_db_c"] == pytest.approx(8.0, abs=1e-5)
    assert b["stages"][1]["bf"] == pytest.approx(0.5, abs=1e-6)
    assert b["outlet"]["t_db_c"] == pytest.approx(
        fwd["outlet"]["t_db_c"], abs=1e-6)


def test_chain_inverse_with_t_shr_target_via_http():
    fwd = _forward_chain()
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": [{"t_adp_c": 10.0, "bf": 0.5},
                   {},
                   {"t_adp_c": 6.5, "bf": 0.5}],
        "target": {"target_t_out_c": fwd["outlet"]["t_db_c"],
                   "target_shr": fwd["totals"]["shr"]},
    })
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["mode"] == "inverse"
    assert b["stages"][1]["adp"]["t_db_c"] == pytest.approx(8.0, abs=1e-5)
    assert b["totals"]["shr"] == pytest.approx(fwd["totals"]["shr"], abs=1e-7)


def test_chain_verify_mode_via_http():
    fwd = _forward_chain()
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": THREE_STAGE_DIRECT,
        "target": {"outlet": {"t_db_c": fwd["outlet"]["t_db_c"],
                              "p_pa": 101325.0, "w": fwd["outlet"]["w"]}},
    })
    assert r.status_code == 200, r.text
    assert r.json()["mode"] == "verify"


# ---- 结构化错误（同一信封，带级次） ---------------------------------------------

def _assert_error(r, code, stage_index=None):
    assert r.status_code == 422, r.text
    err = r.json()["error"]
    assert err["code"] == code
    assert err["stage_index"] == stage_index
    return err


def test_chain_empty_rejected():
    r = client.post("/chain", json={"inlet": INLET, "stages": []})
    err = _assert_error(r, "invalid_request")
    assert "空链" in err["message"]


def test_chain_stage_adp_above_wet_bulb_located_via_http():
    r = client.post("/chain", json={"inlet": INLET, "stages": [
        {"t_adp_c": 10.0, "bf": 0.5},
        {"t_adp_c": 19.5, "bf": 0.5},
    ]})
    err = _assert_error(r, "not_dehumidifying", stage_index=2)
    assert "第 2 级" in err["message"]


def test_chain_all_passthrough_rejected_via_http():
    r = client.post("/chain", json={"inlet": INLET, "stages": [
        {"t_adp_c": 10.0, "bf": 1.0},
        {"t_adp_c": 8.0, "bf": 1.0},
    ]})
    _assert_error(r, "not_dehumidifying")


def test_chain_target_wetter_than_inlet_rejected_via_http():
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": [{"t_adp_c": 10.0, "bf": 0.5}, {}],
        "target": {"outlet": {"t_db_c": 34.0, "w": 0.020}},
    })
    _assert_error(r, "not_dehumidifying")


def test_chain_underdetermined_via_http():
    fwd = _forward_chain()
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": [{"t_adp_c": 10.0, "bf": 0.5}, {}, {}],
        "target": {"outlet": {"t_db_c": fwd["outlet"]["t_db_c"],
                              "p_pa": 101325.0, "w": fwd["outlet"]["w"]}},
    })
    _assert_error(r, "chain_underdetermined", stage_index=2)


def test_chain_overdetermined_via_http():
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": THREE_STAGE_DIRECT,
        "target": {"outlet": {"t_db_c": 11.5, "rh": 0.9}},
    })
    _assert_error(r, "chain_overdetermined")


def test_chain_no_convergence_via_http():
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": [{"t_adp_c": 10.0, "bf": 0.5},
                   {},
                   {"t_adp_c": 6.5, "bf": 0.5}],
        "target": {"outlet": {"t_db_c": 2.0, "w": 0.002}},
    })
    _assert_error(r, "no_convergence", stage_index=2)


def test_chain_pressure_mismatch_located_via_http():
    fwd = _forward_chain()
    r = client.post("/chain", json={
        "inlet": INLET,
        "stages": [
            {"t_adp_c": 10.0, "bf": 0.5},
            {"outlet": {"t_db_c": fwd["stages"][1]["outlet"]["t_db_c"],
                        "p_pa": 90000.0,
                        "w": fwd["stages"][1]["outlet"]["w"]}},
        ],
    })
    _assert_error(r, "invalid_request", stage_index=2)


def test_chain_unknown_field_rejected():
    r = client.post("/chain", json={"inlet": INLET, "stages": [
        {"t_adp_c": 10.0, "bf": 0.5, "bogus": 1},
    ]})
    assert r.status_code == 422


def test_chain_stage_conflicting_knowns_located():
    r = client.post("/chain", json={"inlet": INLET, "stages": [
        {"t_adp_c": 10.0, "bf": 0.5},
        {"t_adp_c": 8.0, "bf": 0.5, "target_shr": 0.5},
    ]})
    _assert_error(r, "invalid_request", stage_index=2)
