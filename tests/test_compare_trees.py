"""POST /v1/compare-trees：两版故障树差异概率与等价性的语义、精确性与 422 校验。

用穷举真值表（枚举全部事件状态）作为参考实现做交叉验证。
"""

import itertools
import random
from fractions import Fraction

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------- 穷举真值表参考实现 ----------

def brute_compare(ev_specs, gates_b, top_b, gates_r, top_r):
    """枚举全部事件赋值，返回 (P(¬B∧R), P(B∧¬R), P(B⊕R), equivalent)。

    两版共用同一份事件（同一赋值、同一先验）；门表各自独立。
    equivalent 要求全部赋值（含零概率赋值）下两版顶事件一致。
    """
    ev_ids = sorted(ev_specs)
    probs = {e: Fraction(ev_specs[e]) for e in ev_ids}
    gate_b = {g[0]: (g[1], g[2]) for g in gates_b}
    gate_r = {g[0]: (g[1], g[2]) for g in gates_r}

    def make_eval(gates):
        def evaluate(state, node):
            if node in state:
                return state[node]
            op, inputs = gates[node]
            vals = [evaluate(state, i) for i in inputs]
            if op == "AND":
                return all(vals)
            if op == "OR":
                return any(vals)
            return not vals[0]

        return evaluate

    eval_b = make_eval(gate_b)
    eval_r = make_eval(gate_r)

    added = removed = Fraction(0)
    equivalent = True
    for bits in itertools.product((0, 1), repeat=len(ev_ids)):
        state = dict(zip(ev_ids, (bool(b) for b in bits)))
        vb = eval_b(state, top_b)
        vr = eval_r(state, top_r)
        if vb == vr:
            continue
        equivalent = False
        weight = Fraction(1)
        for e in ev_ids:
            weight *= probs[e] if state[e] else (Fraction(1) - probs[e])
        if vr:
            added += weight
        else:
            removed += weight
    return added, removed, added + removed, equivalent


def f2s(f):
    return f"{f.numerator}/{f.denominator}"


def make_body(ev_specs, gates_b, top_b, gates_r, top_r):
    return {
        "events": [{"id": k, "p": v} for k, v in ev_specs.items()],
        "baseline": {
            "gates": [
                {"id": i, "op": op, "inputs": ins} for i, op, ins in gates_b
            ],
            "top": top_b,
        },
        "revised": {
            "gates": [
                {"id": i, "op": op, "inputs": ins} for i, op, ins in gates_r
            ],
            "top": top_r,
        },
    }


def post_compare(ev_specs, gates_b, top_b, gates_r, top_r):
    return client.post(
        "/v1/compare-trees",
        json=make_body(ev_specs, gates_b, top_b, gates_r, top_r),
    )


def assert_matches_brute(ev_specs, gates_b, top_b, gates_r, top_r):
    r = post_compare(ev_specs, gates_b, top_b, gates_r, top_r)
    assert r.status_code == 200, r.text
    body = r.json()
    exp_added, exp_removed, exp_change, exp_eq = brute_compare(
        ev_specs, gates_b, top_b, gates_r, top_r
    )
    assert body["added_probability"] == f2s(exp_added)
    assert body["removed_probability"] == f2s(exp_removed)
    assert body["change_probability"] == f2s(exp_change)
    assert body["equivalent"] == exp_eq
    return body


# ---------- 基本语义 ----------

def test_readme_motor_example():
    """README 示例：改版把冷却失效改为双冗余，只删不增。"""
    ev = {"motor_overheat": "1/100", "coolant_loss": "2/100", "pump_fail": "1/200"}
    gates_b = [
        ("cooling_fail", "OR", ["coolant_loss", "pump_fail"]),
        ("top", "OR", ["motor_overheat", "cooling_fail"]),
    ]
    gates_r = [
        ("cooling_fail", "AND", ["coolant_loss", "pump_fail"]),
        ("top", "OR", ["motor_overheat", "cooling_fail"]),
    ]
    body = assert_matches_brute(ev, gates_b, "top", gates_r, "top")
    # R ⊆ B：added 为 0；removed = P(¬m)·P(c⊕p) = 99/100·496/20000
    assert body["added_probability"] == "0/1"
    assert body["removed_probability"] == "3069/125000"
    assert body["change_probability"] == "3069/125000"
    assert body["equivalent"] is False


def test_two_sided_change():
    """双向差异：B=a∧b，R=a∧c，added 与 removed 都非零。"""
    ev = {"a": "1/2", "b": "1/3", "c": "1/5"}
    body = assert_matches_brute(
        ev, [("g", "AND", ["a", "b"])], "g", [("g", "AND", ["a", "c"])], "g"
    )
    # added = P(a∧¬b∧c) = 1/2·2/3·1/5 = 1/15
    assert body["added_probability"] == "1/15"
    # removed = P(a∧b∧¬c) = 1/2·1/3·4/5 = 2/15
    assert body["removed_probability"] == "2/15"
    assert body["change_probability"] == "1/5"
    assert body["equivalent"] is False


def test_same_gate_id_independent_definitions():
    """门 id 跨版同名但定义独立：g 在基线为 AND、在修订为 OR。"""
    ev = {"a": "1/2", "b": "1/3"}
    body = assert_matches_brute(
        ev, [("g", "AND", ["a", "b"])], "g", [("g", "OR", ["a", "b"])], "g"
    )
    # added = P(a⊕b) = 1/2；removed = 0
    assert body["added_probability"] == "1/2"
    assert body["removed_probability"] == "0/1"
    assert body["change_probability"] == "1/2"
    assert body["equivalent"] is False


def test_top_may_be_event_directly():
    """两版顶层都可以直接是事件。"""
    ev = {"a": "1/2", "b": "1/4"}
    body = assert_matches_brute(ev, [], "a", [], "b")
    # added = P(¬a∧b) = 1/2·1/4 = 1/8；removed = P(a∧¬b) = 1/2·3/4 = 3/8
    assert body["added_probability"] == "1/8"
    assert body["removed_probability"] == "3/8"
    assert body["change_probability"] == "1/2"
    assert body["equivalent"] is False


def test_identical_versions_are_equivalent():
    ev = {"a": "1/2", "b": "1/3"}
    gates = [("g", "OR", ["a", "b"])]
    body = assert_matches_brute(ev, gates, "g", gates, "g")
    assert body["added_probability"] == "0/1"
    assert body["removed_probability"] == "0/1"
    assert body["change_probability"] == "0/1"
    assert body["equivalent"] is True


def test_equivalent_despite_different_structure():
    """结构不同但函数等价：a ∨ (a∧b) 与 a 吸收等价。"""
    ev = {"a": "1/2", "b": "1/3"}
    gates_b = [("ab", "AND", ["a", "b"]), ("top", "OR", ["a", "ab"])]
    body = assert_matches_brute(ev, gates_b, "top", [], "a")
    assert body["added_probability"] == "0/1"
    assert body["removed_probability"] == "0/1"
    assert body["change_probability"] == "0/1"
    assert body["equivalent"] is True


def test_zero_probability_divergence_still_not_equivalent():
    """差异只出现在零概率赋值上：概率全为 0 但 equivalent 为 false。"""
    ev = {"a": "0/1", "b": "1/2"}
    # B = b，R = a∨b：仅 a=1 且 b=0 时不同，而 P(a=1)=0
    body = assert_matches_brute(ev, [], "b", [("g", "OR", ["a", "b"])], "g")
    assert body["added_probability"] == "0/1"
    assert body["removed_probability"] == "0/1"
    assert body["change_probability"] == "0/1"
    assert body["equivalent"] is False


def test_not_gates_and_shared_events():
    """NOT 门照常求值；同一事件在两版多处引用共享状态。"""
    ev = {"a": "2/7", "b": "3/11", "c": "5/13"}
    gates_b = [
        ("nb", "NOT", ["b"]),
        ("g1", "AND", ["a", "nb"]),
        ("top", "OR", ["g1", "c"]),
    ]
    gates_r = [
        ("g1", "AND", ["a", "b"]),
        ("nc", "NOT", ["c"]),
        ("top", "AND", ["g1", "nc"]),
    ]
    assert_matches_brute(ev, gates_b, "top", gates_r, "top")


def test_unused_gates_do_not_affect_result():
    """顶层不可达的门照常校验但不影响差异结果。"""
    ev = {"a": "1/2", "b": "1/3"}
    gates_b = [("g", "OR", ["a", "b"]), ("unused", "NOT", ["a"])]
    gates_r = [("h", "AND", ["a", "b"]), ("unused2", "NOT", ["b"])]
    assert_matches_brute(ev, gates_b, "g", gates_r, "h")


def test_random_tree_pairs_cross_validation():
    """随机生成两版故障树（共享事件），逐例与穷举对照。"""
    rng = random.Random(20260927)
    for _ in range(30):
        n = rng.randint(1, 9)
        ev_ids = [f"e{i}" for i in range(n)]
        ev_specs = {}
        for e in ev_ids:
            d = rng.randint(1, 8)
            ev_specs[e] = f"{rng.randint(0, d)}/{d}"

        def random_gates():
            gates = []
            available = list(ev_ids)
            for j in range(rng.randint(0, 10)):
                gid = f"g{j}"
                if len(available) >= 2:
                    op = rng.choice(["AND", "OR", "NOT"])
                else:
                    op = "NOT"
                if op == "NOT":
                    inputs = [rng.choice(available)]
                else:
                    inputs = rng.sample(available, rng.randint(2, min(8, len(available))))
                gates.append((gid, op, inputs))
                available.append(gid)
            top = rng.choice(available)
            return gates, top

        gates_b, top_b = random_gates()
        gates_r, top_r = random_gates()
        assert_matches_brute(ev_specs, gates_b, top_b, gates_r, top_r)


# ---------- 422 错误路径 ----------

def test_unknown_reference_in_either_version():
    ev = {"a": "1/2"}
    # 基线版引用未知节点
    r = post_compare(ev, [("g", "AND", ["a", "x"])], "g", [], "a")
    assert r.status_code == 422 and "未知" in r.json()["detail"]
    # 修订版引用未知节点
    r = post_compare(ev, [], "a", [("g", "OR", ["a", "y"])], "g")
    assert r.status_code == 422 and "未知" in r.json()["detail"]
    # 修订版顶层未知
    r = post_compare(ev, [], "a", [], "nope")
    assert r.status_code == 422 and "顶层" in r.json()["detail"]


def test_cross_version_reference_rejected():
    """基线版不能引用只在修订版定义的门（反之亦然）。"""
    ev = {"a": "1/2"}
    r = post_compare(
        ev,
        [("g1", "NOT", ["g2"])], "g1",   # g2 只在修订版定义
        [("g2", "NOT", ["a"])], "g2",
    )
    assert r.status_code == 422 and "未知" in r.json()["detail"]


def test_cycle_in_either_version():
    ev = {"a": "1/2"}
    r = post_compare(ev, [("g", "OR", ["a", "g"])], "g", [], "a")
    assert r.status_code == 422 and "环" in r.json()["detail"]
    r = post_compare(
        ev, [], "a",
        [("g1", "OR", ["a", "g2"]), ("g2", "AND", ["a", "g1"])], "g1",
    )
    assert r.status_code == 422 and "环" in r.json()["detail"]


def test_bad_probability_rejected():
    r = post_compare({"a": "3/2"}, [], "a", [], "a")
    assert r.status_code == 422 and "概率" in r.json()["detail"]


def test_gate_id_clashes_with_event_in_either_version():
    ev = {"a": "1/2", "b": "1/3"}
    r = post_compare(ev, [("a", "NOT", ["b"])], "a", [], "b")
    assert r.status_code == 422 and "冲突" in r.json()["detail"]
    r = post_compare(ev, [], "b", [("b", "NOT", ["a"])], "b")
    assert r.status_code == 422 and "冲突" in r.json()["detail"]


def test_duplicate_gate_id_within_version():
    ev = {"a": "1/2", "b": "1/3"}
    r = client.post("/v1/compare-trees", json={
        "events": [{"id": "a", "p": "1/2"}, {"id": "b", "p": "1/3"}],
        "baseline": {
            "gates": [
                {"id": "g", "op": "AND", "inputs": ["a", "b"]},
                {"id": "g", "op": "OR", "inputs": ["a", "b"]},
            ],
            "top": "g",
        },
        "revised": {"gates": [], "top": "a"},
    })
    assert r.status_code == 422 and "重复" in r.json()["detail"]


def test_bad_arity_in_either_version():
    ev = {"a": "1/2", "b": "1/3"}
    r = post_compare(ev, [("g", "NOT", ["a", "b"])], "g", [], "a")
    assert r.status_code == 422 and "NOT" in r.json()["detail"]
    r = post_compare(ev, [], "a", [("g", "AND", ["a"])], "g")
    assert r.status_code == 422


def test_request_structure_errors():
    ev = [{"id": "a", "p": "1/2"}]
    # 缺 revised
    r = client.post("/v1/compare-trees", json={
        "events": ev, "baseline": {"gates": [], "top": "a"}})
    assert r.status_code == 422
    # 缺 top
    r = client.post("/v1/compare-trees", json={
        "events": ev,
        "baseline": {"gates": [], "top": "a"},
        "revised": {"gates": []},
    })
    assert r.status_code == 422
    # 顶层额外字段
    r = client.post("/v1/compare-trees", json={
        "events": ev,
        "baseline": {"gates": [], "top": "a"},
        "revised": {"gates": [], "top": "a"},
        "extra": 1,
    })
    assert r.status_code == 422
    # 版本内额外字段
    r = client.post("/v1/compare-trees", json={
        "events": ev,
        "baseline": {"gates": [], "top": "a", "states": {}},
        "revised": {"gates": [], "top": "a"},
    })
    assert r.status_code == 422
    # 0 个事件 / 33 个事件
    r = client.post("/v1/compare-trees", json={
        "events": [],
        "baseline": {"gates": [], "top": "a"},
        "revised": {"gates": [], "top": "a"},
    })
    assert r.status_code == 422
    r = client.post("/v1/compare-trees", json={
        "events": [{"id": f"e{i}", "p": "1/2"} for i in range(33)],
        "baseline": {"gates": [], "top": "e0"},
        "revised": {"gates": [], "top": "e0"},
    })
    assert r.status_code == 422
    # 单版 65 个门
    r = client.post("/v1/compare-trees", json={
        "events": ev,
        "baseline": {
            "gates": [
                {"id": f"g{i}", "op": "NOT", "inputs": ["a"]} for i in range(65)
            ],
            "top": "g0",
        },
        "revised": {"gates": [], "top": "a"},
    })
    assert r.status_code == 422


def test_node_limit_counts_both_versions_and_diff(monkeypatch):
    """两版建图及差异运算累计超限返回 422 并说明原因。"""
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 40)
    evs = {f"e{i}": "1/2" for i in range(8)}

    def xor_chain():
        gates = []
        prev = "e0"
        for i in range(1, 8):
            o, a, n, x = f"o{i}", f"a{i}", f"n{i}", f"x{i}"
            gates.append((o, "OR", [prev, f"e{i}"]))
            gates.append((a, "AND", [prev, f"e{i}"]))
            gates.append((n, "NOT", [a]))
            gates.append((x, "AND", [o, n]))
            prev = x
        return gates, prev

    gates_b, top_b = xor_chain()
    gates_r, top_r = xor_chain()
    r = post_compare(evs, gates_b, top_b, gates_r, top_r)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "非终端节点" in detail and "40" in detail


def test_node_limit_message_mentions_limit(monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "MAX_NONTERMINAL_NODES", 5)
    evs = {f"e{i}": "1/2" for i in range(8)}
    gates = []
    prev = "e0"
    for i in range(1, 8):
        o, a, n, x = f"o{i}", f"a{i}", f"n{i}", f"x{i}"
        gates.append((o, "OR", [prev, f"e{i}"]))
        gates.append((a, "AND", [prev, f"e{i}"]))
        gates.append((n, "NOT", [a]))
        gates.append((x, "AND", [o, n]))
        prev = x
    r = post_compare(evs, gates, prev, gates, prev)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "非终端节点" in detail and "5" in detail
