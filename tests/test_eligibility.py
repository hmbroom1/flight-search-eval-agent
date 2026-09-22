from datetime import date

from trip_agent.eligibility import check_offer, pick_best, validate_date_range

OFFER = {"price": 289, "depart_time": "2027-03-12 07:05", "return_date": "2027-03-15", "stops": 0,
         "total_duration_min": 177}


def test_budget_is_inclusive():
    assert check_offer(OFFER, 289, "2027-03-12", "2027-03-15")[0]
    assert not check_offer(OFFER, 288, "2027-03-12", "2027-03-15")[0]
    assert not check_offer(OFFER, 288.99, "2027-03-12", "2027-03-15")[0]


def test_wrong_dates_and_missing_price_fail():
    assert not check_offer(OFFER, 999, "2027-03-11", "2027-03-15")[0]
    assert not check_offer(OFFER, 999, "2027-03-12", "2027-03-16")[0]
    ok, reasons = check_offer({**OFFER, "price": None}, 999, "2027-03-12", "2027-03-15")
    assert not ok and "no price" in reasons[0]


def test_tie_break_prefers_fewer_stops():
    a = {**OFFER, "offer_id": "a", "stops": 1}
    b = {**OFFER, "offer_id": "b", "stops": 0}
    assert pick_best([a, b])["offer_id"] == "b"


def test_date_validation():
    assert validate_date_range("2027-03-15", "2027-03-12")
    assert validate_date_range("spring", "2027-03-12")
    assert validate_date_range("2020-01-01", "2020-01-02", today=date(2026, 1, 1))
    assert validate_date_range("2027-03-12", "2027-03-15") is None


def test_unquoted_yaml_dates_become_strings(tmp_path):
    from trip_agent.eligibility import load_scenarios

    p = tmp_path / "s.yaml"
    p.write_text("scenarios:\n  - id: x\n    depart_date: 2027-03-12\n    return_date: 2027-03-15\n")
    sc = load_scenarios(p)[0]
    assert sc["depart_date"] == "2027-03-12" and sc["return_date"] == "2027-03-15"
