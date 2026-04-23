from src.quality.grading import assign_grade, condition_from_label, grade_from_condition


def test_assign_grade_a_case():
    assert assign_grade(85, 90, 80) == "A"


def test_assign_grade_b_case():
    assert assign_grade(74, 90, 80) == "B"


def test_assign_grade_c_case():
    assert assign_grade(64, 90, 80) == "C"


def test_condition_from_label_normalizes_fresh_to_healthy():
    assert condition_from_label("fresh") == "healthy"
    assert condition_from_label("Apple__Healthy") == "healthy"


def test_condition_from_label_normalizes_rotten():
    assert condition_from_label("Tomato__Rotten") == "rotten"


def test_grade_from_condition_keeps_legacy_grade_field():
    assert grade_from_condition("healthy") == "A"
    assert grade_from_condition("rotten") == "C"

