from src.quality.grading import assign_grade


def test_assign_grade_a_case():
    assert assign_grade(85, 90, 80) == "A"


def test_assign_grade_b_case():
    assert assign_grade(74, 90, 80) == "B"


def test_assign_grade_c_case():
    assert assign_grade(64, 90, 80) == "C"

