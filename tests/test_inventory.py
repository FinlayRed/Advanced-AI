from src.quality.inventory import ProducerInventory


def test_inventory_updates_per_producer():
    inv = ProducerInventory()
    inv.process_inspection(
        producer_id="p1",
        product_type="apple",
        quantity=10,
        color_score=90,
        size_score=92,
        ripeness_score=88,
        model_confidence=0.9,
    )
    inv.process_inspection(
        producer_id="p2",
        product_type="apple",
        quantity=5,
        color_score=90,
        size_score=92,
        ripeness_score=88,
        model_confidence=0.9,
    )

    assert inv.get_total_stock("p1", "apple") == 10
    assert inv.get_total_stock("p2", "apple") == 5


def test_low_confidence_returns_manual_review():
    inv = ProducerInventory()
    rec = inv.process_inspection(
        producer_id="p1",
        product_type="tomato",
        quantity=10,
        color_score=90,
        size_score=90,
        ripeness_score=90,
        model_confidence=0.2,
    )
    assert rec["action"] == "manual_review"


def test_grade_b_with_high_stock_discounts():
    inv = ProducerInventory()
    rec = inv.process_inspection(
        producer_id="p1",
        product_type="banana",
        quantity=120,
        color_score=74,
        size_score=90,
        ripeness_score=90,
        model_confidence=0.95,
        surplus_threshold=100,
    )
    assert rec["grade"] == "B"
    assert rec["action"] == "discount_10_20"

