from utils import stable_id


def test_stable_id_deterministic():
    assert stable_id("https://arxiv.org/abs/1234.5678") == stable_id("https://arxiv.org/abs/1234.5678")


def test_stable_id_distinct():
    assert stable_id("a") != stable_id("b")


def test_stable_id_length():
    assert len(stable_id("anything")) == 16
    assert len(stable_id("anything", length=8)) == 8
