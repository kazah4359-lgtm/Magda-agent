import pytest
from magda_agent.attention.salience import SalienceNetwork


def test_noisy_low_value_input():
    salience_network = SalienceNetwork()

    event_empty = {"content": ""}
    score_empty, explain_empty = salience_network.score_event(event_empty)
    assert score_empty == 0.0
    assert "noisy or empty input" in explain_empty

    event_whitespace = {"content": "   "}
    score_whitespace, explain_whitespace = salience_network.score_event(event_whitespace)
    assert score_whitespace == 0.0
    assert "noisy or empty input" in explain_whitespace

    event_short = {"content": "a"}
    score_short, explain_short = salience_network.score_event(event_short)
    assert score_short == 0.0
    assert "noisy or empty input" in explain_short


def test_failed_ci_event_receives_high_salience():
    salience_network = SalienceNetwork()

    event_ci_flag = {"content": "normal log", "is_ci_failure": True}
    score_ci_flag, explain_ci_flag = salience_network.score_event(event_ci_flag)
    assert score_ci_flag >= 0.7  # 0.1 base + 0.6 CI
    assert "CI/test failure" in explain_ci_flag

    event_ci_keyword = {"content": "the ci failed on master"}
    score_ci_keyword, explain_ci_keyword = salience_network.score_event(event_ci_keyword)
    assert score_ci_keyword >= 0.7
    assert "CI/test failure" in explain_ci_keyword


def test_security_risk_event_receives_high_salience():
    salience_network = SalienceNetwork()

    event_sec_flag = {"content": "normal log", "is_security_risk": True}
    score_sec_flag, explain_sec_flag = salience_network.score_event(event_sec_flag)
    assert score_sec_flag >= 0.9  # 0.1 base + 0.8 security
    assert "security risk" in explain_sec_flag

    event_sec_keyword = {"content": "found a vulnerability in dependency"}
    score_sec_keyword, explain_sec_keyword = salience_network.score_event(event_sec_keyword)
    assert score_sec_keyword >= 0.9
    assert "security risk" in explain_sec_keyword


def test_score_clamped_to_one():
    salience_network = SalienceNetwork()

    # Event with multiple high-salience triggers
    event_max = {
        "content": "urgent security vulnerability ci failed",
        "is_security_risk": True,
        "is_ci_failure": True
    }
    score_max, _ = salience_network.score_event(event_max)
    assert score_max == 1.0


def test_normal_input_salience():
    salience_network = SalienceNetwork()

    event_normal = {"content": "hello how are you"}
    score_normal, explain_normal = salience_network.score_event(event_normal)
    assert score_normal == 0.1
    assert "base score" in explain_normal
