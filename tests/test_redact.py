"""Redaction tests use synthetic, obviously-fake secrets — never real ones."""
from code_rag.redact import redact_text


def test_telegram_token_is_redacted():
    # 9 digits + ":" + exactly 35 chars, matching the real Telegram token shape.
    text = 'BOT_TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"'
    out, counts = redact_text(text)
    assert "123456789" not in out
    assert "[REDACTED:telegram_token]" in out
    assert counts["telegram_token"] == 1


def test_openai_style_key_is_redacted():
    text = 'OPENAI_API_KEY = "sk-fake0123456789abcdefFAKEFAKEFAKE"'
    out, counts = redact_text(text)
    assert "sk-fake0123456789" not in out
    assert "[REDACTED:openai_key]" in out
    assert counts["openai_key"] == 1


def test_google_api_key_is_redacted():
    text = 'GOOGLE_API_KEY = "AIzaFAKE0123456789abcdefFAKEFAKEFAKEFAK"'
    out, counts = redact_text(text)
    assert "AIzaFAKE" not in out
    assert "[REDACTED:google_api_key]" in out


def test_github_token_is_redacted():
    # ghp_ + exactly 36 alnum chars, matching the real GitHub token shape.
    text = "token = ghp_FAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKEFAKE"
    out, counts = redact_text(text)
    assert "ghp_FAKE" not in out
    assert "[REDACTED:github_token]" in out


def test_bearer_token_is_redacted():
    text = "Authorization: Bearer fake.jwt.header.payload.signaturefakefake"
    out, counts = redact_text(text)
    assert "[REDACTED:bearer_token]" in out
    assert counts["bearer_token"] == 1


def test_generic_credential_assignment_is_redacted():
    text = 'DB_PASSWORD = "SuperFakePassword1234567890"'
    out, counts = redact_text(text)
    assert "SuperFakePassword1234567890" not in out
    assert "[REDACTED:credential]" in out
    # The variable name itself must survive so the chunk stays readable.
    assert "DB_PASSWORD" in out


def test_phone_number_is_redacted():
    text = "Позвони мне: +7 999 123 45 67 после обеда"
    out, counts = redact_text(text)
    assert "999 123 45 67" not in out
    assert "[REDACTED:phone]" in out


def test_ip_address_is_redacted():
    text = "proxy host 192.168.100.42 port 1080"
    out, counts = redact_text(text)
    assert "192.168.100.42" not in out
    assert "[REDACTED:ip_address]" in out


def test_ordinary_code_is_left_untouched():
    text = "def add(a, b):\n    return a + b\n"
    out, counts = redact_text(text)
    assert out == text
    assert sum(counts.values()) == 0


def test_counts_are_aggregated_by_kind():
    text = (
        'TOKEN_1 = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"\n'
        'TOKEN_2 = "987654321:ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsr"\n'
    )
    out, counts = redact_text(text)
    assert counts["telegram_token"] == 2
