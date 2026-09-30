"""Contextual header, identifier tokenization, stopword removal and RU/EN stemming."""
from code_rag import textindex as ti


def test_make_header_format():
    header = ti.make_header("httpx/httpx/_client.py", "Client.send", "method")
    assert header == "httpx/httpx/_client.py | Client.send | method"


def test_make_header_normalises_windows_separators():
    assert ti.make_header("httpx\\httpx\\_client.py", "send", "method").startswith("httpx/httpx/_client.py")


def test_split_identifier_splits_path_separators():
    assert ti.split_identifier("rich-demo/src/live_render.py") == [
        "rich", "demo", "src", "live", "render", "py",
    ]


def test_split_identifier_splits_underscored_symbol():
    assert ti.split_identifier("generate_image") == ["generate", "image"]


def test_split_identifier_splits_dotted_method_name():
    assert ti.split_identifier("BigClass.method_two") == ["big", "class", "method", "two"]


def test_split_identifier_splits_camel_case():
    assert ti.split_identifier("HTTPServerError") == ["http", "server", "error"]


def test_header_tokens_text_includes_kind_and_repo():
    tokens = ti.header_tokens_text("imagekit/src/image_client.py", "imagekit", "generate_image", "function").split()
    assert "client" in tokens
    assert "image" in tokens
    assert "generate" in tokens
    assert "function" in tokens
    assert "imagekit" in tokens


def test_stopwords_are_recognized_ru_and_en():
    assert ti.is_stopword("как")
    assert ti.is_stopword("что")
    assert ti.is_stopword("the")
    assert ti.is_stopword("does")
    assert not ti.is_stopword("деплоер")
    assert not ti.is_stopword("deployer")


def test_query_tokens_drops_stopwords_and_dedupes():
    tokens = ti.query_tokens("Как деплоер публикует сайт?")
    assert tokens == ["деплоер", "публикует", "сайт"]
    assert "как" not in tokens


def test_stem_text_drops_stopwords_and_stems_russian():
    assert ti.stem_text("Как деплоер публикует сайты") == "деплоер публик сайт"


def test_stem_text_stems_english():
    assert ti.stem_text("How does the deployer publish sites") == "deploy publish site"


def test_stem_token_picks_backend_by_script():
    # Cyrillic input -> Russian stemmer, Latin input -> English stemmer.
    assert ti.stem_token("публикует") == "публик"
    assert ti.stem_token("publishing") == ti._en_stemmer.stemWord("publishing")


def test_index_time_and_query_time_stemming_agree():
    # The whole point of storing a stemmed body column is that indexing and
    # querying can never drift apart because they call the same function.
    word = "публикует"
    assert ti.stem_token(word) in ti.stem_text(f"текст {word} текст").split()
