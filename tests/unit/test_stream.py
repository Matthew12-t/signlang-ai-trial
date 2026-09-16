from src.stt.stream import merge_overlapping_text


def test_merge_overlapping_text_removes_repeated_words() -> None:
    assert (
        merge_overlapping_text("the deadline is Friday", "is Friday at five")
        == "the deadline is Friday at five"
    )


def test_merge_overlapping_text_appends_when_there_is_no_overlap() -> None:
    assert merge_overlapping_text("hello", "world") == "hello world"
