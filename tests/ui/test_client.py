from types import SimpleNamespace

import pytest

from ui.client import AnalyzeError, analyze_upload


def test_success_returns_report_dict():
    def fake_post(url, *, files=None, data=None, timeout=None):
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "metadata": {"source_name": "people.csv"},
                "findings": [],
            },
        )

    report = analyze_upload(
        "http://localhost:8000/",
        "people.csv",
        b"a,label\n1,x\n",
        post=fake_post,
    )

    assert report["metadata"]["source_name"] == "people.csv"
    assert report["findings"] == []


def test_4xx_raises_analyze_error_with_api_detail():
    def fake_post(url, *, files=None, data=None, timeout=None):
        return SimpleNamespace(
            status_code=400,
            json=lambda: {"detail": "Column 'nope' not found in dataset"},
            text='{"detail":"Column \'nope\' not found in dataset"}',
        )

    with pytest.raises(AnalyzeError, match="nope") as exc_info:
        analyze_upload(
            "http://127.0.0.1:8000",
            "people.csv",
            b"a,b\n1,2\n",
            target="nope",
            post=fake_post,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Column 'nope' not found in dataset"


def test_non_json_error_body_falls_back_to_text():
    def fake_post(url, *, files=None, data=None, timeout=None):
        def bad_json():
            raise ValueError("not json")

        return SimpleNamespace(
            status_code=502,
            json=bad_json,
            text="bad gateway from proxy",
        )

    with pytest.raises(AnalyzeError) as exc_info:
        analyze_upload(
            "http://127.0.0.1:8000",
            "people.csv",
            b"a,b\n1,2\n",
            post=fake_post,
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == "bad gateway from proxy"


def test_target_and_sheet_name_sent_only_when_set():
    captured: dict = {}

    def fake_post(url, *, files=None, data=None, timeout=None):
        captured.setdefault("calls", []).append(data)
        return SimpleNamespace(status_code=200, json=lambda: {"ok": True})

    analyze_upload(
        "http://127.0.0.1:8000",
        "people.csv",
        b"a,b\n1,2\n",
        post=fake_post,
    )
    analyze_upload(
        "http://127.0.0.1:8000",
        "book.xlsx",
        b"fake",
        target="label",
        sheet_name="sales",
        post=fake_post,
    )

    assert captured["calls"][0] == {}
    assert captured["calls"][1] == {"target": "label", "sheet_name": "sales"}
