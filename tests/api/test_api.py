import pandas as pd
from fastapi.testclient import TestClient

from api.main import app

# Note: Starlette warns that httpx+TestClient is deprecated in favour of
# httpx2. Tests pass; revisit on the next dependency update.

client = TestClient(app)

# Valid CSV used for the upload-limit boundary pair (exact size / size - 1).
_LIMIT_PROBE_CSV = b"a,b\n1,2\n3,4\n"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_csv_with_target():
    content = b"a,label\n1,x\n2,y\n"
    response = client.post(
        "/analyze",
        files={"file": ("people.csv", content, "text/csv")},
        data={"target": "label"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["source_name"] == "people.csv"
    assert body["metadata"]["file_format"] == "csv"
    assert body["metadata"]["row_count"] == 2
    assert body["metadata"]["column_count"] == 2
    assert body["target"] == "label"
    assert "findings" in body


def test_analyze_excel_with_sheet_name(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": ["1"], "label": ["x"]}).to_excel(
            writer, sheet_name="first", index=False
        )
        pd.DataFrame({"b": ["2"]}).to_excel(writer, sheet_name="second", index=False)

    response = client.post(
        "/analyze",
        files={
            "file": (
                "book.xlsx",
                path.read_bytes(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
        data={"sheet_name": "first", "target": "label"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["file_format"] == "excel"
    assert body["metadata"]["sheet_name"] == "first"
    assert body["metadata"]["available_sheets"] == ["first", "second"]
    assert body["metadata"]["row_count"] == 1
    assert body["metadata"]["column_count"] == 2


def test_txt_upload_gives_415():
    response = client.post(
        "/analyze",
        files={"file": ("notes.txt", b"a,b\n1,2\n", "text/plain")},
    )

    assert response.status_code == 415
    assert "detail" in response.json()


def test_empty_file_gives_422():
    response = client.post(
        "/analyze",
        files={"file": ("empty.csv", b"", "text/csv")},
    )

    assert response.status_code == 422
    assert "detail" in response.json()


def test_bad_target_gives_400():
    content = b"a,label\n1,x\n2,y\n"
    response = client.post(
        "/analyze",
        files={"file": ("people.csv", content, "text/csv")},
        data={"target": "nope"},
    )

    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_upload_exactly_at_limit_is_accepted(monkeypatch):
    monkeypatch.setattr("api.main.MAX_UPLOAD_BYTES", len(_LIMIT_PROBE_CSV))

    response = client.post(
        "/analyze",
        files={"file": ("exact.csv", _LIMIT_PROBE_CSV, "text/csv")},
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["row_count"] == 2


def test_upload_one_byte_over_limit_gives_413(monkeypatch):
    monkeypatch.setattr("api.main.MAX_UPLOAD_BYTES", len(_LIMIT_PROBE_CSV) - 1)

    response = client.post(
        "/analyze",
        files={"file": ("over.csv", _LIMIT_PROBE_CSV, "text/csv")},
    )

    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()
