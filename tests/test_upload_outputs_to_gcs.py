from pathlib import Path
import sys
import types

import pytest

from scripts.upload_outputs_to_gcs import (
    destination_blob_name,
    iter_files,
    parse_gcs_uri,
)


def test_parse_gcs_uri_accepts_bucket_and_prefix():
    assert parse_gcs_uri("gs://example-bucket/job-123/") == (
        "example-bucket",
        "job-123",
    )


def test_parse_gcs_uri_rejects_non_gcs_uri():
    with pytest.raises(ValueError):
        parse_gcs_uri("https://example.com/job-123")


def test_destination_blob_name_preserves_outputs_directory_prefix():
    source_dir = Path("/workspace/repo/outputs")
    file_path = source_dir / "cloud" / "exp12" / "variant_summary.csv"

    assert (
        destination_blob_name(source_dir, file_path, "escher-exp12-20260614-002418")
        == "escher-exp12-20260614-002418/outputs/cloud/exp12/variant_summary.csv"
    )


def test_iter_files_skips_directories(tmp_path):
    (tmp_path / "outputs" / "cloud").mkdir(parents=True)
    file_path = tmp_path / "outputs" / "cloud" / "summary.json"
    file_path.write_text("{}", encoding="utf-8")

    assert list(iter_files(tmp_path / "outputs")) == [file_path]


def test_upload_directory_uses_explicit_auth_request(tmp_path, monkeypatch):
    uploaded = []
    default_calls = []

    class FakeRequest:
        pass

    class FakeBlob:
        def __init__(self, name):
            self.name = name

        def upload_from_filename(self, filename):
            uploaded.append((self.name, Path(filename).name))

    class FakeBucket:
        def blob(self, name):
            return FakeBlob(name)

    class FakeClient:
        def __init__(self, project=None, credentials=None):
            assert project == "test-project"
            assert credentials == "test-credentials"

        def bucket(self, name):
            assert name == "example-bucket"
            return FakeBucket()

    def fake_default(*, request):
        assert isinstance(request, FakeRequest)
        default_calls.append(request)
        return "test-credentials", "test-project"

    google_mod = types.ModuleType("google")
    google_mod.__path__ = []
    auth_mod = types.ModuleType("google.auth")
    auth_mod.default = fake_default
    cloud_mod = types.ModuleType("google.cloud")
    storage_mod = types.ModuleType("google.cloud.storage")
    storage_mod.Client = FakeClient
    transport_mod = types.ModuleType("google.auth.transport")
    requests_mod = types.ModuleType("google.auth.transport.requests")
    requests_mod.Request = FakeRequest

    google_mod.auth = auth_mod
    google_mod.cloud = cloud_mod
    cloud_mod.storage = storage_mod
    auth_mod.transport = transport_mod
    transport_mod.requests = requests_mod

    for name, module in {
        "google": google_mod,
        "google.auth": auth_mod,
        "google.cloud": cloud_mod,
        "google.cloud.storage": storage_mod,
        "google.auth.transport": transport_mod,
        "google.auth.transport.requests": requests_mod,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source_dir = tmp_path / "outputs"
    source_dir.mkdir()
    (source_dir / "summary.json").write_text("{}", encoding="utf-8")

    from scripts.upload_outputs_to_gcs import upload_directory

    assert upload_directory(source_dir, "gs://example-bucket/job-123/") == 1
    assert len(default_calls) == 1
    assert uploaded == [("job-123/outputs/summary.json", "summary.json")]
