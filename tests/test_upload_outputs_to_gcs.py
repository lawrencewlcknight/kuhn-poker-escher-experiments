from pathlib import Path

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
