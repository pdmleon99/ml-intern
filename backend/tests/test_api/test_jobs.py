

def test_create_job_missing_key(client):
    """401 when no X-LLM-Api-Key header."""
    response = client.post("/api/jobs", json={
        "user_description": "test",
        "dataset_source": "upload",
        "uploaded_file_path": "/tmp/test.csv",
    })
    assert response.status_code == 401


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_get_job_not_found(client, auth_headers):
    response = client.get("/api/jobs/nonexistent-job-id", headers=auth_headers)
    assert response.status_code == 404


def test_list_jobs_empty(client, auth_headers):
    response = client.get("/api/jobs", headers=auth_headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_validate_key_endpoint_bad_key(client):
    """Returns {valid: false} for a bad key — does not crash."""
    response = client.post("/api/validate-key", headers={
        "X-LLM-Api-Key": "sk-bad-key",
        "X-LLM-Provider": "anthropic",
        "X-LLM-Model": "claude-haiku-4-5",
    })
    assert response.status_code == 200
    data = response.json()
    assert "valid" in data


def test_upload_missing_file(client, auth_headers):
    """Upload endpoint returns 422 if no file."""
    response = client.post("/api/datasets/upload", headers=auth_headers)
    assert response.status_code == 422


def test_upload_invalid_extension(client, auth_headers, tmp_path):
    """Upload endpoint rejects unsupported file types."""
    bad_file = tmp_path / "data.exe"
    bad_file.write_bytes(b"not a real dataset")
    with open(bad_file, "rb") as f:
        response = client.post(
            "/api/datasets/upload",
            headers={k: v for k, v in auth_headers.items() if k != "Content-Type"},
            files={"file": ("data.exe", f, "application/octet-stream")},
        )
    assert response.status_code == 400
