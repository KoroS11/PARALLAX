"""Unit tests for PARALLAX dark terminal dashboard SPA."""

import pytest
from httpx import ASGITransport, AsyncClient

from parallax.api.app import create_app
from parallax.api.settings import Settings
from parallax.contracts.detectors import DetectorResult
from parallax.dashboard.app import render_dashboard_html


class DummyDetector:
    """Mock detector for fast hermetic testing."""

    name = "dummy-detector"

    async def predict(self, text: str) -> DetectorResult:
        return DetectorResult(
            name=self.name,
            is_attack=False,
            score=0.05,
            attack_class=None,
            latency_ms=1.0,
            error=None,
        )


@pytest.fixture
def test_app():
    settings = Settings(environment="testing", database_url="sqlite:///:memory:")
    return create_app(settings=settings, detectors=[DummyDetector()])


@pytest.mark.asyncio
async def test_dashboard_endpoint_serves_html(test_app):
    """Verify GET /dashboard returns self-contained HTML page with all view tabs."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        html = resp.text

        # Verify key UI components and views exist in rendered markup
        assert "PARALLAX" in html
        assert "CYBER-PHYSICAL ADVERSARIAL DEFENSE" in html
        assert "[1] LIVE SCAN" in html
        assert "[2] SANDBOX REPLAY" in html
        assert "[3] EVALUATION METRICS (E1-E7)" in html
        assert "[4] BENCHMARK & LATENCY (E5)" in html
        assert "[5] CHALLENGER TRACE" in html
        assert "[6] ABLATION & TRADE-OFF" in html
        assert "Multi-View Preprocessor Output" in html
        assert "CPS Scenario Catalog" in html
        assert "Wave C Empirical Benchmark Results" in html
        assert "Adversarial Review Subsystem: Challenger & Judge Trace" in html
        assert "Ablation Study" in html
        assert "5-Arm Comparative Benchmark" in html


@pytest.mark.asyncio
async def test_dashboard_ablation_endpoint(test_app):
    """Verify GET /dashboard/api/ablation returns available status or 404."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/dashboard/api/ablation")
        assert resp.status_code in (200, 404)
        data = resp.json()
        assert "status" in data


@pytest.mark.asyncio
async def test_root_path_redirects_to_dashboard(test_app):
    """Verify GET / redirects to /dashboard."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/", follow_redirects=False)
        assert resp.status_code == 307
        assert resp.headers["location"] == "/dashboard"


@pytest.mark.asyncio
async def test_dashboard_summary_api(test_app):
    """Verify GET /dashboard/api/summary returns status and 16 expanded scenarios."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/dashboard/api/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "online"
        assert data["num_scenarios"] == 16
        assert "C1" in data["claims"]
        assert "C2" in data["claims"]
        assert "C3" in data["claims"]
        assert "D1" in data["claims"]


@pytest.mark.asyncio
async def test_dashboard_wave_f_api(test_app):
    """Verify GET /dashboard/api/wave-f returns status and Wave F empirical benchmarks."""
    async with AsyncClient(
        transport=ASGITransport(app=test_app), base_url="http://test"
    ) as client:
        resp = await client.get("/dashboard/api/wave-f")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "available"
        assert data["wave"] == "F"


def test_render_dashboard_html_contains_scenarios():
    """Verify direct HTML renderer generates standalone markup with registered scenarios."""
    html = render_dashboard_html()
    assert "<!DOCTYPE html>" in html
    assert "smart_grid_substation" in html
    assert "chemical_reactor" in html
    assert "nuclear_cooling_loop" in html
    # Ensure zero external unpkg / cdnjs / googleapis links (air-gapped guarantee)
    assert "http://" not in html.lower() or "http://test" in html.lower()
    assert "cdnjs.cloudflare.com" not in html
    assert "unpkg.com" not in html
    assert "fonts.googleapis.com" not in html
