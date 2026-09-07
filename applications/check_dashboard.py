"""Browser interaction checks with CUDA reference predictions."""

import json

import torch
from playwright.sync_api import sync_playwright

from .regression import pendulum
from .shared import ROOT, predict_equation, save, setup


def main():
    setup()
    errors = []
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto((ROOT / "index.html").as_uri())
        model = json.loads((ROOT / "results/surrogate/equation.json").read_text())
        x = torch.tensor([[0.4, 0.03], [1.2, 0.15], [1.8, 0.27]], device="cuda")
        mean, std = (
            torch.tensor(model["xmean"], device="cuda"),
            torch.tensor(model["xstd"], device="cuda"),
        )
        ref = (1 - torch.cos(x[:, 0])) * torch.exp(
            predict_equation(model, (x - mean) / std) * model["ystd"] + model["ymean"]
        )
        browser_values = page.evaluate(
            "rows=>rows.map(r=>predict('surrogate',r))", x.cpu().tolist()
        )
        error = float((torch.tensor(browser_values, device="cuda") - ref).abs().max())
        assert error < 1e-5, error
        simulation = page.evaluate("rows=>rows.map(r=>simulate(...r).at(-1)[1])", x.cpu().tolist())
        sim_error = float(
            (torch.tensor(simulation, device="cuda", dtype=torch.float64) - pendulum(x.double()))
            .abs()
            .max()
        )
        assert sim_error < 1e-9, sim_error
        for width in [1440, 390]:
            page.set_viewport_size({"width": width, "height": 1080})
            page.locator('[data-panel="surrogate"]').click()
            page.locator("#angle").fill("1.8")
            page.locator("#angle").dispatch_event("input")
            assert page.locator("#energy").inner_text()
            page.locator('[data-panel="distillation"]').click()
            assert "dB" in page.locator("#noise").inner_text()
            page.locator("#f3").fill("999")
            assert "Outside" in page.locator("#noise").inner_text()
            page.locator("#resetAirfoil").click()
            page.locator('[data-panel="llm"]').click()
            for suite in ["pairs-prose", "range-prose", "pairs-symbolic", "pairs-code"]:
                page.locator("#suite").select_option(suite)
                page.locator("#problem").select_option("1")
                assert page.locator("#llmPlot polyline").count() == 3
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator('[data-panel="surrogate"]').click()
            page.screenshot(path=str(ROOT / f"results/demo-{width}.png"), full_page=True)
            results.append(
                {"width": width, "tabs": 3, "llm_suites": 4, "horizontal_overflow": False}
            )
        browser.close()
    assert not errors, errors
    save(
        ROOT / "results/browser-validation.json",
        {
            "viewports": results,
            "javascript_errors": errors,
            "javascript_equation_vs_cuda_max_error": error,
            "javascript_RK4_vs_cuda_max_error": sim_error,
        },
    )
    print("Browser interactions and CUDA reference comparisons passed")


if __name__ == "__main__":
    main()
