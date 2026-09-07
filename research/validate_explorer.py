"""Browser interactions and independent JavaScript equations against GPU measurements."""

import time

import torch
from playwright.sync_api import sync_playwright

from data import ROOT, save
from model_io import setup


def main():
    setup()
    errors = []
    checks = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1080})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto((ROOT / "explorer.html").as_uri())
        values = page.evaluate("""() => window.emlExplorer.data.map(d => {
            const differences=[];
            for(const row of d.cases) for(let i=0;i<d.alphas.length;i++) {
                const z=window.emlExplorer.features(row,d.alphas[i]);
                const predicted=window.emlExplorer.parts(d,z).value;
                differences.push((predicted-row.curves.eml.predicted_coefficient[i])/d.output_scale);
            }
            return {model:d.model,operation:d.operation,mode:d.mode,differences};
        })""")
        for row in values:
            difference = torch.tensor(row.pop("differences"), device="cuda", dtype=torch.float64)
            assert torch.isfinite(difference).all()
            error = float(difference.abs().max())
            assert error < 0.001, (row, error)
            checks.append(
                {
                    **row,
                    "values_compared": difference.numel(),
                    "maximum_error_in_training_standard_deviations": error,
                }
            )
        screenshots = []
        for width in [1440, 390]:
            page.set_viewport_size({"width": width, "height": 1080})
            for index in range(len(values)):
                page.locator("#dataset").select_option(str(index))
                styles = page.locator("#style option").evaluate_all("rows=>rows.map(r=>r.value)")
                for style in styles:
                    page.locator("#style").select_option(style)
                    page.locator("#problem").select_option("1")
                    page.locator("#alpha").evaluate(
                        "el=>{el.value='5';el.dispatchEvent(new Event('input'));}"
                    )
                    assert page.locator("#alpha-value").inner_text() == "1.000"
                    assert page.locator("#coefficient polyline").count() == 3
                    assert page.locator("#margin polyline").count() == 3
                    assert page.locator("#units rect").count() >= 2
                    assert "EML units" in page.locator("#nodes").inner_text()
                    assert "training mean" in page.locator("#ablation").inner_text()
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            page.locator("#dataset").select_option("0")
            screenshot = ROOT / f"explorer-{width}.png"
            page.screenshot(path=str(screenshot), full_page=True)
            screenshots.append(str(screenshot))
        browser.close()
    assert not errors, errors
    save(
        ROOT / "explorer-validation.json",
        {
            "completed_epoch": time.time(),
            "gpu_reference_checks": checks,
            "viewports": [1440, 390],
            "javascript_errors": errors,
            "screenshots": screenshots,
            "scope": "Browser equations use independently implemented JavaScript; differences from GPU-measured predictions are evaluated on CUDA. Browser rendering itself runs in Chromium.",
        },
    )
    print("EXPLORER VALIDATION PASSED", len(checks), flush=True)


if __name__ == "__main__":
    main()
