#!/usr/bin/env python3
"""Bundle the JUROR demo into a single self-contained HTML file.

Usage:
    python bundle.py

Output:
    juror_demo_standalone.html  (open by double-clicking, no server needed)

This bundles BOTH the demo trace player AND the live-LLM runtime, so the
distributed single file supports Demo mode (no key) and Live mode
(user pastes an API key in Settings).
"""

import os
import json
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO = os.path.join(HERE, "demo")


def main():
    html  = open(os.path.join(DEMO, "index.html"),    encoding="utf-8").read()
    css   = open(os.path.join(DEMO, "style.css"),     encoding="utf-8").read()
    live  = open(os.path.join(DEMO, "live-agent.js"), encoding="utf-8").read()
    anim  = open(os.path.join(DEMO, "animation.js"),  encoding="utf-8").read()

    traces = {}
    for i in (1, 2, 3):
        path = os.path.join(DEMO, "traces", f"case_{i:03d}.json")
        with open(path, encoding="utf-8") as f:
            traces[i] = json.load(f)

    # Replace the fetch-based loadCase with one that reads the embedded global.
    new_loadCase = """  async loadCase(n) {
    this.stop();
    this.reset();
    const trace = (window.JUROR_TRACES || {})[n];
    if (!trace) {
      this.showError('Embedded trace not found for case ' + n);
      return;
    }
    // Deep copy so resets / re-plays start from a clean state.
    this.trace = JSON.parse(JSON.stringify(trace));
    this.bindCaseInfo();
    this.flatten();
    this.updateClockTotal();
  }"""

    pattern = re.compile(
        r"  async loadCase\(n\) \{[\s\S]*?      this\.showError\([\s\S]*?\}\s*\}",
        re.MULTILINE,
    )
    if not pattern.search(anim):
        raise SystemExit("ERROR: original loadCase block not found in animation.js")
    anim_patched = pattern.sub(new_loadCase, anim, count=1)

    traces_js = "window.JUROR_TRACES = " + json.dumps(traces) + ";"

    bundled_js = "\n".join([
        traces_js,
        live,
        anim_patched,
    ])

    # Inline CSS + both JS files into the HTML.
    html_out = html.replace(
        '<link rel="stylesheet" href="style.css">',
        "<style>\n" + css + "\n</style>",
    ).replace(
        '<script src="live-agent.js"></script>\n<script src="animation.js"></script>',
        "<script>\n" + bundled_js + "\n</script>",
    )

    # Sanity
    assert 'href="style.css"'    not in html_out, "local style.css still linked"
    assert 'src="animation.js"'  not in html_out, "animation.js still external"
    assert 'src="live-agent.js"' not in html_out, "live-agent.js still external"
    assert "window.JUROR_TRACES" in html_out,    "traces not embedded"
    assert "window.JUROR_LIVE"   in html_out,    "live agent not embedded"

    out_path = os.path.join(HERE, "juror_demo_standalone.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_out)

    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path}  ({size_kb:.1f} KB)")
    print(f"Embedded cases: " +
          ", ".join(f"#{k}={v['case_id']}" for k, v in traces.items()))
    print("\nUsage:")
    print("  · Double-click the file to open it in any modern browser.")
    print("  · Demo mode runs immediately (no key needed).")
    print("  · For Live mode: click Settings (top-right) → choose provider →")
    print("    paste API key → 'Activate Live'. Then click Run.")


if __name__ == "__main__":
    main()
