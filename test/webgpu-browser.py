#!/usr/bin/env python3
"""Run a WebGPU browser probe. Requires Playwright and Chromium.

Example: python3 test/webgpu-browser.py test/rigid-render.mjs --software
A probe exports async run(demo) and returns a JSON-serializable result.
"""
import argparse
import functools
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import quote
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    parser.add_argument("--browser", help="Path to Chromium (defaults to Playwright Chromium)")
    parser.add_argument("--software", action="store_true", help="Use the SwiftShader adapter")
    parser.add_argument("--scene", default="rigid-mesh")
    parser.add_argument("--probe-args", type=json.loads, default={}, help="JSON options passed to run(demo, options)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=144)
    args = parser.parse_args()
    module = args.probe.resolve().relative_to(ROOT).as_posix()

    # The test renders the real pipelines into a readable texture. This also works
    # on headless systems where the adapter cannot present to a canvas swap chain.
    init = """
      window.deviceErrors = [];
      const targets = new WeakMap();
      GPUCanvasContext.prototype.configure = function({device, format}) {
        const size = RESOLUTION;
        this.canvas.width = size[0];
        this.canvas.height = size[1];
        device.addEventListener('uncapturederror', e => window.deviceErrors.push(e.error.message));
        targets.set(this, device.createTexture({
          size, format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.TEXTURE_BINDING
        }));
      };
      GPUCanvasContext.prototype.getCurrentTexture = function() { return targets.get(this); };
    """.replace("RESOLUTION", json.dumps([args.width, args.height]))
    flags = ["--enable-unsafe-webgpu"]
    if args.software:
        flags += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                  "--use-webgpu-adapter=swiftshader", "--ignore-gpu-blocklist"]
    with ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(SimpleHTTPRequestHandler, directory=ROOT)) as server:
        Thread(target=server.serve_forever, daemon=True).start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path=args.browser, args=flags)
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.add_init_script(init)

                def three_module(route):
                    # Use the system trust store for the demo's pinned external modules.
                    with urlopen(route.request.url, timeout=30) as response:
                        route.fulfill(body=response.read(), content_type="text/javascript",
                                      headers={"access-control-allow-origin": "*"})

                page.route("https://cdn.jsdelivr.net/npm/three@0.170.0/**", three_module)
                page.goto(f"http://127.0.0.1:{server.server_port}/webgpu.html?scene={quote(args.scene)}")
                page.wait_for_function("window.wgpuDemo || document.querySelector('#err')?.textContent", timeout=60000)
                result = page.evaluate("""async ([path, options]) => {
                  const demo = window.wgpuDemo;
                  if (!demo) throw Error(document.querySelector('#err')?.textContent || 'Demo startup failed');
                  demo.stopLoop();
                  await demo.gpu.device.queue.onSubmittedWorkDone();
                  const {run} = await import(path);
                  let result;
                  try { result = await run(demo, options); }
                  catch (error) {
                    if (window.deviceErrors.length) throw Error(error.message + "; " + window.deviceErrors[0]);
                    throw error;
                  }
                  await demo.gpu.device.queue.onSubmittedWorkDone();
                  if (window.deviceErrors.length) throw Error(window.deviceErrors.slice(0, 3).join("; "));
                  return result;
                }""", ["/" + module, args.probe_args])
                if errors:
                    raise RuntimeError("\n".join(errors))
                print(json.dumps(result, indent=2))
                browser.close()
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
