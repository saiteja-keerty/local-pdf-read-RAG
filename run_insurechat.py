"""Launcher for InsureChat that imports the package and starts the Gradio demo.

This file is intended as a simple entrypoint for Hugging Face Spaces which
executes a top-level script. It imports the `insurechat.app` module and then
launches the `demo` Gradio interface defined there.
"""
from insurechat import app as _app

if __name__ == "__main__":
    try:
        _app.demo.launch()
    except Exception as e:
        print("Failed to launch InsureChat demo:", e)
        raise
