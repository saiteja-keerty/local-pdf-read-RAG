"""Launcher for InsureChat that imports the package and starts the Gradio demo.

This file is intended as a simple entrypoint for Hugging Face Spaces which
executes a top-level script. It imports the `insurechat.app` module and then
launches the `demo` Gradio interface defined there.
"""
from insurechat import app as _app
import atexit
import asyncio


def _close_async_loop():
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            try:
                loop.stop()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
    except Exception:
        pass


atexit.register(_close_async_loop)


if __name__ == "__main__":
    try:
        # Explicit server args help Spaces and make behavior predictable.
        _app.demo.launch(server_name="0.0.0.0", server_port=7860, share=False, inbrowser=False)
    except Exception as e:
        print("Failed to launch InsureChat demo:", e)
        raise
    finally:
        _close_async_loop()

if __name__ == "__main__":
    try:
        _app.demo.launch()
    except Exception as e:
        print("Failed to launch InsureChat demo:", e)
        raise
