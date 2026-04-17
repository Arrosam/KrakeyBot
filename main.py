"""Root launcher: `python main.py` starts the CogniBot heartbeat loop."""
import asyncio

from src.main import build_runtime_from_config


if __name__ == "__main__":
    try:
        asyncio.run(build_runtime_from_config().run())
    except KeyboardInterrupt:
        pass
