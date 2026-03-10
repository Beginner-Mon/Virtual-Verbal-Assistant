"""MotionGenerationTool — calls the DART Text-to-Motion API.

Wraps the DART server (http://localhost:5001) behind a clean single-method
interface so the OrchestratorAgent can request motion generation without
knowing the HTTP details.

API contract:
    POST http://localhost:5001/generate
    Body:  {"text_prompt": "<exercise name>", "guidance_scale": 5.0, "num_steps": 50}
    200:   {"request_id": "abc123", "motion_file_url": "/download/motion_abc123.npz", "num_frames": 160, "fps": 30, "duration_seconds": 5.33, "text_prompt": "..."}
"""

from typing import Any, Dict

import requests

from utils.logger import get_logger

logger = get_logger(__name__)

# Defaults — can be overridden at construction time
_DEFAULT_ENDPOINT = "http://localhost:5001/generate"
_DEFAULT_TIMEOUT  = 30   # seconds


class MotionGenerationTool:
    """Tool that calls the DART Text-to-Motion API to generate motion data.

    The OrchestratorAgent invokes this tool when the detected intent is
    ``visualize_motion`` and the caller requires an animation file.

    Note:
        This tool is intentionally **not** wired into the pipeline yet.
        It will be plugged in during a later integration step.
    """

    def __init__(
        self,
        endpoint: str = _DEFAULT_ENDPOINT,
        timeout: int  = _DEFAULT_TIMEOUT,
    ) -> None:
        """Initialise the tool with a configurable endpoint and timeout.

        Args:
            endpoint: Full URL of the DART generate_motion endpoint.
            timeout:  HTTP request timeout in seconds.
        """
        self._endpoint = endpoint
        self._timeout  = timeout
        logger.info(
            f"[MotionGenerationTool] Initialised — endpoint={self._endpoint} "
            f"timeout={self._timeout}s"
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_motion(self, prompt: str) -> Dict[str, Any]:
        """Call the DART API and return structured motion metadata.

        Args:
            prompt: Natural-language description of the exercise/movement
                    (e.g. ``"chin tuck"`` or ``"push up"``).

        Returns:
            Dict with keys:
                motion_file (str)  — filename of the generated .npz file
                frames      (int)  — total frame count
                fps         (int)  — frames per second

        Raises:
            ValueError: If the API response is missing required fields.

        Example::

            tool = MotionGenerationTool()
            result = tool.generate_motion("chin tuck")
            # {"motion_file": "motion_abc123.npz", "frames": 160, "fps": 30}
        """
        if not prompt or not prompt.strip():
            raise ValueError("[MotionGenerationTool] prompt must be a non-empty string")

        prompt = prompt.strip()
        logger.info(f"[MotionGenerationTool] generate_motion prompt='{prompt}'")

        try:
            response = requests.post(
                self._endpoint,
                json={
                    "text_prompt": prompt,
                    "guidance_scale": 5.0,
                    "num_steps": 50,
                    "respacing": "ddim50"
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data: Dict[str, Any] = response.json()

        except requests.exceptions.ConnectionError as exc:
            msg = f"[MotionGenerationTool] Cannot reach DART server at {self._endpoint}: {exc}"
            logger.error(msg)
            return {"error": msg, "motion_file": None, "frames": 0, "fps": 0}

        except requests.exceptions.Timeout:
            msg = (
                f"[MotionGenerationTool] DART request timed out after {self._timeout}s "
                f"for prompt='{prompt}'"
            )
            logger.error(msg)
            return {"error": msg, "motion_file": None, "frames": 0, "fps": 0}

        except requests.exceptions.HTTPError as exc:
            msg = (
                f"[MotionGenerationTool] DART returned HTTP {exc.response.status_code} "
                f"for prompt='{prompt}': {exc}"
            )
            logger.error(msg)
            return {"error": msg, "motion_file": None, "frames": 0, "fps": 0}

        except Exception as exc:
            msg = f"[MotionGenerationTool] Unexpected error: {type(exc).__name__}: {exc}"
            logger.error(msg)
            return {"error": msg, "motion_file": None, "frames": 0, "fps": 0}

        # Validate required fields - DART returns motion_file_url and num_frames
        missing = [f for f in ("motion_file_url", "num_frames", "fps") if f not in data]
        if missing:
            raise ValueError(
                f"[MotionGenerationTool] DART response missing required fields: {missing}. "
                f"Got: {list(data.keys())}"
            )

        # Convert DART response format to internal format
        motion_filename = data["motion_file_url"].split("/")[-1]  # Extract filename from URL
        result = {
            "motion_file": motion_filename,
            "frames":      int(data["num_frames"]),
            "fps":         int(data["fps"]),
        }
        logger.info(
            f"[MotionGenerationTool] Motion generated — file={result['motion_file']} "
            f"frames={result['frames']} fps={result['fps']}"
        )
        return result


# ---------------------------------------------------------------------------
# Example usage (run directly for quick smoke-test)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    tool = MotionGenerationTool()
    try:
        result = tool.generate_motion("chin tuck")
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(f"Validation error: {e}")
