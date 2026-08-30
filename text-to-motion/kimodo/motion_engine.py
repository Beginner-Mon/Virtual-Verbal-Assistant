"""Nạp model + sinh motion — dùng chung bởi mcp_server.py và worker.py.

Tách ra để worker không nhân bản code nạp model. Model nạp mất 38.2 giây (đo trên A10G)
và chiếm 16.36/24 GB VRAM, nên chỉ nạp một lần cho cả vòng đời tiến trình.
"""
from __future__ import annotations

import os

DEFAULT_MODEL_NAME = "Kimodo-SMPLX-RP-v1"


def build_base_name(job_id: str) -> str:
    """Tên file = job_id. Không timestamp, không random — nếu không thì Lambda không thể
    tính trước URL và cache theo nội dung không bao giờ trúng."""
    return job_id


class MotionEngine:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.resolved_model_name = None
        self.device = None

    def load(self) -> None:
        import torch
        from kimodo import load_model

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model, self.resolved_model_name = load_model(
            self.model_name, device=self.device,
            default_family="Kimodo", return_resolved_name=True,
        )

    def generate(self, prompt: str, duration: float, steps: int) -> dict:
        texts = [t.strip() + "." for t in prompt.split(".") if t.strip()]
        num_frames = [int(duration * self.model.fps)] * len(texts)
        return self.model(
            texts, num_frames, constraint_lst=[], num_denoising_steps=steps,
            num_samples=1, multi_prompt=True, num_transition_frames=5,
            post_processing=True, return_numpy=True,
        )

    def save_outputs(self, output: dict, out_dir: str, base_name: str) -> tuple[str, str]:
        """Ghi NPZ rồi convert sang BVH ngay tại đây. status.md ghi converter đang là CLI
        thủ công 'chưa ai gọi từ backend' — gọi in-process là đóng gap đó."""
        from kimodo.exports.motion_io import save_kimodo_npz
        from npz_to_bvh import convert_npz_to_bvh

        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, f"{base_name}.npz")
        single = {k: (v[0] if hasattr(v, "shape") and v.shape and v.shape[0] == 1 else v)
                  for k, v in output.items()}
        save_kimodo_npz(npz_path, single)

        bvh_path = os.path.join(out_dir, f"{base_name}.bvh")
        convert_npz_to_bvh(npz_path, bvh_path)
        return npz_path, bvh_path
