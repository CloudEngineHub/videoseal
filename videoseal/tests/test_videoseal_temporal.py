# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Test temporal pooling with trained videoseal model.
Verifies that temporal pooling doesn't break detection accuracy.

Run with:
    python -m videoseal.tests.test_videoseal_temporal
"""

import copy
import torch
import torchvision
import unittest
import matplotlib.pyplot as plt
from pathlib import Path

import videoseal
from videoseal.evals.metrics import bit_accuracy


class TestTemporalPooling(unittest.TestCase):
    """Test temporal pooling with trained pixelseal model on real video."""

    @classmethod
    def setUpClass(cls):
        """Load trained model and video once."""
        cls.model = videoseal.load("pixelseal")
        cls.model.eval()
        cls.nbits = cls.model.embedder.msg_processor.nbits
        
        # Load video from assets using torchvision
        video_path = Path(__file__).parent.parent.parent / "assets" / "videos" / "1.mp4"
        video, _, _ = torchvision.io.read_video(str(video_path))
        cls.video = video.permute(0, 3, 1, 2).float() / 255.0
        cls.video = cls.video[:16]  # Use first 16 frames
        
        print(f"\nLoaded pixelseal model with {cls.nbits} bits")
        print(f"Video shape: {cls.video.shape} (frames, C, H, W)")
        print(f"Default video_mode: {cls.model.video_mode}")
        print(f"Default step_size: {cls.model.step_size}")

    def _get_accuracy(self, model, video, msg):
        """Helper to embed, detect, and return per-frame accuracy."""
        with torch.no_grad():
            outputs = model.embed(video, msg, is_video=True)
            preds = model.detect(outputs["imgs_w"], is_video=True)
        
        pred_msgs = preds["preds"][:, 1:]  # Skip detection bit
        per_frame = [bit_accuracy(pred_msgs[i:i+1], msg).item() for i in range(video.shape[0])]
        return per_frame, sum(per_frame) / len(per_frame)

    def test_baseline_step_size_1(self):
        """Baseline: step_size=1 (watermark every frame)."""
        print("\n--- Baseline: step_size=1 ---")
        model = copy.deepcopy(self.model)
        model.step_size = 1
        model.eval()
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        
        _, acc = self._get_accuracy(model, self.video, msg)
        print(f"  Bit accuracy: {acc*100:.1f}%")
        
        self.assertGreater(acc, 0.95, f"Expected >95% accuracy, got {acc*100:.1f}%")

    def test_temporal_pooling_step_size_1(self):
        """Temporal pooling with step_size=1."""
        print("\n--- Temporal Pooling: step_size=1 ---")
        model = copy.deepcopy(self.model)
        model.step_size = 1
        model.embedder.time_pooling = True
        model.embedder.time_pooling_depth = 1
        model.embedder.time_pooling_kernel_size = 2
        model.embedder.time_pooling_stride = 2
        model.eval()
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        
        _, acc = self._get_accuracy(model, self.video, msg)
        print(f"  Bit accuracy: {acc*100:.1f}%")
        
        self.assertGreater(acc, 0.95, f"Expected >95% accuracy, got {acc*100:.1f}%")

    def test_video_mode_comparison(self):
        """Compare video_modes with step_size=4."""
        print("\n--- Video Mode Comparison (step_size=4) ---")
        print(f"{'Mode':<15} {'Mean Acc':>10}")
        print("-" * 27)
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        results = {}
        
        for video_mode in ["repeat", "alternate", "interpolate"]:
            model = copy.deepcopy(self.model)
            model.step_size = 4
            model.video_mode = video_mode
            model.eval()
            
            _, acc = self._get_accuracy(model, self.video, msg)
            results[video_mode] = acc
            print(f"{video_mode:<15} {acc*100:>9.1f}%")
        
        # "repeat" should have highest accuracy
        self.assertGreater(results["repeat"], results["alternate"],
            "repeat mode should beat alternate mode")

    def test_step_size_with_repeat_mode(self):
        """Test step_size effect with video_mode='repeat'."""
        print("\n--- Step Size Effect (video_mode='repeat') ---")
        print(f"{'Step Size':<12} {'Mean Acc':>10}")
        print("-" * 24)
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        results = {}
        
        for step_size in [1, 2, 4, 8]:
            model = copy.deepcopy(self.model)
            model.step_size = step_size
            model.video_mode = "repeat"
            model.eval()
            
            _, acc = self._get_accuracy(model, self.video, msg)
            results[step_size] = acc
            print(f"{step_size:<12} {acc*100:>9.1f}%")
        
        # With repeat mode, accuracy drops with step_size but stays >70%
        for step, acc in results.items():
            self.assertGreater(acc, 0.70, f"step={step} too low: {acc*100:.1f}%")

    def test_temporal_pooling_with_step_size(self):
        """Test temporal pooling combined with different step sizes."""
        print("\n--- Temporal Pooling + Step Size + Video Mode ---")
        print(f"{'Step':<6} {'TP':>6} {'Mode':<12} {'Mean Acc':>10}")
        print("-" * 38)
        
        torch.manual_seed(42)
        msg = torch.randint(0, 2, (1, self.nbits)).float()
        results = []
        
        configs = [
            {"step": 1, "tp": False, "mode": "repeat"},
            {"step": 1, "tp": True, "mode": "repeat"},
            {"step": 4, "tp": False, "mode": "repeat"},
            {"step": 4, "tp": True, "mode": "repeat"},
            {"step": 4, "tp": False, "mode": "alternate"},
            {"step": 4, "tp": True, "mode": "alternate"},
        ]
        
        for cfg in configs:
            model = copy.deepcopy(self.model)
            model.step_size = cfg["step"]
            model.video_mode = cfg["mode"]
            if cfg["tp"]:
                model.embedder.time_pooling = True
                model.embedder.time_pooling_depth = 1
                model.embedder.time_pooling_kernel_size = 2
                model.embedder.time_pooling_stride = 2
            model.eval()
            
            per_frame, acc = self._get_accuracy(model, self.video, msg)
            results.append({"cfg": cfg, "acc": acc, "per_frame": per_frame})
            
            tp_str = "Yes" if cfg["tp"] else "No"
            print(f"{cfg['step']:<6} {tp_str:>6} {cfg['mode']:<12} {acc*100:>9.1f}%")
        
        # Plot comparison
        self._plot_results(results)

    def _plot_results(self, results):
        """Generate per-frame accuracy plots: TP=False vs TP=True (repeat mode only)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        
        # Filter for repeat mode only
        repeat_results = [r for r in results if r['cfg']['mode'] == 'repeat']
        
        # Subplot 1: TP = False
        ax = axes[0]
        for r in repeat_results:
            if not r['cfg']['tp']:
                label = f"step_size={r['cfg']['step']} (mean={r['acc']*100:.1f}%)"
                ax.plot([a*100 for a in r["per_frame"]], marker='o', markersize=5, label=label)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Bit Accuracy (%)')
        ax.set_title('TP = False (video_mode=repeat)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Subplot 2: TP = True
        ax = axes[1]
        for r in repeat_results:
            if r['cfg']['tp']:
                label = f"step_size={r['cfg']['step']} (mean={r['acc']*100:.1f}%)"
                ax.plot([a*100 for a in r["per_frame"]], marker='o', markersize=5, label=label)
        ax.set_xlabel('Frame')
        ax.set_title('TP = True (video_mode=repeat)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = Path(__file__).parent / "temporal_pooling_full_comparison.png"
        plt.savefig(output_path, dpi=150)
        print(f"\nPlot saved to: {output_path}")
        plt.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
