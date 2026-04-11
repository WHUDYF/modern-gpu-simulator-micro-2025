import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "summarize_runs.py"


class SummarizeRunsTest(unittest.TestCase):
    def test_collects_basic_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "model_gpt2_ctx128_gen1_run1"
            traces = run_dir / "traces"
            traces.mkdir(parents=True)

            (run_dir / "run.log").write_text(
                "model=gpt2\ncontext_len=128\ngen_tokens=1\ndecode_time_s=0.123456\n",
                encoding="utf-8",
            )
            (traces / "dynamic_trace.pb").write_bytes(b"x" * 1024)
            (traces / "stats.csv").write_text("kernel_id,kernel_name\n0,attention_kernel\n", encoding="utf-8")
            extra = traces / "extra_info" / "enhanced_execution_info.json"
            extra.parent.mkdir(parents=True)
            extra.write_text("{}", encoding="utf-8")
            tb = traces / "threadblocks"
            tb.mkdir()
            (tb / "tb0.pb").write_bytes(b"y" * 2048)

            out_csv = root / "summary.csv"
            subprocess.run(
                [sys.executable, str(SCRIPT), "--results-root", str(root), "--output", str(out_csv)],
                check=True,
            )

            with out_csv.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["model"], "gpt2")
            self.assertEqual(row["context_len"], "128")
            self.assertEqual(row["gen_tokens"], "1")
            self.assertEqual(row["run_id"], "1")
            self.assertEqual(row["num_kernel_rows"], "1")
            self.assertEqual(row["has_dynamic_trace_pb"], "1")


if __name__ == "__main__":
    unittest.main()
