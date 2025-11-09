Set-StrictMode -Version Latest
python -c "import sys, os; sys.path.insert(0, os.path.join(os.getcwd(), 'src')); from process import run_full_pipeline; run_full_pipeline(n_steps=200)"
