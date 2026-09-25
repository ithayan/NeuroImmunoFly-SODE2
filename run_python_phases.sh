#!/bin/bash
echo "[Phase 2] Running PyTorch S-PINN Integration..."
python models/matlab_spinn_engine.py

echo "[Phase 3] Computing Validation Metrics..."
python evaluation/matlab_compute_metrics.py

echo "[Phase 4] Generating Publication-Grade Visualizations..."
python evaluation/matlab_plot_generator.py

echo "Python phases complete!"
