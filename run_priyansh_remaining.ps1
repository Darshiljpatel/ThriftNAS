# Priyansh Track 2 - Remaining Tasks Script
# Run after MobileNetV2 completes

Write-Host "=== Task 3/4: Vanilla DARTS Search (lambda=0.0, 25 epochs) ===" -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1
python -u search.py --lam 0.0 --epochs 25 --batch_size 48 --out_dir results/search_lam0.0

Write-Host "=== Task 4/4: Retrain lambda=0.0 genotype (50 epochs) ===" -ForegroundColor Cyan
python -u retrain.py --genotype results/search_lam0.0/genotype.json --out_dir results/retrain_lam0.0

Write-Host "=== All Priyansh tasks DONE! ===" -ForegroundColor Green
