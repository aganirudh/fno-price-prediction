import os
import re

files = [
    'main.py',
    'training/ensemble_train.py',
    'training/hybrid_train.py',
    'backtest/ensemble_backtest.py',
    'data_pipeline/kaggle/dataset_loader.py',
    'data_pipeline/kaggle/fundamentals_processor.py',
    'data_pipeline/kaggle/ensemble_data_prep.py',
    'models/ensemble/regime_detector.py',
    'models/ensemble/ensemble_selector.py',
    'models/ensemble/hybrid_controller.py',
    'monitoring/dashboard.py',
    'pcp_arb_env/observations.py'
]

def clean_file(path):
    if not os.path.exists(path):
        return
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace common emojis/non-ascii with safe text
    replacements = {
        '✅': '[OK]',
        '🔄': '[Update]',
        '📈': '[Chart]',
        '📊': '[Stats]',
        '🎯': '[Target]',
        '🤖': '[AI]',
        '📉': '[Down]',
        '🏥': '[Health]',
        '—': '-',
        '₹': 'INR',
        '': ' ',
    }
    
    new_content = content
    for k, v in replacements.items():
        new_content = new_content.replace(k, v)
    
    # Remove any other non-ascii characters just in case
    new_content = ''.join([c if ord(c) < 128 else ' ' for c in new_content])
    
    if new_content != content:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Cleaned {path}")

for f in files:
    clean_file(f)
