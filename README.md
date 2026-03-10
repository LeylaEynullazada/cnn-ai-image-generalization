# Real vs Fake Image Classification

## Structure
- `train/real/`, `train/fake/` — training images
- `test/real/`, `test/fake/` — test images

## Run
```bash
pip install -r requirements.txt
python scripts/test_pipeline.py
```

## Project Goal
Classify images as real or AI-generated(fake) using a CNN

## Run on Google Colab
1. Upload this repo to Colab (or clone from GitHub)
2. Upload 'train/' and 'test/' folders with your data, or mout Google Drive where the data is stored.
3. Set 'DATA_DIR' in the notebook to the folder containing 'train/' and 'test/' (e.g. '"."' if data is in repo root or '"/content/drive/MyDrive/..."' if on Drive).
4. Run all cells.

## Team
|Name|Contributions|
|------|-------------|
|Leyla Eynullazada|Data loading, preprocessing, basic modeling, and training loop, repository structure|
|Raul Aghayev|Further modeling, basic evaluation and experiments|
