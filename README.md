# Gemini QC Validation Pipeline

Validate human-labeled image annotations using Google's Gemini AI. This tool checks if your bounding boxes and polygons are correctly labeled by asking Gemini to verify each annotation.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a `data` folder with:
- `data/images/` - Your images (JPEG/PNG)
- `data/annotations.json` - COCO-format annotations

### 3. Get a Gemini API Key

Get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### 4. Run the Demo

```bash
export GEMINI_API_KEY="your-api-key-here"
python demo/run_demo.py
```

This will:
- Load your images and annotations
- Crop each annotated object
- Ask Gemini to verify the label
- Generate a report with confidence scores

**Results** are saved to `demo/output/demo_results.json`

## Testing Without Your Data

Run the unit tests to verify everything works:

```bash
pytest
```

This tests the cropping and JSON parsing logic without calling the Gemini API (free and fast).

## Advanced Usage

For production use with custom settings:

```bash
export GEMINI_API_KEY="your-api-key"
python -m qc_pipeline.run_validation \
  --images-dir data/images \
  --coco-json data/annotations.json \
  --output results.json \
  --padding 8
```

**Options:**
- `--model-name` - Change AI model (default: `gemini-3-flash-preview`)
- `--disable-polygon-mask` - Use rectangular crops instead of masked polygons
- `--guidelines` - Path to text file with labeling guidelines

## How It Works

1. **Loads** your COCO annotations
2. **Crops** each annotated object from the image
3. **Asks Gemini**: "Is this a [label]?"
4. **Reports** matches/mismatches with confidence scores

## Requirements

- Python 3.10+
- Gemini API key
- Images in JPEG/PNG format
- Annotations in COCO JSON format

## Documentation

- [Feasibility Assessment](docs/gemini_qc_feasibility.md)
- [Validation Flow Details](docs/gemini_qc_flow.md)
