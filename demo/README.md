# QC Pipeline Demo

End-to-end demonstration of the QC validation pipeline using sample test data.

## Overview

This demo showcases the complete workflow:

1. **Load** COCO annotations and images from `test_dataset/`
2. **Crop** each annotated object (bounding box or polygon)
3. **Validate** crops with Gemini using the prompt: *"Is this a [label]?"*
4. **Report** confidence scores and generate visual outputs

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r ../requirements.txt

# (Optional) Set your Gemini API key for validation
export GEMINI_API_KEY="your-api-key-here"
```

### Run the Demo

```bash
cd demo
python run_demo.py
```

**Note**: The demo works with or without a Gemini API key:
- **With API key**: Full validation with confidence scores
- **Without API key**: Crops are generated but validation is skipped

## What Happens

### Step-by-Step Flow

1. **Dataset Loading**
   - Reads `test_dataset/export-#6w0PnZX02NffcYns8L1C.json`
   - Contains 2 images with annotations for "Big Horse" and "baby horse"

2. **Crop Generation**
   - For each annotation, extracts the labeled region
   - Applies 8px padding and polygon masking
   - Saves crops to `demo/output/crops/`
   - Filenames: `crop_{annotation_id}_{label}.png`

3. **Gemini Validation** (if API key provided)
   - Sends crop + label to Gemini
   - Prompt: *"Look at this cropped image and determine if it contains a '{label}'"*
   - Returns: prediction_label, confidence (0-1), rationale

4. **Report Generation**
   - Saves `demo/output/demo_results.json` with:
     - Per-annotation results (image, label, crop path, confidence)
     - Summary statistics (totals, averages, confidence by category)

5. **Console Summary**
   - Prints progress and final statistics
   - Shows average confidence per category

## Output Structure

```
demo/output/
├── crops/
│   ├── crop_0_Big_Horse.png
│   ├── crop_1_baby_horse.png
│   └── ...
└── demo_results.json
```

### Sample `demo_results.json`

```json
{
  "results": [
    {
      "image_id": 0,
      "annotation_id": 0,
      "image_file": "burger.jpeg",
      "label": "Big Horse",
      "crop_path": "crops/crop_0_Big_Horse.png",
      "confidence": 0.95,
      "gemini_response": "{\"prediction_label\": \"Big Horse\", \"confidence\": 0.95, \"rationale\": \"...\"}",
      "error": null
    }
  ],
  "summary": {
    "total_annotations": 2,
    "total_crops_saved": 2,
    "gemini_validations": 2,
    "average_confidence": 0.92,
    "confidence_by_category": {
      "Big Horse": 0.95,
      "baby horse": 0.89
    }
  }
}
```

## Console Output Example

```
======================================================================
QC Pipeline Demo - End-to-End Validation
======================================================================

[1/5] Loading dataset from: /path/to/test_dataset/export-#6w0PnZX02NffcYns8L1C.json
  ✓ Loaded 2 annotations from 2 images
  ✓ Categories: Big Horse, baby horse

[2/5] Initializing Gemini validator...
  ✓ Gemini validator ready (demo prompt: 'Is this a [label]?')

[3/5] Processing annotations and generating crops...
  [1/2] Processing annotation 0 (Big Horse)... ✓ (confidence: 0.95)
  [2/2] Processing annotation 1 (baby horse)... ✓ (confidence: 0.89)

[4/5] Generating summary statistics...

[5/5] Saving results to: demo/output/demo_results.json
  ✓ Results saved

======================================================================
SUMMARY
======================================================================
Total annotations:     2
Crops saved:           2
Gemini validations:    2
Average confidence:    0.920

Confidence by category:
  Big Horse            0.950
  baby horse           0.890
======================================================================

✓ Demo complete! Check demo/output for results.
```

## Customization

Edit `run_demo.py` to customize:

- **Dataset path**: Change `dataset_dir` and `coco_json` variables
- **Output location**: Modify `output_dir`
- **Crop settings**: Adjust `CropConfig(padding=..., mask_polygon=...)`
- **Gemini model**: Change model name or temperature in `create_demo_validator()`

## Future Extensions

The demo is designed to support additional validation models:

```python
# Placeholder for open-source VLMs
# TODO: Add CLIP, LLaVA, or other vision models
# validator_opensource = create_opensource_validator(...)
# response_opensource = validator_opensource.validate_crop(...)
```

## Troubleshooting

**No crops generated?**
- Verify `test_dataset/` contains images and COCO JSON
- Check image filenames match the `file_name` fields in JSON

**Gemini errors?**
- Confirm `GEMINI_API_KEY` is set correctly
- Check API quota/rate limits
- Review error messages in console output

**Low confidence scores?**
- Inspect crops in `demo/output/crops/` visually
- Labels may not match actual image content (expected for QC testing)
- Adjust prompt template in `qc_pipeline/gemini_validator_demo.py`

