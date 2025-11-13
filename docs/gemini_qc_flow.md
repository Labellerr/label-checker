## QC Validation Flow

1. **Load inputs**
   - Accept directory of source images and paired COCO annotation file.
   - Parse categories, images, annotations into structured objects.

2. **Generate crops**
   - For each annotation:
     - Extract bounding boxes or polygon masks.
     - Crop the corresponding region from the source image with optional context padding.
     - Encode crop as PNG bytes for API submission.

3. **Prepare Gemini prompt**
   - Build instruction template with project label schema, expected class, and any guideline text.
   - Attach crop image as multimodal input.

4. **Request validation**
   - Call Gemini multimodal model with safety settings and temperature tuned for deterministic outputs.
   - Ask model to respond with JSON containing:
     ```json
     {
       "prediction_label": "<label_name>",
       "confidence": 0.0,
       "rationale": "<optional textual explanation>"
     }
     ```

5. **Compare with human label**
   - Compute boolean `is_match = (prediction_label == expected_label)` within tolerance for synonyms.
   - Capture Gemini confidence and rationale.

6. **Aggregate results**
   - Record per-annotation outcome with identifiers: image id, annotation id, bbox/mask metadata.
   - Produce summary statistics (accuracy, mean confidence, disagreement list).

### Result Schema

```json
{
  "image_id": 123,
  "annotation_id": 456,
  "expected_label": "traffic_light",
  "gemini_prediction": {
    "label": "traffic_light",
    "confidence": 0.91,
    "rationale": "Detected a traffic light with red signal inside the crop."
  },
  "is_match": true,
  "metadata": {
    "bbox": [x, y, width, height],
    "polygon": [[x1, y1], [x2, y2], "..."]
  }
}
```

Pipeline output: NDJSON or JSON array of entries plus metrics report per class.

