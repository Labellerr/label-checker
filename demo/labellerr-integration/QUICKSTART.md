# Quick Start Guide

Get started with Labellerr QC validation in 3 minutes.

## Step 1: Install Dependencies

```bash
cd /path/to/label-checker
pip install -r requirements.txt
```

## Step 2: Get Your API Keys

### Labellerr Credentials
- Log into your Labellerr account
- Navigate to Settings → API
- Copy: API Key, API Secret, Client ID
- Note your Project ID from the project page

### Gemini API Key
- Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- Click "Create API Key"
- Copy the key

## Step 3: Run the App

```bash
cd demo/labellerr-integration
python run_labellerr_qc.py
```

Open your browser to: `http://localhost:7860`

## Step 4: Validate Annotations

1. **Enter credentials** in the UI
2. **Select statuses** (e.g., "completed")
3. **Click "Fetch & Run QC"**
4. **Review results** and low-confidence annotations

## What Happens?

```
Your Labellerr Project
    ↓ (export by status)
COCO JSON with filtered annotations
    ↓ (download images)
Local images + annotations
    ↓ (crop + validate)
Gemini AI validation
    ↓
Results with confidence scores
```

## Output Location

Results saved to: `demo/labellerr-integration/output/`

- `annotations.json` - Exported COCO data
- `images/` - Downloaded images
- `crops/` - Annotation crops
- `qc_results.json` - Validation results

## Example Results

```
📊 SUMMARY STATISTICS
Total annotations:        50
Crops saved:              50
Gemini validations:       48
Average confidence:       0.870
Low confidence count:     5 (< 0.5)

Confidence by category:
  • Horse                        0.920
  • Dog                          0.830
```

## Next Steps

- Review low-confidence annotations in the UI gallery
- Adjust confidence threshold to find more issues
- Export results JSON for further analysis
- Check the full [README.md](README.md) for advanced options

## Troubleshooting

**Can't import Labellerr SDK?**
- Check the SDK path in `labellerr_fetcher.py` (line 19)
- Ensure SDK is at `../../../SDKPython-1/`

**No images downloaded?**
- Verify project has annotations with selected status
- Check network connectivity

**Gemini errors?**
- Verify API key is valid
- Check rate limits

See [README.md](README.md) for detailed troubleshooting.

