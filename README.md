# SAMS - Student Attendance Management System

CS402.3 (Computer Graphics and Visualization) coursework prototype. Reads a
photographed lecture signing sheet, works out which students signed it, and
stores/visualizes attendance.

## What it does

1. **`sams.py`** - processes one signing-sheet photo against the student
   roster in `info.xml`: greyscale -> denoise -> binarize -> locate the
   student table -> crop each row's Signature cell -> classify present/absent
   by ink density -> store the result in a local SQLite database
   (`data/sams.db`). Every intermediate step is saved as an image under
   `data/processed/<sheet-name>/` for the report screenshots.
2. **`infovis.py`** - shows a chart summarising one student's attendance
   record across every sheet processed so far.
3. **`investigate.py`** *(bonus)* - compares a student's signatures across
   sessions using ORB feature matching and flags any that don't look like
   they came from the same hand.

```
python sams.py data/signing_sheets/12.07.2019.jpeg info.xml
python infovis.py 10009301
python investigate.py 10009301
```

## Setup

```
python -m pip install -r requirements.txt
```

Requires Python 3.10+, OpenCV, NumPy and Matplotlib.

## Signing sheets

The five sheets from "CGV Signing Sheets.zip" are already in
`data/signing_sheets/`, renamed to `dd.mm.yyyy.jpeg` so the date can be read
straight from the filename (matching the coursework's own CLI example:
`python sams.py 10.07.2019.png info.xml`). To add more sheets, drop them in
the same folder with the same naming convention, then run `sams.py` once per
sheet, in date order:

```
for %f in (31.05.2019 21.06.2019 28.06.2019 05.07.2019 12.07.2019) do python sams.py data/signing_sheets/%f.jpeg info.xml
```

## Project layout

```
sams.py                  Entrypoint: process one signing sheet
infovis.py                Entrypoint: visualize one student's attendance
investigate.py             Entrypoint: signature consistency check (bonus)
info.xml                  Sample student roster
sams_core/
  config.py                Paths + the signing sheet's static layout constants
  models.py                 Student / SigningSession / AttendanceRecord
  xml_loader.py              Reads info.xml -> Student roster
  database.py                 SQLite schema + repository
  attendance_service.py        Glues roster + image pipeline + database together
  image_pipeline/
    preprocessing.py           Load, resize, greyscale, denoise
    binarization.py             Adaptive/Otsu thresholding
    layout.py                    Table + row-boundary detection
    cell_extractor.py             Crops each row's Signature cell
    signature_detector.py          Ink-density presence/absence classifier
    pipeline.py                     Orchestrates the stages above
  visualization/
    report.py                  Attendance summary statistics
    charts.py                   Matplotlib chart building
  recognition/
    feature_matcher.py          ORB keypoint similarity scoring
    verifier.py                  Cross-session signature consistency report
  utils/
    logging_utils.py            Step-by-step console progress
    image_io.py                  Saves intermediate images for the report
tests/                        pytest unit + integration tests
  generate_sample_sheet.py     Synthetic signing sheet generator (no real photos needed)
data/
  signing_sheets/              Put the input photos here
  signatures/                  Extracted signature crops (generated)
  processed/                   Step-by-step artifact images (generated)
  sams.db                      SQLite database (generated)
```

## Design notes / known limitations

- **Row matching is positional**, not OCR-based: the Nth detected table row
  is matched to the Nth student in `info.xml`. This mirrors the coursework's
  own worked example and avoids depending on a Tesseract install, but it does
  mean a sheet with a genuinely different row count/order than `info.xml`
  needs a matching `info.xml`.
- **Row detection has a fallback**: if the ruled lines can't be reconstructed
  reliably from a phone photo (faint ruling, motion blur), the table height is
  split evenly using the *known* number of students from `info.xml`, since the
  layout is static per the coursework brief.
- **Presence is based on ink density**, not handwriting recognition, so it
  can't semantically tell a genuine signature apart from an invigilator's
  "absent" annotation written in the same cell -- both count as "ink present".
  Worth a paragraph in the report's discussion/challenges section.
- **Signature verification (`investigate.py`)** needs at least two captured
  signatures for a student to compare. Measured against this project's own
  sample sheets, ORB similarity for genuine same-student comparisons landed
  around 0.06-0.15, while different-student comparisons landed around
  0.06-0.28 -- the two overlap substantially, because a cropped signature
  cell is only ~30x250px and leaves ORB little texture to work with. Treat a
  flagged mismatch as "worth a manual look" (and check the saved comparison
  chart), not a proven forgery. A more accurate approach would need
  higher-resolution captures and/or a purpose-built verifier (e.g. a Siamese
  network trained on genuine/forged pairs) rather than a generic keypoint
  matcher -- a good discussion point for the report.

## Testing

```
python -m pytest tests/ -v
```

Tests run against synthetically generated signing sheets
(`tests/generate_sample_sheet.py`), so the full suite runs without needing
the real scanned photos.
