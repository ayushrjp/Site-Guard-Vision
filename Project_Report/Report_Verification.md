# SiteGuardVision Report Verification

## Deliverables

- `SiteGuardVision_Final_Project_Report.docx` — editable Word package with editable paragraph text, heading styles, explicit page breaks, and 18 embedded image assets.
- `SiteGuardVision_Final_Project_Report.pdf` — PDF rendered from the same report source HTML.
- `SiteGuardVision_Figures/` — 12 generated technical diagrams and 6 genuine dashboard screenshots captured from the locally running application.
- `build_report.py` and `report_source.html` — reproducible report source files.

## Page and layout check

- The PDF begins with the `%PDF-1.4` signature.
- The exported PDF contains **exactly 70 page objects**, counted from its page tree/object markers after final export.
- The report source uses A4 dimensions, Times New Roman, 12 pt body text, 1.5 line spacing, 16 pt chapter-level headings, margins, running headers/footers, and explicit page breaks.
- The DOCX package was opened as a ZIP archive; `word/document.xml` parsed as well-formed XML and the package contains 18 embedded PNGs.
- Word automation was unavailable in this environment, so the DOCX was not opened and repaginated in Microsoft Word. PDF pagination is verified; Word may reflow content on another installation. Please open the DOCX in Word and inspect pagination before submission.
- The contents and figure list are present as editable text, not dynamically generated Word fields. Update page references if editing changes pagination.

## Source verification and evidence scope

The report was prepared from the inspected implementation and artifacts, including `server.py`, `infer_engine.py`, `storage.py`, `rag/`, `templates/`, `static/js/`, `dataset/data.yaml`, the training arguments/results, custom evaluation JSON reports, and test files. Metrics are attributed to the saved custom scripts and are not presented as official Ultralytics validation.

The report records the 90-image evaluation results, person/PPE separation, association and three-state logic, SQLite and optional RAG design, six-epoch saved training history, and the crowded-image annotation concern. It distinguishes implementation from live verification: the captured assistant state showed Ollama offline and zero indexed documents; tests mock external RAG services. No live Ollama/RAG generation is claimed.

The six included UI screenshots are genuine captures from the local application. The technical diagrams are newly generated explanatory diagrams based on inspected source; they are not application screenshots or model predictions.

## Remaining checks/placeholders

- Fill in student, institution, department, supervisor, academic year, certificate, signatures, and required declaration/acknowledgement details.
- Confirm institutional report template, citation style, approval requirements, and reference formatting.
- Verify all bibliography URLs and access dates, dataset attribution/license obligations, and any additional citations required by the institution.
- Review every PDF page and open the DOCX in the target Microsoft Word version; do not treat this report as an institutional approval or production-safety certification.
- Production source, dataset, and model checkpoints were not modified to create these report deliverables.
