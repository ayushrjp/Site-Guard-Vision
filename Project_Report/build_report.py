"""Build the isolated SiteGuardVision report source and its technical figures."""
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "SiteGuardVision_Figures"
FIG.mkdir(exist_ok=True)

# Evidence-led chapter plan: exactly 70 planned A4 pages, including front matter.
sections = [
("Preliminary", "Cover Page", "SiteGuardVision — AI-Powered Construction Site Safety Monitoring and PPE Compliance System. Final-year engineering project report. Submitted by: [Student name and enrolment number]. Department: [Department]. Institution: [University/College]. Supervisor: [Supervisor name]. Academic year: [Year]. All institutional and approval particulars intentionally remain placeholders."),
("Preliminary", "Certificate", "This page is reserved for the institution’s prescribed project certificate. The project title and student particulars should be completed only after institutional review. Supervisor, Head of Department, external examiner, dates and signatures are placeholders; no approval is represented by this draft."),
("Preliminary", "Declaration", "I/We declare that this report describes the SiteGuardVision implementation and its available evidence. Any third-party materials and dataset licensing must be acknowledged in the final institutional copy. Student name, registration number, date and signature: [To be completed by the student]."),
("Preliminary", "Acknowledgement", "The authors may acknowledge their supervisor, department, institution, dataset contributors and others who provided verified assistance. Names and contributions should be added by the student. This draft does not assert any specific institutional endorsement or contribution."),
("Preliminary", "Abstract", "SiteGuardVision is an image-only Flask application for reviewing construction-site photographs. Separate YOLO-based checkpoints detect PPE items and people; geometry-based association relates detected hard hats and vests to worker boxes. The application reports Potential Violation, No Violation Detected, or Insufficient Evidence rather than asserting legal compliance. SQLite persists analysis history and settings. An optional document-grounded assistant combines PDF/TXT/Markdown ingestion, ChromaDB retrieval and local Ollama generation. Custom evaluation on 90 test images reported PPE-only macro precision 0.842, recall 0.734, mAP50 0.847 and mAP50–95 0.521. The full application reported 21 potential violations, 5 no-violation detections and 64 insufficient-evidence outcomes; annotation incompleteness and person-count error constrain interpretation. Ollama/RAG operation was not live-verified in the captured environment."),
("Preliminary", "Contents and Report Conventions", "Contents — 1 Introduction, 2 Problem Statement and Objectives, 3 Literature Survey and Existing Systems, 4 Proposed System and Methodology, 5 System Requirements and Technology Stack, 6 System Architecture and Design, 7 Dataset Preparation and Model Development, 8 Computer Vision and PPE Detection Implementation, 9 Worker-to-PPE Association and Safety Assessment, 10 Flask Dashboard and Frontend Implementation, 11 Database and RAG Safety Assistant, 12 Testing, Results and Performance Evaluation, 13 Limitations and Future Scope, 14 Conclusion, References and Appendices. The report follows the supplied chapter plan and allocates one planned page to each numbered entry. Headings use decimal chapter numbering; figures and tables are numbered by chapter. Page count is verified against the exported PDF; update the Word contents and page numbers if local edits alter pagination."),
("Preliminary", "List of Figures and Tables", "Figures: 6.1 System architecture; 6.2 Frontend/backend architecture; 6.3 Image workflow; 6.4 Detection pipeline; 6.5 Two-model workflow; 9.1 PPE association; 9.2 Safety decision flow; 11.1 SQLite persistence; 11.2 RAG ingestion; 11.3 RAG question-answering; 6.6 Flask request-response; 12.1 Evaluation workflow. Supporting genuine interface screenshots appear in Chapters 8, 10, 11 and 12. Tables enumerate dataset, configuration, metric, schema and test evidence."),
]

chapter_pages = [
("1 Introduction", [
"Construction sites contain changing work fronts, temporary access routes, variable lighting and workers who may be partially occluded. Personal protective equipment (PPE) such as hard hats and high-visibility vests reduces exposure to certain hazards when correctly selected and worn. Manual observation can be intermittent, while fixed cameras and computer-vision models offer a way to flag image evidence for human review. A detector does not understand the complete work context and must not be treated as a safety authority.",
"The engineering problem addressed here is deliberately narrower than continuous surveillance: a user submits one photograph, the application detects people and PPE, associates evidence spatially, and presents a cautious assessment with an annotated image. It is an image-review decision-support prototype, not a live video monitor, identity system, autonomous enforcement mechanism or substitute for site inspection.",
"The software combines a Flask server, two YOLO checkpoints, image-association logic, an SQLite store and a browser dashboard. Optional components include a local ChromaDB vector collection and Ollama language models for retrieving safety-document passages. The separation between core visual analysis and optional assistant services allows the application to remain useful when the local language-model service is unavailable.",
"The project is evaluated at multiple levels: detection against labels, person detection, application-level status agreement, latency and regression tests. Each level answers a different question. Box metrics measure localization/classification; application status metrics compare image-level decisions to annotation-derived labels; neither establishes actual legal compliance or operational safety effectiveness.",
"This report records source behavior and artifacts present in the project at inspection time. Where a capability was not live-verified, it is identified as such. Numeric values are reproduced from saved JSON evaluation reports rather than newly generated claims. Missing training epochs and annotation gaps are treated as limitations, not silently interpolated.",
]),
("2 Problem Statement and Objectives", [
"The core problem is to make PPE evidence in a construction photograph easier to review without converting uncertain detections into categorical compliance claims. False negatives can hide a missing item; false positives can create unnecessary escalation. A third outcome—insufficient evidence—makes uncertainty visible when detected equipment cannot be confidently associated or workers are not adequately observed.",
"The implemented objectives are to accept a photo or bundled sample, run separate PPE and person detectors, associate positive and explicitly negative PPE detections to worker regions, display counts and per-worker evidence, retain analysis history, expose model/evaluation information, and provide a document-grounded assistant when its dependencies and index are available.",
"The system is explicitly not a real-time camera-stream platform, identity tracker, incident-response system, regulatory certification, automated disciplinary tool, or proof that unmarked workers comply. Its output is a screening cue. A human reviewer must inspect the original image, consider site rules, and decide whether further action is appropriate.",
"Success criteria include transparent status semantics, traceable evidence, stable upload validation, useful history, reproducible evaluation scripts, safe behavior when optional RAG services are down, and visible limitations. The saved application evaluation indicates that cautious status behavior is common: 64 of 90 images returned insufficient evidence at the recorded threshold and size.",
]),
("3 Literature Survey and Existing Systems", [
"Object detection methods predict object categories and bounding boxes in one or more image passes. YOLO-family systems are commonly selected when a practical balance between latency and detection quality is needed. This project uses pretrained Ultralytics YOLO checkpoints for inference and a recorded YOLOv8n training configuration. The report does not claim that this implementation is novel relative to published detector architectures.",
"Conventional PPE inspection includes direct supervisor observation, checklists, access-control procedures and camera-assisted review. Human inspection provides context that image models lack, but coverage depends on time and vantage point. Camera analytics can scale review, yet performance is sensitive to resolution, angle, occlusion, lighting, worker density and annotation quality.",
"A detector typically produces class, confidence and box coordinates. Non-maximum suppression (NMS) reduces overlapping predictions within inference, while subsequent association logic assigns PPE evidence to people. These are distinct steps: NMS does not solve worker-to-equipment assignment and box overlap alone does not establish that an item is worn by a particular worker.",
"Precision is TP/(TP+FP) and recall is TP/(TP+FN). Average precision integrates precision over recall; mAP averages AP across classes and, in mAP50–95, IoU thresholds from 0.50 to 0.95. Metric definitions depend on matching, class mapping, confidence filtering and averaging. The included custom scripts state their greedy IoU matching and 101-point interpolation; their scores must not be described as official Ultralytics validation.",
"Retrieval-augmented generation (RAG) retrieves relevant passages from a local document collection and supplies them to a language model. This can make answers traceable to supplied materials, but citations indicate retrieved context rather than legal correctness. SiteGuardVision’s assistant code adds an insufficient-context response and an offline fallback; the environment screenshot showed no indexed documents and Ollama offline.",
]),
("4 Proposed System and Methodology", [
"The user selects a sample or uploads a supported image. The server validates the request, loads application settings, calls the inference engine and persists a structured result. The browser then presents the source/annotated image, status, detector metadata, timing, counts and worker-level PPE evidence. A stored record may be reopened through its analysis identifier.",
"The PPE model and person detector are distinct. The PPE checkpoint emits PPE-related classes; a separate COCO person checkpoint supplies worker candidates because the PPE model’s own Person class did not produce useful detections in diagnostics. Keeping source labels separate prevents one detector’s output from being misrepresented as the other’s.",
"Association uses worker geometry to define head and torso regions and associates PPE boxes using overlap/centre tests and one-to-one matching safeguards. Explicit no-helmet/no-vest boxes can support a potential-violation state. Missing positive detections without explicit negative evidence are not treated as confirmed violations; they contribute uncertainty.",
"A positive explicit violation detection yields Potential Violation. A fully supported set of worker-level positive PPE evidence may yield No Violation Detected. Cases with no workers, absent PPE evidence, unassigned detections or unresolved worker evidence remain Insufficient Evidence. These labels intentionally avoid a blanket ‘compliant’ verdict.",
"Evaluation consists of custom box-level metrics, separate person-model evaluation, application-level comparison to annotation-derived violation labels, unit/route tests and sample-image inference tests. Saved results use 90 test images. They inherit the dataset’s annotation limitations and do not establish generalization outside that split.",
]),
("5 System Requirements and Technology Stack", [
"The application is implemented in Python with Flask 3.0.3, Ultralytics 8.3.40, Pillow 10.4.0, ChromaDB 1.5.9, pypdf 5.9.0 and Requests 2.34.2 as pinned in requirements.txt. The interface uses Jinja templates, CSS and browser-native JavaScript. Persistence uses Python’s SQLite support; optional vector retrieval uses ChromaDB.",
"Model execution requires the project checkpoint files and a compatible Ultralytics/PyTorch/OpenCV environment. The saved training arguments record CPU execution, batch size 2, 512 training image size, 30 requested epochs, patience 10, seed 42 and deterministic mode. This is recorded configuration, not a complete hardware specification.",
"The default operational settings in storage.py are confidence 0.25, image size 960 and Ollama chat model qwen2.5:3b. Allowed inference sizes are 640, 960 and 1280; confidence is bounded between 0.05 and 0.95. Embedding defaults are configured separately in rag/config.py as nomic-embed-text.",
"The web application is designed to continue operating when optional RAG dependencies or Ollama are unavailable. Operational deployment still requires upload-size controls, trusted network boundaries, access control, retention policy, dependency updates and model-file integrity procedures; the report does not claim these deployment controls are implemented.",
]),
("6 System Architecture and Design", [
"The architecture separates request handling, inference, storage and optional retrieval. Flask routes coordinate user requests and responses; infer_engine owns model loading and image analysis; storage owns SQLite records and settings; rag modules handle document loading, embedding, vector persistence, retrieval and assistant prompting. Templates render server-side page structure while JavaScript handles responsive navigation and asynchronous assistant interactions.",
"Figure 6.1 summarizes the system components and their trust boundaries. A user-provided image travels to the local server and inference models; a structured result is persisted in SQLite. The optional assistant uses safety documents, embeddings, ChromaDB retrieval and Ollama generation. The assistant is not in the critical visual inference path.",
"Figure 6.2 shows the browser/server division. HTML templates are rendered with Jinja, static styles/scripts are served by Flask, and JSON APIs expose assistant status, documents, settings and chat operations. The interface’s health indicators represent observed service state, not a guarantee of availability.",
"Figure 6.3 describes upload handling through validation, decoding, inference, persistence and rendering. Errors are returned without implying that an image was assessed. Analysis history stores structured output and metadata; settings are validated before persistence.",
"Figure 6.4 presents detection as model output plus postprocessing. Input resizing and model-specific preprocessing are delegated to the Ultralytics prediction interface. Predictions carry classes, confidence and coordinates; class mapping, geometry and assessment are application logic layered after detector inference.",
"Figure 6.5 emphasizes two independent detection sources. Person candidates come from the dedicated COCO model; hard-hat/vest and explicit absence detections come from the PPE model. The application combines evidence only at the association stage. Figure 6.6 shows the Flask request-response path from route dispatch to template/JSON output.",
]),
("7 Dataset Preparation and Model Development", [
"The dataset configuration at dataset/data.yaml defines five labels: helmet, no-helmet, no-vest, person and vest. The recorded split comprises 997 training, 119 validation and 90 test images, totaling 1,206. The dataset YAML identifies a Roboflow workspace/project and CC BY 4.0 license; users should confirm license terms before redistribution.",
"Training records at runs/detect/siteguard_ppe_5class/args.yaml describe a YOLOv8n initialization, 30 planned epochs, image size 512, batch 2, CPU device, seed 42, deterministic setting and early-stopping patience 10. The available results.csv contains only six epoch rows. It is therefore not evidence of a complete 30-epoch run, and no missing epochs are reconstructed here.",
"The evaluation mapping remaps dataset IDs 0→3, 1→8, 2→10, 3→11 and 4→13 for the PPE checkpoint. The custom script evaluates 90 test images with confidence 0.25 for P/R and 0.001 for AP, NMS IoU 0.7 and IoU thresholds 0.50 through 0.95. The separate person checkpoint is evaluated against class-3 person annotations.",
"Dataset annotation completeness is a material concern. The crowded test image ppe_0675… contains visibly more people than its three ground-truth person boxes; the saved application report recorded nine predicted persons for that image. This example is evidence of a possible annotation mismatch, not a complete audit of the dataset.",
"A previous annotation-screening run was stopped before completion; only a partial screen exists and it is not treated as a verified dataset-wide audit. Future model work should conduct a documented annotation review, preserve train/validation/test separation, report corrected labels and rerun evaluations on a versioned dataset.",
]),
("8 Computer Vision and PPE Detection Implementation", [
"The inference engine uses two YOLO model instances loaded from models/yolo_ppe_core.pt and models/person_yolov8n.pt. Model loading is protected to avoid repeated concurrent initialization. Prediction settings include configured confidence and image size; the selected runtime values are preserved with analysis metadata where available.",
"Object detection returns a set of candidate bounding boxes with class identities and confidence scores. Coordinates are interpreted relative to the decoded image; confidence is a model score, not a calibrated probability of legal violation. Resizing/letterboxing, tensor preparation and detector-specific postprocessing are provided by Ultralytics rather than reimplemented as a bespoke training pipeline.",
"PPE model classes include hardhat, no-hardhat, no-safety-vest and safety-vest using checkpoint labels after mapping. The dataset’s person category is not used as the application’s authoritative person count: person detection is delegated to the separate COCO model. This prevents conflating a weak PPE-model person class with the dedicated detector.",
"Non-maximum suppression is used in the model inference configuration to suppress redundant overlapping predictions. The application also handles duplicate/overlapping candidates before assignment where appropriate. Neither suppression nor a high confidence threshold repairs missed labels or detects PPE hidden by occlusion.",
"Figure 8.1 is a genuine captured analysis screen, and Figure 8.2 is a saved annotated analysis output when available. Screenshots document the interface state at capture time; they are not controlled benchmark evidence. Image processing remains still-image based and no continuous video pipeline is implemented.",
"The user-facing result includes an annotated image, detection counts and worker evidence. Visualization aids inspection but should not obscure the original photo. A practical review should compare both versions, especially for small, occluded or crowded workers.",
]),
("9 Worker-to-PPE Association and Safety Assessment", [
"For each detected person box, the engine derives smaller head and torso regions from relative coordinates. A hard-hat candidate is considered against the head region; vest evidence is considered against the torso region. Geometric overlap and centre-point inclusion provide complementary cues when predicted object boxes have different sizes.",
"Intersection over Area (IoA) divides the intersection area by the area of a selected box, unlike IoU, which divides by the union. IoA is useful when one object box is substantially contained within a worker subregion. The implementation uses configured geometric thresholds; this is a heuristic association, not pose estimation or a learned person-item relation.",
"Matching is constrained to avoid assigning one PPE detection to multiple workers. Duplicate detections are suppressed and detections that cannot be assigned remain unassigned evidence. In crowded scenes, overlapping worker regions, truncated bodies and perspective can still yield ambiguous or incorrect assignments.",
"Per-worker states distinguish Present, Missing and Unconfirmed evidence. A positive PPE detection associated with the relevant body region supports Present. An explicit no-PPE detection can support Missing. If there is no reliable evidence, the correct state is Unconfirmed rather than assuming absence or presence.",
"Figure 9.1 illustrates the association logic; Figure 9.2 depicts the conservative assessment flow. Potential Violation requires explicit negative-PPE evidence. No Violation Detected means no explicit violation was detected and the evidence was adequate for the application’s rule, not that the worker is legally compliant. Insufficient Evidence covers uncertainty, including no visible people.",
]),
("10 Flask Dashboard and Frontend Implementation", [
"Flask exposes the dashboard at /, image analysis at POST /analyze, stored analysis at /analysis/<id>, assistant and report pages, CSV export, evaluation and settings pages, sample-file serving, health endpoints and assistant/settings APIs. The route inventory is documented in server.py’s module header and implementation.",
"Jinja templates inherit a shared layout and render page-specific content. The navigation includes Analysis, Safety Assistant, Reports, Dataset & Evaluation and Settings. CSS provides the dark responsive operations dashboard. Vanilla JavaScript handles navigation selection, mobile sidebar controls and assistant interactions without a frontend framework.",
"The analysis page presents upload/sample selection and result summaries. The detail view exposes the annotated image and assessment evidence. Reports lists persisted analyses and provides CSV export; Evaluation displays available report summaries; Settings provides configurable thresholds and document management controls.",
"The Safety Assistant browser code submits questions and renders answers, sources, timing and modes. Browser-side source labels are supplied by the server. The interface’s current screenshot showed Ollama offline and no indexed documents; therefore chat generation and live retrieval are not claimed as verified in this session.",
"Figure 10.1 shows a genuine captured dashboard state. Figures 10.2 and 10.3 show stored analysis and report history. These screenshots are included as source artifacts and should be viewed at original resolution for legibility.",
]),
("11 Database and RAG Safety Assistant", [
"SQLite persistence is implemented in storage.py. The database path defaults to data/siteguard.db and can be overridden with SITEGUARD_DB_PATH. Schema creation is idempotent and each operation opens a short-lived connection. Stored analysis records support history/reopening; settings persist confidence, image size and chat model defaults.",
"The RAG subsystem is modular. Document loading supports PDF via pypdf and plain text/Markdown, cleans text and creates paragraph-aware overlapping chunks with source page metadata. Ingestion computes a SHA-256 digest, embeds in batches and maintains a manifest so unchanged documents can be skipped.",
"OllamaEmbeddings calls the local embedding API and supports modern /api/embed with fallback to /api/embeddings. ChromaDB stores vectors persistently under the configured vector directory. Retrieval selects top-k passages and applies a minimum-similarity threshold. The current manifest recorded zero documents, so no knowledge passages were available at capture time.",
"The assistant separates structured image facts from general safety guidance. Image facts are summarized from analysis data; policy guidance must be grounded in retrieved documents. If context is insufficient, the code returns a fixed insufficient-information message rather than improvising. If Ollama is offline, the core application remains available and assistant status reports the failure.",
"RAG components are implemented and covered by unit tests that mock external services. The README and captured UI indicate Ollama was unavailable during implementation/verification. Accordingly, live embedding, indexing and Qwen2.5 generation were not independently verified in this report run.",
]),
("12 Testing, Results and Performance Evaluation", [
"The PPE-only custom evaluation covers 90 test images and 623 labeled objects. Macro P/R/mAP50/mAP50–95 are 0.842/0.734/0.847/0.521 over PPE classes only. Including the PPE checkpoint’s ineffective Person class changes the macro scores; the report uses PPE-only values when discussing PPE and reports person detection separately.",
"Classwise PPE results are: helmet P 0.836, R 0.887; no-helmet P 0.842, R 0.667; no-vest P 0.809, R 0.623; vest P 0.883, R 0.760. Lower recall for negative classes means some annotated missing-equipment examples may not be detected. All values are custom greedy IoU/101-point AP results, not official Ultralytics validation.",
"The dedicated person detector report records P 0.7398, R 0.9299, F1 0.824, AP50 0.8629 and AP50–95 0.4581. At confidence 0.25, it reports 199 true positives, 70 false positives and 15 false negatives among 269 predictions against 214 annotated people.",
"The full application evaluation at image size 960 and confidence 0.25 covers 90 images: 64 insufficient evidence, 21 potential violations and 5 no-violation detections. Annotation-derived violation agreement is precision 1.0, recall 0.5833, F1 0.7368 and accuracy 0.8333 (21 TP, 0 FP, 15 FN, 54 TN). These are annotation agreement values, not ground-truth compliance rates.",
"Person-count MAE is 1.24; exact count match is 45.56%. Mean total time is 4,792.6 ms, with 4,469.7 ms model inference (3,648.3 ms PPE and 821.4 ms person). The environment’s hardware is not characterized in the report; latency should not be generalized.",
"Tests include pure unit checks, Flask route/validation/rendering tests, real CPU sample inference when weights are available, and RAG tests with mocked services. The test suite was inspected; no fresh full test-suite execution is asserted here. Figure 12.1 shows the layered evaluation workflow.",
]),
("13 Limitations and Future Scope", [
"Current limitations include still-image-only operation, sensitivity to camera viewpoint and occlusion, heuristic geometric association, person false positives/count error, incomplete or inconsistent dataset annotations and a small 90-image test split. Saved training history contains only six epoch rows despite a 30-epoch target.",
"The most consequential semantic limitation is that no explicit violation detection cannot establish compliance. The application partly addresses this with Insufficient Evidence and carefully worded statuses, but human review remains necessary. Annotation-derived negative labels may themselves be incorrect when missing boxes are simply unannotated.",
"Future work should first audit labels under a versioned protocol, strengthen evaluation with independent reviewers and stratified scenarios, measure calibration and confidence sensitivity, and compare association strategies. Any continuous-stream extension requires temporal tracking, privacy safeguards, workload testing and fail-safe human escalation.",
"Operational extensions could include authentication, role-based access, retention controls, deployment monitoring, model version pinning and an independently tested RAG corpus. Live Ollama/RAG validation should include document ingestion, citation traceability, adversarial prompts and failure recovery. These are recommendations, not implemented claims.",
]),
("14 Conclusion", [
"SiteGuardVision implements a cautious image-based PPE review workflow combining separate person and PPE detectors, geometry-based assignment, a three-state assessment, Flask/Jinja dashboard, SQLite history and an optional local document-grounded assistant. Saved custom evaluations show promising PPE precision alongside missed detections, person-count errors and incomplete annotation concerns. The 90-image application evaluation’s 64 insufficient-evidence outcomes underscore the system’s conservative design and current limits. The prototype is decision support only; it does not certify compliance. Before operational use, annotation quality, model performance, RAG live behavior, privacy and deployment controls require further independent verification.",
]),
("Appendices", [
"Appendix A — Runtime configuration: storage defaults confidence 0.25, image size 960 and Ollama model qwen2.5:3b; valid image sizes are 640, 960 and 1280; confidence bounds are 0.05–0.95. RAG defaults in rag/config.py specify localhost:11434, qwen2.5:3b and nomic-embed-text. Environment variables can override service and storage paths. These are defaults, not a prescribed production deployment.",
"Appendix B — Representative test inventory: geometry and class mapping; PPE association and assessment; upload validation; Flask route rendering; persistence and export; document text cleaning/chunking; citation and insufficient-context logic; mocked Ollama status/generation; and optional real CPU inference against bundled samples. Check tests/test_siteguard.py and tests/test_rag_assistant.py for exact assertions.",
"Appendix C — Evidence and reproducibility: primary metric artifacts are evaluation/results/evaluation_report.json, person_evaluation_report.json and application_evaluation_report_960_dupfix.json. Training metadata is runs/detect/siteguard_ppe_5class/args.yaml and results.csv. Reproduction requires the same model files, dataset split, dependency versions, thresholds and custom scripts. Reported custom metrics are not directly interchangeable with Ultralytics validation.",
"Appendix D — Selected references: [1] J. Redmon, S. Divvala, R. Girshick and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” CVPR, 2016, pp. 779–788, doi:10.1109/CVPR.2016.91. [2] Ultralytics, YOLO documentation, https://docs.ultralytics.com/. [3] Pallets, Flask documentation, https://flask.palletsprojects.com/. [4] SQLite, Documentation, https://sqlite.org/docs.html. [5] Chroma, Documentation, https://docs.trychroma.com/. [6] Ollama, API documentation, https://github.com/ollama/ollama/blob/main/docs/api.md. [7] pypdf, Documentation, https://pypdf.readthedocs.io/. Accessed 15 November 2026. The dataset YAML attributes the dataset to Roboflow and states CC BY 4.0; verify attribution requirements. Institutional citation style and full bibliographic details should be checked before submission.",
]),
]

pages = list(sections)
planned_chapter_pages = {
    "1 Introduction":4, "2 Problem Statement and Objectives":3,
    "3 Literature Survey and Existing Systems":5, "4 Proposed System and Methodology":5,
    "5 System Requirements and Technology Stack":4, "6 System Architecture and Design":6,
    "7 Dataset Preparation and Model Development":6,
    "8 Computer Vision and PPE Detection Implementation":6,
    "9 Worker-to-PPE Association and Safety Assessment":4,
    "10 Flask Dashboard and Frontend Implementation":4,
    "11 Database and RAG Safety Assistant":4,
    "12 Testing, Results and Performance Evaluation":5,
    "13 Limitations and Future Scope":2, "14 Conclusion":1, "Appendices":4,
}
for chapter, topics in chapter_pages:
    if chapter == "7 Dataset Preparation and Model Development":
        topics = topics + ["Reproducibility requires retaining exact checkpoints, software versions, mapping rules and split manifests. Re-running the custom scripts on revised annotations changes the experiment and must be reported as a new version rather than silently replacing these scores."]
    topics = topics[:planned_chapter_pages[chapter]]
    for i, text in enumerate(topics, 1):
        title = chapter if i == 1 else f"{chapter} — {['Scope and Context','Implementation Detail','Evidence and Interpretation','Design Considerations','Summary'][ (i-2) % 5 ]}"
        pages.append(("Chapter", title, text))
assert len(pages) == 70, len(pages)

diagrams = {
    "Figure_6_1_System_Architecture.png": (["Browser UI","Flask routes","Inference engine","SQLite history","Optional RAG","Ollama + ChromaDB"], [(0,1),(1,2),(1,3),(1,4),(4,5)]),
    "Figure_6_2_Frontend_Backend.png": (["Jinja templates","CSS / JS","HTTP / JSON","Flask endpoints","Storage + services"], [(0,2),(1,2),(2,3),(3,4)]),
    "Figure_6_3_Image_Workflow.png": (["Upload / sample","Validate + decode","Two model inference","Associate evidence","Assess + persist","Render result"], [(0,1),(1,2),(2,3),(3,4),(4,5)]),
    "Figure_6_4_YOLO_Pipeline.png": (["Image","Preprocess","YOLO backbone / head","Boxes + scores","NMS / mapping","Application output"], [(0,1),(1,2),(2,3),(3,4),(4,5)]),
    "Figure_6_5_Two_Model_Workflow.png": (["Site photo","PPE checkpoint","Person checkpoint","Separate detections","Association layer","Worker evidence"], [(0,1),(0,2),(1,3),(2,3),(3,4),(4,5)]),
    "Figure_9_1_Association.png": (["Person box","Head / torso regions","IoA + centre checks","One-to-one assignment","Present / Missing / Unconfirmed"], [(0,1),(1,2),(2,3),(3,4)]),
    "Figure_9_2_Assessment_Flow.png": (["Input evidence","Explicit no-PPE found?","Yes → Potential Violation","Evidence adequate?","Yes → No Violation Detected","No → Insufficient Evidence"], [(0,1),(1,2),(1,3),(3,4),(3,5)]),
    "Figure_11_1_Database_Architecture.png": (["Flask handlers","SQLite connection","analyses table","settings table","CSV / history views"], [(0,1),(1,2),(1,3),(2,4)]),
    "Figure_11_2_RAG_Ingestion.png": (["PDF / TXT / MD","Clean + chunk","SHA-256 manifest","Ollama embeddings","ChromaDB index"], [(0,1),(1,2),(1,3),(3,4)]),
    "Figure_11_3_RAG_QA.png": (["Question","Embed + retrieve","Relevant chunks","Grounded prompt","Ollama chat","Answer + citations"], [(0,1),(1,2),(2,3),(3,4),(4,5)]),
    "Figure_6_6_Flask_Request.png": (["Browser request","Route + validation","Inference / storage","Template or JSON","Browser response"], [(0,1),(1,2),(2,3),(3,4)]),
    "Figure_12_1_Evaluation_Workflow.png": (["Versioned test split","PPE box metrics","Person metrics","Application agreement","Manual error review","Report limitations"], [(0,1),(0,2),(1,3),(2,3),(3,4),(4,5)]),
}

def draw(name, labels, edges):
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    n=len(labels); positions=[(0.12+i*0.76/max(1,n-1), 0.52+0.19*(i%2)) for i in range(n)]
    for a,b in edges:
        ax.add_patch(FancyArrowPatch(positions[a],positions[b],arrowstyle="-|>",mutation_scale=13,
                     linewidth=1.25,color="#35678a",connectionstyle="arc3,rad=0.08",zorder=1))
    for (x,y),label in zip(positions,labels):
        box=FancyBboxPatch((x-0.075,y-0.075),0.15,0.15,boxstyle="round,pad=0.02",
                           linewidth=1.4,edgecolor="#17496c",facecolor="#eaf3f8",zorder=2)
        ax.add_patch(box); ax.text(x,y,label,ha="center",va="center",fontsize=8.2,wrap=True,zorder=3)
    fig.tight_layout(); fig.savefig(FIG/name,dpi=180,bbox_inches="tight"); plt.close(fig)
for name,(labels,edges) in diagrams.items(): draw(name,labels,edges)

def para(text):
    return "<p>"+html.escape(text)+"</p>"
def page_html(i, page):
    typ,title,text=page
    title_html=html.escape(title)
    chunks=[s.strip() for s in text.split(". ") if s.strip()]
    body="".join(para(("".join([]) + sentence + ("" if sentence.endswith(".") else "."))) for sentence in chunks)
    extras=""
    # Reserve concise technical captions and visual evidence on selected pages.
    diagram_pages={24:"Figure_6_1_System_Architecture.png",29:"Figure_6_2_Frontend_Backend.png",
                   30:"Figure_6_3_Image_Workflow.png",31:"Figure_6_4_YOLO_Pipeline.png",
                   32:"Figure_6_5_Two_Model_Workflow.png",33:"Figure_6_6_Flask_Request.png",
                   48:"Figure_9_1_Association.png",49:"Figure_9_2_Assessment_Flow.png",
                   57:"Figure_11_1_Database_Architecture.png",58:"Figure_11_2_RAG_Ingestion.png",
                   59:"Figure_11_3_RAG_QA.png",64:"Figure_12_1_Evaluation_Workflow.png"}
    screenshot_pages={45:("screenshot_dashboard.png","Figure 8.1. Genuine dashboard screenshot captured during local review."),
                      46:("screenshot_analysis.png","Figure 8.2. Genuine stored analysis screenshot; shown as interface evidence, not benchmark output."),
                      54:("screenshot_reports.png","Figure 10.1. Genuine Reports page screenshot."),
                      55:("screenshot_evaluation.png","Figure 10.2. Genuine Evaluation page screenshot."),
                      56:("screenshot_settings.png","Figure 10.3. Genuine Settings screenshot."),
                      60:("screenshot_assistant.png","Figure 11.1. Assistant status captured with Ollama offline and zero indexed documents.")}
    if i in diagram_pages:
        name=diagram_pages[i]
        extras=f'<figure><img src="{FIG/name}" /><figcaption>{name.replace("_"," ").replace(".png","")}. System flow diagram; component boundaries reflect inspected source.</figcaption></figure>'
    elif i in screenshot_pages:
        name,cap=screenshot_pages[i]
        extras=f'<figure><img class="shot" src="{FIG/name}" /><figcaption>{html.escape(cap)}</figcaption></figure>'
    return f'<section class="page"><header>SiteGuardVision | Final-Year Project Report</header><h1>{title_html}</h1><h2>{typ} · Page {i:02d}</h2>{body}{extras}<footer>SiteGuardVision · [Institution] · {i}</footer></section>'

css="""@page { size:A4; margin:18mm 20mm 18mm 22mm; }
body { font-family:'Times New Roman',serif; font-size:12pt; line-height:1.5; color:#171717; }
.page { page-break-after:always; height:257mm; position:relative; overflow:hidden; }
.page:last-child { page-break-after:auto; }
header { border-bottom:1px solid #8796a0; font-size:9pt; color:#36566c; padding-bottom:3mm; margin-bottom:7mm; }
h1 { font-size:16pt; color:#173e59; margin:0 0 3mm; }
h2 { font-size:11pt; color:#496171; margin:0 0 6mm; font-weight:normal; }
p { text-align:justify; margin:0 0 4mm; }
figure { margin:5mm auto 0; text-align:center; }
figure img { width:165mm; height:67mm; object-fit:contain; }
figure img.shot { width:160mm; height:115mm; object-fit:contain; }
figcaption { font-size:9pt; font-style:italic; text-align:center; margin-top:2mm; }
footer { position:absolute; bottom:0; left:0; right:0; border-top:1px solid #8796a0; padding-top:2mm; text-align:right; font-size:9pt; color:#36566c; }
"""
html_doc='<!doctype html><html><head><meta charset="utf-8"><style>'+css+'</style></head><body>'
html_doc+="".join(page_html(i,p) for i,p in enumerate(pages,1))
html_doc+="</body></html>"
(ROOT/"report_source.html").write_text(html_doc,encoding="utf-8")

# Produce a standards-based editable DOCX using only the Python standard library.
# Inline images are stored in /word/media; all prose remains ordinary editable text.
import re, zipfile
from xml.sax.saxutils import escape
from PIL import Image

def wx(text): return escape(str(text), {'"': '&quot;'})
def run(text, bold=False, italic=False, size=24):
    props = f'<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="{size}"/>' + ('<w:b/>' if bold else '') + ('<w:i/>' if italic else '') + '</w:rPr>'
    return f'<w:r>{props}<w:t xml:space="preserve">{wx(text)}</w:t></w:r>'
def paragraph(text='', style=None, align=None, before=0, after=120, line=360, bold=False, size=24):
    ppr=f'<w:pPr><w:spacing w:before="{before}" w:after="{after}" w:line="{line}" w:lineRule="auto"/>'
    if style: ppr+=f'<w:pStyle w:val="{style}"/>'
    if align: ppr+=f'<w:jc w:val="{align}"/>'
    ppr+='</w:pPr>'
    return f'<w:p>{ppr}{run(text,bold=bold,size=size)}</w:p>'
def picture(rel_id, cx, cy):
    return f'''<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="60"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{rel_id}" name="Figure {rel_id}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{rel_id}" name="image{rel_id}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="rIdImg{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''
def page_break(): return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'

diagram_pages={24:"Figure_6_1_System_Architecture.png",29:"Figure_6_2_Frontend_Backend.png",30:"Figure_6_3_Image_Workflow.png",31:"Figure_6_4_YOLO_Pipeline.png",32:"Figure_6_5_Two_Model_Workflow.png",33:"Figure_6_6_Flask_Request.png",48:"Figure_9_1_Association.png",49:"Figure_9_2_Assessment_Flow.png",57:"Figure_11_1_Database_Architecture.png",58:"Figure_11_2_RAG_Ingestion.png",59:"Figure_11_3_RAG_QA.png",64:"Figure_12_1_Evaluation_Workflow.png"}
screenshot_pages={45:"screenshot_dashboard.png",46:"screenshot_analysis.png",54:"screenshot_reports.png",55:"screenshot_evaluation.png",56:"screenshot_settings.png",60:"screenshot_assistant.png"}
media={}
rels=[]
for n,(idx,name) in enumerate([(i,n) for i,n in {**diagram_pages,**screenshot_pages}.items()],1):
    media[idx]=FIG/name
    rels.append(f'<Relationship Id="rIdImg{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{idx}.png"/>')
document_body=[]
for idx,(kind,title,text) in enumerate(pages,1):
    if idx>1: document_body.append(page_break())
    document_body.append(paragraph("SiteGuardVision | Final-Year Project Report",align="right",after=80,line=240,size=18))
    document_body.append(paragraph(title,style="Heading1",before=100,after=120,line=300,size=32,bold=True))
    document_body.append(paragraph(f"{kind} · Page {idx:02d}",after=180,line=260,size=20))
    for sentence in [s.strip() for s in text.split(". ") if s.strip()]:
        sentence=sentence if sentence.endswith(".") else sentence+"."
        document_body.append(paragraph(sentence,align="both",after=120,line=360))
    image=diagram_pages.get(idx) or screenshot_pages.get(idx)
    if image:
        rid=next(k for k,v in media.items() if v.name==image)
        with Image.open(media[rid]) as im:
            max_w=5.8*914400; max_h=(2.35 if idx in diagram_pages else 3.45)*914400
            scale=min(max_w/im.width,max_h/im.height)
            cx=int(im.width*scale); cy=int(im.height*scale)
        document_body.append(picture(rid,cx,cy))
        caption=(f"{image.replace('_',' ').replace('.png','')}. Technical system diagram based on inspected implementation." if idx in diagram_pages else f"Genuine dashboard screenshot captured during local review: {image.replace('screenshot_','').replace('.png','')}.")
        document_body.append(paragraph(caption,align="center",after=80,line=220,size=18))
    document_body.append(paragraph(f"SiteGuardVision · [Institution] · {idx}",align="right",before=120,after=0,line=220,size=18))

sectPr='''<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1020" w:right="1134" w:bottom="1020" w:left="1247" w:header="500" w:footer="500" w:gutter="0"/></w:sectPr>'''
doc_xml='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body>'+''.join(document_body)+sectPr+'</w:body></w:document>'
styles='''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:line="360" w:lineRule="auto" w:after="120"/></w:pPr></w:pPrDefault></w:docDefaults><w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style><w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:sz w:val="32"/><w:color w:val="173E59"/></w:rPr></w:style></w:styles>'''
rels_xml='<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'+''.join(rels)+'</Relationships>'
types='<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
with zipfile.ZipFile(ROOT/"SiteGuardVision_Final_Project_Report.docx","w",zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml",'<?xml version="1.0" encoding="UTF-8"?>'+types)
    z.writestr("_rels/.rels",'<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    z.writestr("word/document.xml",doc_xml); z.writestr("word/styles.xml",styles); z.writestr("word/_rels/document.xml.rels",rels_xml)
    for rid,path in media.items(): z.write(path,f"word/media/image{rid}.png")
print(f"Generated {len(pages)} planned pages and {len(diagrams)} diagrams.")
