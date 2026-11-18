"""Detailed, diagnostic-only error and dataset-quality analysis.

The existing run_20261115 baseline is read-only. All raw inference outputs go
to a new run folder, and production code/settings/checkpoints are untouched.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "ppe_evaluation"
RUN = BASE / "error_analysis_20261115"
MODEL = ROOT / "models" / "yolo_ppe_core.pt"
DATA = ROOT / "dataset"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAP = {0: 3, 1: 8, 2: 10, 3: 11, 4: 13}
NAMES = {0:"Helmet",1:"No Helmet",2:"No Vest",3:"Person",4:"Safety Vest"}
PPE = [0,1,2,4]
CONF_FLOOR = 0.001
PROD_CONF = 0.25
IMAGE_SIZE = 960
NMS_IOU = 0.70
MATCH_IOU = 0.50


def read_gt(label_path, width, height):
    objects = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected five YOLO fields")
        cls = int(values[0])
        if cls not in MAP:
            raise ValueError(f"{label_path}:{line_no}: unsupported class {cls}")
        x,y,w,h = map(float, values[1:])
        objects.append({
            "class_id":cls,
            "bbox":[(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height],
            "width_norm":w,"height_norm":h,
        })
    return objects


def iou(a,b):
    x1=max(a[0],b[0]); y1=max(a[1],b[1]); x2=min(a[2],b[2]); y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1])
    bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)


def greedy_same_class(preds, gts, cutoff):
    used_g=set()
    matches={}
    for pi,p in sorted(
        ((i,p) for i,p in enumerate(preds) if p["confidence"]>=cutoff),
        key=lambda item:item[1]["confidence"], reverse=True,
    ):
        candidates=[gi for gi,g in enumerate(gts) if gi not in used_g and g["class_id"]==p["class_id"]]
        if not candidates:
            continue
        overlaps=[iou(p["bbox"],gts[gi]["bbox"]) for gi in candidates]
        j=int(np.argmax(overlaps))
        if overlaps[j]>=MATCH_IOU:
            gi=candidates[j]
            used_g.add(gi)
            matches[pi]=gi
    return matches,used_g


def csv_write(path, rows, fields):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def dataset_quality():
    class_rows=[]; box_rows=[]; report={"splits":{}}
    for split in ["train","valid","test"]:
        image_dir=DATA/split/"images"; label_dir=DATA/split/"labels"
        paths=sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS)
        counts=Counter(); resolutions=Counter(); box_data=defaultdict(list)
        total=0; small_medium_large=Counter()
        for path in paths:
            with Image.open(path) as image:
                width,height=image.size
                image.verify()
            resolutions[f"{width}x{height}"]+=1
            objs=read_gt(label_dir/f"{path.stem}.txt",width,height)
            for obj in objs:
                c=obj["class_id"]; counts[c]+=1; total+=1
                area_norm=obj["width_norm"]*obj["height_norm"]
                area640=area_norm*640*640
                side640=(area640)**0.5
                size="small" if area640<32**2 else "medium" if area640<96**2 else "large"
                small_medium_large[size]+=1
                box_data[c].append({
                    "w_norm":obj["width_norm"],"h_norm":obj["height_norm"],
                    "area_fraction":area_norm,"equiv_area_640":area640,
                    "equiv_side_640":side640,"size":size,
                })
        for c,name in NAMES.items():
            vals=box_data[c]
            class_rows.append({
                "split":split,"class_id":c,"class":name,"instances":counts[c],
                "share_of_split_objects":counts[c]/total if total else 0,
            })
            areas=[x["area_fraction"] for x in vals]
            widths=[x["w_norm"] for x in vals]; heights=[x["h_norm"] for x in vals]
            box_rows.append({
                "split":split,"class_id":c,"class":name,"instances":len(vals),
                "mean_width_fraction":float(np.mean(widths)) if vals else 0,
                "median_width_fraction":float(np.median(widths)) if vals else 0,
                "mean_height_fraction":float(np.mean(heights)) if vals else 0,
                "median_height_fraction":float(np.median(heights)) if vals else 0,
                "mean_area_fraction":float(np.mean(areas)) if vals else 0,
                "median_area_fraction":float(np.median(areas)) if vals else 0,
                "p10_area_fraction":float(np.percentile(areas,10)) if vals else 0,
                "p90_area_fraction":float(np.percentile(areas,90)) if vals else 0,
                "small_lt_32px_side_equiv":sum(x["size"]=="small" for x in vals),
                "medium_32_to_96px_side_equiv":sum(x["size"]=="medium" for x in vals),
                "large_gte_96px_side_equiv":sum(x["size"]=="large" for x in vals),
            })
        report["splits"][split]={
            "images":len(paths),"instances":total,
            "class_counts":{NAMES[c]:counts[c] for c in NAMES},
            "image_resolutions":dict(sorted(resolutions.items())),
            "image_width_min_median_max":[min(int(k.split("x")[0]) for k in resolutions),
                int(np.median([int(k.split("x")[0]) for k,n in resolutions.items() for _ in range(n)])),
                max(int(k.split("x")[0]) for k in resolutions)],
            "image_height_min_median_max":[min(int(k.split("x")[1]) for k in resolutions),
                int(np.median([int(k.split("x")[1]) for k,n in resolutions.items() for _ in range(n)])),
                max(int(k.split("x")[1]) for k in resolutions)],
            "box_size_categories_equiv_640":dict(small_medium_large),
        }
    csv_write(BASE/"class_distribution.csv",class_rows,
        ["split","class_id","class","instances","share_of_split_objects"])
    csv_write(BASE/"bounding_box_analysis.csv",box_rows,list(box_rows[0]))
    (RUN/"dataset_quality_summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def analyze_model():
    model=YOLO(str(MODEL))
    results=[]; image_times=[]
    image_paths=sorted(p for p in (DATA/"test"/"images").iterdir() if p.suffix.lower() in EXTS)
    for index,path in enumerate(image_paths,1):
        with Image.open(path) as im:
            width,height=im.size
        gts=read_gt(DATA/"test"/"labels"/f"{path.stem}.txt",width,height)
        started=time.perf_counter()
        output=model.predict(source=str(path),conf=CONF_FLOOR,iou=NMS_IOU,
            imgsz=IMAGE_SIZE,device="cpu",verbose=False,max_det=300,save=False)[0]
        image_times.append((time.perf_counter()-started)*1000)
        preds=[]
        for box,conf,model_cls in zip(output.boxes.xyxy.tolist(),output.boxes.conf.tolist(),output.boxes.cls.tolist()):
            model_cls=int(model_cls)
            ds_cls=next((d for d,m in MAP.items() if m==model_cls),None)
            if ds_cls is None: continue
            preds.append({"class_id":ds_cls,"confidence":float(conf),
                "bbox":[float(v) for v in box],"model_class_id":model_cls})
        results.append({"image":path.name,"width":width,"height":height,"preds":preds,"gts":gts})
        print(f"[{index}/{len(image_paths)}] {path.name}",flush=True)
    return results,image_times


def main():
    RUN.mkdir(parents=True,exist_ok=False)
    quality=dataset_quality()
    results,image_times=analyze_model()
    gt_counts=Counter(g["class_id"] for image in results for g in image["gts"])
    class_counts=Counter()
    pred_rows=[]; fn_rows=[]; summary={}
    for image in results:
        preds=image["preds"]; gts=image["gts"]
        candidate_matches,_=greedy_same_class(preds,gts,CONF_FLOOR)
        accepted_matches,accepted_gts=greedy_same_class(preds,gts,PROD_CONF)
        wrong_class_pairs=[]
        unmatched_gts=[i for i in range(len(gts)) if i not in accepted_gts]
        unmatched_preds=[i for i,p in enumerate(preds) if i not in accepted_matches and p["confidence"]>=PROD_CONF]
        # Match remaining objects across classes to identify likely class errors.
        cross=[]
        for pi in unmatched_preds:
            for gi in unmatched_gts:
                overlap=iou(preds[pi]["bbox"],gts[gi]["bbox"])
                if overlap>=MATCH_IOU and preds[pi]["class_id"]!=gts[gi]["class_id"]:
                    cross.append((overlap,preds[pi]["confidence"],pi,gi))
        used_cross_p=set();used_cross_g=set()
        for overlap,_,pi,gi in sorted(cross,reverse=True):
            if pi in used_cross_p or gi in used_cross_g: continue
            used_cross_p.add(pi);used_cross_g.add(gi)
            wrong_class_pairs.append((pi,gi,overlap))
        cross_by_pred={pi:(gi,ov) for pi,gi,ov in wrong_class_pairs}
        cross_by_gt={gi:(pi,ov) for pi,gi,ov in wrong_class_pairs}

        for pi,p in enumerate(preds):
            cls=p["class_id"]; class_counts[cls]+=1
            same_gts=[(gi,g) for gi,g in enumerate(gts) if g["class_id"]==cls]
            best_same=max(same_gts,key=lambda pair:iou(p["bbox"],pair[1]["bbox"]),default=None)
            best_iou=iou(p["bbox"],best_same[1]["bbox"]) if best_same else 0.0
            is_prod=p["confidence"]>=PROD_CONF
            if pi in accepted_matches:
                category="correct_detection"
                gt_idx=accepted_matches[pi]
            elif not is_prod:
                category="below_threshold_correct_candidate" if best_iou>=MATCH_IOU else "below_threshold_unmatched"
                gt_idx=best_same[0] if best_same and best_iou>=MATCH_IOU else None
            elif pi in cross_by_pred:
                gt_idx,overlap=cross_by_pred[pi]; category="incorrect_class_prediction"; best_iou=overlap
            elif best_same and best_iou>=MATCH_IOU:
                category="duplicate_or_assignment_collision";gt_idx=best_same[0]
            elif best_same and best_iou>0:
                category="poor_iou_localization";gt_idx=best_same[0]
            else:
                category="background_false_positive";gt_idx=None
            pred_rows.append({
                "image":image["image"],"class_id":cls,"class":NAMES[cls],
                "confidence":p["confidence"],"bbox_xyxy":json.dumps(p["bbox"]),
                "production_accepted":is_prod,"match_category":category,
                "ground_truth_class":NAMES[gts[gt_idx]["class_id"]] if gt_idx is not None else "",
                "ground_truth_bbox_xyxy":json.dumps(gts[gt_idx]["bbox"]) if gt_idx is not None else "",
                "iou":best_iou if gt_idx is not None else "",
            })

        for gi,g in enumerate(gts):
            if gi in accepted_gts: continue
            cls=g["class_id"]
            same=[(pi,p) for pi,p in enumerate(preds) if p["class_id"]==cls]
            best=max(same,key=lambda pair:iou(pair[1]["bbox"],g["bbox"]),default=None)
            best_conf=best[1]["confidence"] if best else None
            best_iou=iou(best[1]["bbox"],g["bbox"]) if best else 0.0
            cross_pred=cross_by_gt.get(gi)
            if cross_pred:
                pi,overlap=cross_pred; category="incorrect_class_prediction"
                cand=preds[pi]; cand_iou=overlap
            elif best and best_conf<PROD_CONF and best_iou>=MATCH_IOU:
                category="rejected_below_0.25";cand=best[1];cand_iou=best_iou
            elif best and best_iou>0 and best_iou<MATCH_IOU:
                category="poor_iou_localization";cand=best[1];cand_iou=best_iou
            elif best and best_iou>=MATCH_IOU:
                category="duplicate_or_assignment_collision";cand=best[1];cand_iou=best_iou
            elif best:
                category=(
                    "accepted_same_class_candidate_no_overlap"
                    if best[1]["confidence"]>=PROD_CONF else
                    "below_threshold_same_class_candidate_no_overlap"
                )
                cand=best[1];cand_iou=best_iou
            else:
                category="no_same_class_candidate";cand=None;cand_iou=0.0
            fn_rows.append({
                "image":image["image"],"class_id":cls,"class":NAMES[cls],
                "ground_truth_bbox_xyxy":json.dumps(g["bbox"]),
                "candidate_class":NAMES[cand["class_id"]] if cand else "",
                "candidate_confidence":cand["confidence"] if cand else "",
                "candidate_bbox_xyxy":json.dumps(cand["bbox"]) if cand else "",
                "candidate_iou":cand_iou,"production_threshold":PROD_CONF,
                "failure_category":category,
                "note":"PPE checkpoint model-level error; no production assessment was changed.",
            })

    csv_write(RUN/"all_predictions.csv",pred_rows,
        ["image","class_id","class","confidence","bbox_xyxy","production_accepted","match_category","ground_truth_class","ground_truth_bbox_xyxy","iou"])
    csv_write(BASE/"false_negative_analysis.csv",fn_rows,
        ["image","class_id","class","ground_truth_bbox_xyxy","candidate_class","candidate_confidence","candidate_bbox_xyxy","candidate_iou","production_threshold","failure_category","note"])

    conf_rows=[]
    for cls,name in NAMES.items():
        rows=[r for r in pred_rows if r["class_id"]==cls]
        accepted=[r for r in rows if r["production_accepted"]]
        correct=[r["confidence"] for r in accepted if r["match_category"]=="correct_detection"]
        incorrect=[r["confidence"] for r in accepted if r["match_category"]!="correct_detection"]
        below=[r for r in rows if not r["production_accepted"]]
        fns=[r for r in fn_rows if r["class_id"]==cls]
        summary[name]={
            "ground_truth":gt_counts[cls],"predictions_at_floor":len(rows),
            "accepted_at_025":len(accepted),"correct_accepted":len(correct),
            "incorrect_accepted":len(incorrect),"false_negatives_at_025":len(fns),
            "precision_at_025":len(correct)/len(accepted) if accepted else 0,
            "recall_at_025":len(correct)/gt_counts[cls] if gt_counts[cls] else 0,
            "below_threshold_predictions":len(below),
            "below_threshold_candidates_iou50":sum(r["match_category"]=="below_threshold_correct_candidate" for r in below),
            "correct_confidence_mean":float(np.mean(correct)) if correct else None,
            "correct_confidence_median":float(np.median(correct)) if correct else None,
            "incorrect_confidence_mean":float(np.mean(incorrect)) if incorrect else None,
            "incorrect_confidence_median":float(np.median(incorrect)) if incorrect else None,
            "below_threshold_confidence_mean":float(np.mean([r["confidence"] for r in below])) if below else None,
            "fn_categories":dict(Counter(r["failure_category"] for r in fns)),
        }
        conf_rows.append({"class_id":cls,"class":name,**summary[name]})
    csv_write(BASE/"confidence_analysis.csv",conf_rows,list(conf_rows[0]))

    # Save one low-confidence distribution and a TP-vs-FP score comparison.
    plt.figure(figsize=(10,6))
    bins=np.linspace(0,1,21)
    for cls in PPE:
        vals=[r["confidence"] for r in pred_rows if r["class_id"]==cls]
        if vals: plt.hist(vals,bins=bins,alpha=.35,label=NAMES[cls])
    plt.axvline(PROD_CONF,color="black",linestyle="--",label="Production cutoff 0.25")
    plt.xlabel("Confidence");plt.ylabel("Prediction count")
    plt.title("Test-set PPE confidence distribution (diagnostic floor 0.001)")
    plt.legend();plt.tight_layout();plt.savefig(BASE/"confidence_distribution_detailed.png",dpi=160);plt.close()
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,cls in zip(axes.flat,PPE):
        good=[r["confidence"] for r in pred_rows if r["class_id"]==cls and r["production_accepted"] and r["match_category"]=="correct_detection"]
        bad=[r["confidence"] for r in pred_rows if r["class_id"]==cls and r["production_accepted"] and r["match_category"]!="correct_detection"]
        ax.hist(good,bins=bins,alpha=.6,label="Correct accepted")
        ax.hist(bad,bins=bins,alpha=.6,label="Incorrect accepted")
        ax.axvline(PROD_CONF,color="black",linestyle="--")
        ax.set_title(NAMES[cls]);ax.set_xlabel("Confidence");ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(BASE/"confidence_correct_vs_incorrect.png",dpi=160);plt.close(fig)

    report={
        "source_baseline":"reports/ppe_evaluation/run_20261115",
        "model_sha256":hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "images_evaluated":len(results),"threshold":PROD_CONF,
        "diagnostic_candidate_floor":CONF_FLOOR,"imgsz":IMAGE_SIZE,
        "nms_iou":NMS_IOU,"match_iou":MATCH_IOU,
        "mean_wall_ms_per_image":float(np.mean(image_times)),
        "classes":summary,
        "fn_count_by_class":dict(Counter(r["class"] for r in fn_rows)),
        "fn_category_counts":dict(Counter(r["failure_category"] for r in fn_rows)),
        "false_negative_count_ppe_classes":sum(r["class_id"] in PPE for r in fn_rows),
        "false_negative_count_ppe_checkpoint_person":sum(r["class_id"]==3 for r in fn_rows),
        "confidence_limits":"Boxes below 0.001 are not emitted by Ultralytics and therefore absent from candidate records.",
        "interpretation_limit":"Image labels are annotation references, not production detections or ground truth of site compliance.",
        "dataset_quality":quality,
    }
    (RUN/"error_analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"images":len(results),"fn_by_class":report["fn_count_by_class"],
        "fn_categories":report["fn_category_counts"],"confidence":summary,
        "output_folder":str(RUN)},indent=2),flush=True)

if __name__=="__main__":
    main()
