"""
exp000 ~ exp003 공통 코드

네 실험은 아래 두 가지만 다르고 나머지(데이터 분할, 입력 크기, 증강, 시드, 학습 설정, 검증 방법)는
모두 같아야 비교가 공정하다. 그래서 공통 부분을 이 파일 하나에 모으고, 각 exp 파일은
run_experiment(...)에 두 설정만 넘긴다.
  - use_repeat : 희소 클래스 이미지 반복 학습 여부
  - cls_pw     : 클래스 가중치 세기 (0.0이면 사용 안 함)
"""
import csv
import math
import os
import time
from collections import Counter

import cv2
import matplotlib
import numpy as np
import torch
import ultralytics
import yaml

matplotlib.use("Agg")  # .py 실행 시 창을 띄우지 않고 그림을 파일로만 저장
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT  # 설치된 버전의 기본 학습 설정값

# ============================================================
# 경로: 이 파일(experiments/common.py)의 위치에서 프로젝트 폴더를 자동으로 찾는다.
# ============================================================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "construction-ppe-ultralytics-11")

# ============================================================
# 학습 설정 (모든 실험 동일)
# ============================================================
EPOCHS = 50  # 50 (프로젝트 일정과 16회 실험 시간 고려)
IMGSZ = 640  # 보드(NPU) 입력 크기와도 동일
BATCH = 16   
SEED = 0     # 모든 실험 동일

# Detect 모듈 위치: yolov8n.yaml은 22번, yolo11n.yaml은 23번 층
# freeze=N → 0 ~ N-1번 층(백본+넥)을 고정하고 Detect만 학습 / None → 전체 미세조정
MODEL_CONDITIONS = {
    "v8n_head":  {"weights": "yolov8n.pt", "detect_index": 22, "freeze": 22},
    "v8n_full":  {"weights": "yolov8n.pt", "detect_index": 22, "freeze": None},
    "v11n_head": {"weights": "yolo11n.pt", "detect_index": 23, "freeze": 23},
    "v11n_full": {"weights": "yolo11n.pt", "detect_index": 23, "freeze": None},
}

# ============================================================
# 데이터 증강: 
#   fliplr 0.5          ↔ RandomHorizontalFlip (좌우 구분 클래스가 없어 안전)
#   flipud 0.0          ↔ RandomVerticalFlip   (뒤집힌 작업자는 현실에 없음 → 사용 안 함)
#   degrees 0.0         ↔ RandomRotation       (회전하면 축 정렬 박스가 커져 정답이 부정확 → 사용 안 함)
#   hsv_h/s/v           ↔ 색상·밝기 변화       (현장 조명 차이)
#   scale, translate    ↔ 크기 조정            (멀리 있거나 가장자리에 있는 작업자)
#   mosaic 1.0          : 4장을 합쳐 작은 객체 학습, 마지막 10 epoch(close_mosaic)는 끔
# 전처리(레터박스, BGR→RGB, HWC→CHW, /255)는 Ultralytics 내부에서 하므로 따로 하지 않는다.
# ============================================================
AUG_KEYS = ["hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
            "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix",
            "copy_paste", "close_mosaic"]

# ============================================================
# 희소 클래스 이미지 반복 기준 (학습 데이터 통계만 사용)
#   1) 희소 클래스: 학습 박스 수가 클래스 평균 박스 수의 RARE_RATIO(50%) 미만
#   2) 클래스 반복 횟수: round(sqrt(평균 박스 수 / 그 클래스 박스 수))
#      - 제곱근: 반복 횟수를 완만하게 만들어 같은 이미지를 너무 많이 보는 과적합을 줄임
#   3) 이미지 반복 횟수: 그 이미지에 있는 클래스들의 반복 횟수 중 최댓값
# ============================================================
RARE_RATIO = 0.5

# ============================================================
# 데이터 확인
# ============================================================
def read_split_classes(split):
    """split(train/val)의 라벨을 읽어 {라벨파일명: [클래스 번호, ...]} 로 반환한다."""
    label_dir = os.path.join(DATA_DIR, "labels", split)
    per_image = {}
    for file in sorted(os.listdir(label_dir)):
        with open(os.path.join(label_dir, file), "r") as f:
            per_image[file] = [int(line.split()[0]) for line in f if line.strip()]
    return per_image


def count_classes(per_image, image_repeat=None):
    """클래스별 박스 수와 이미지 수를 센다. image_repeat가 있으면 반복 횟수만큼 곱해서 센다."""
    box_count = Counter()  # 클래스별 박스 개수
    img_count = Counter()  # 그 클래스가 1개 이상 들어있는 이미지 수
    for file, classes in per_image.items():
        k = image_repeat[file] if image_repeat else 1
        for c in classes:
            box_count[c] += k
        for c in set(classes):  # set: 한 이미지에서 같은 클래스는 1번만
            img_count[c] += k
    return box_count, img_count


def print_class_table(title, class_names, box_count, img_count):
    total = sum(box_count.values())
    print(f"\n{title} 박스 {total}개")
    print(f"{'번호':>4} {'클래스':<10} {'박스':>6} {'비율%':>6} {'이미지':>6}")
    for c in range(len(class_names)):
        print(f"{c:>4} {class_names[c]:<10} {box_count[c]:>6} "
              f"{box_count[c] / total * 100:>6.1f} {img_count[c]:>6}")


# ============================================================
# 희소 클래스 이미지 반복 목록
# ============================================================
def make_repeat_list(per_image, class_names, out_txt, out_csv):
    """희소 클래스가 있는 학습 이미지를 반복한 이미지 목록(txt)을 만든다.
    Ultralytics는 data.yaml의 train에 이미지 경로 목록 txt를 받을 수 있고,
    같은 경로가 여러 줄 있으면 그만큼 반복해서 사용한다. (이미지를 복사하지 않음)"""
    num_classes = len(class_names)
    box_count, img_count = count_classes(per_image)
    mean_count = sum(box_count.values()) / num_classes

    # 1) 희소 클래스와 2) 클래스별 반복 횟수
    class_repeat = {}
    for c in range(num_classes):
        if box_count[c] < RARE_RATIO * mean_count:
            class_repeat[c] = max(1, round(math.sqrt(mean_count / box_count[c])))

    print(f"\n[희소 클래스 기준] 클래스 평균 박스 수 {mean_count:.1f}개, "
          f"그 {RARE_RATIO * 100:.0f}%({RARE_RATIO * mean_count:.1f}개) 미만")
    for c, r in class_repeat.items():
        print(f"  {class_names[c]:<10} 박스 {box_count[c]:>4}개 → "
              f"sqrt({mean_count:.1f}/{box_count[c]}) = {math.sqrt(mean_count / box_count[c]):.2f} → 반복 {r}회")

    # 3) 이미지별 반복 횟수 = 그 이미지 클래스들의 반복 횟수 중 최댓값
    image_repeat = {}
    lines = []
    for file, classes in per_image.items():
        k = max([class_repeat.get(c, 1) for c in set(classes)] + [1])
        image_repeat[file] = k
        img_path = os.path.join(DATA_DIR, "images", "train", file.replace(".txt", ".jpg"))
        lines += [img_path] * k

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 반복 후 전체 클래스 분포 다시 집계 (한 이미지에 여러 클래스가 있어 흔한 클래스도 같이 늘어남)
    new_box, new_img = count_classes(per_image, image_repeat)
    n_before, n_after = len(per_image), len(lines)
    print(f"\n[반복 결과] 학습 이미지 {n_before}장 → {n_after}장 (x{n_after / n_before:.2f})")
    print(f"  최다/최소 클래스 박스 비율: {max(box_count.values()) / min(box_count.values()):.2f}"
          f" → {max(new_box.values()) / min(new_box.values()):.2f}")
    print_class_table("[반복 후 train]", class_names, new_box, new_img)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class", "repeat", "boxes_before", "boxes_after",
                         "share_before_%", "share_after_%"])
        tb, ta = sum(box_count.values()), sum(new_box.values())
        for c in range(num_classes):
            writer.writerow([c, class_names[c], class_repeat.get(c, 1), box_count[c], new_box[c],
                             round(box_count[c] / tb * 100, 2), round(new_box[c] / ta * 100, 2)])
    return n_after


# ============================================================
# data.yaml 
# ============================================================
def create_data_yaml(path_to_data_yaml, class_names, train_path):
    """train_path만 실험마다 달라지고 val은 모든 실험에서 원본 검증 폴더를 그대로 쓴다."""
    data = {
        "path": DATA_DIR,
        "train": train_path,
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    with open(path_to_data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True)
    print(f"\nCreated config file at {path_to_data_yaml}")
    with open(path_to_data_yaml, "r", encoding="utf-8") as f:
        print(f.read())


# ============================================================
# 증강 확인: 좌우 반전 시 정답 박스도 함께 바뀌는지 
# ============================================================
def read_yolo_labels(label_path):
    """YOLO 라벨 파일을 [class, x_center, y_center, w, h] 리스트로 읽는다."""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            if line.strip():
                c, x, y, w, h = line.split()
                boxes.append([int(c), float(x), float(y), float(w), float(h)])
    return boxes


def hflip_image_and_boxes(image, boxes):
    """ RandomHorizontalFlip을 탐지용으로 확장: 이미지와 정답 박스를 함께 좌우 반전한다."""
    flipped = cv2.flip(image, 1)  # 1: 좌우 반전
    # 정규화 좌표이므로 x_center만 1 - x 로 바뀌고 y, w, h는 그대로
    flipped_boxes = [[c, 1.0 - x, y, w, h] for c, x, y, w, h in boxes]
    return flipped, flipped_boxes


def draw_boxes(image, boxes, class_names):
    """ 
    미착용(no_) 클래스는 빨간색, 나머지는 초록색."""
    img = image.copy()
    H, W = img.shape[:2]
    for c, x, y, w, h in boxes:
        # 정규화 중심좌표 → 픽셀 모서리 좌표 (xmin, ymin, xmax, ymax)
        x1, y1 = int((x - w / 2) * W), int((y - h / 2) * H)
        x2, y2 = int((x + w / 2) * W), int((y + h / 2) * H)
        color = (0, 0, 255) if class_names[c].startswith("no_") else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, class_names[c], (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def save_aug_check(per_image, class_names, run_dir):
    """증강 설정값을 기록하고, 좌우 반전 전후 이미지+박스를 그림으로 저장한다."""
    aug_cfg = {k: DEFAULT_CFG_DICT[k] for k in AUG_KEYS}
    print("\n[증강 설정: Ultralytics 기본값 그대로 사용]")
    for k, v in aug_cfg.items():
        print(f"  {k:<13}: {v}")
    with open(os.path.join(run_dir, "aug_settings.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(aug_cfg, f, sort_keys=False)

    # 희소 클래스(no_goggle=7)가 있고 박스가 15개 이하인 첫 학습 이미지
    sample = next(f for f, cl in per_image.items() if 7 in cl and len(cl) <= 15)
    boxes = read_yolo_labels(os.path.join(DATA_DIR, "labels", "train", sample))
    img_path = os.path.join(DATA_DIR, "images", "train", sample.replace(".txt", ".jpg"))
    # cv2.imread는 한글 경로를 못 읽으므로 np.fromfile + cv2.imdecode 사용
    image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    flipped, flipped_boxes = hflip_image_and_boxes(image, boxes)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(draw_boxes(image, boxes, class_names), cv2.COLOR_BGR2RGB))
    plt.title("Original + GT boxes")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(draw_boxes(flipped, flipped_boxes, class_names), cv2.COLOR_BGR2RGB))
    plt.title("Horizontal flip + transformed boxes")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "aug_check_fliplr.png"), dpi=120)
    plt.close()


# ============================================================
# 학습 
# ============================================================
def train_one(cond_name, cond, data_yaml, run_dir, cls_pw):
    """조건 하나를 원래 사전학습 가중치에서 새로 시작해 학습한다."""
    model = YOLO(cond["weights"])  # 조건마다 새로 불러오므로 서로 영향을 주지 않음
    start = time.time()
    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        optimizer="auto",
        freeze=cond["freeze"],
        cls_pw=cls_pw,
        project=run_dir,
        name=cond_name,
        exist_ok=True,
    )
    elapsed = time.time() - start

    # optimizer=auto가 실제로 고른 값은 학습 후 trainer.args에 남는다 (args.yaml에는 auto로 저장됨)
    trainer = model.trainer
    train_images = len(trainer.train_loader.dataset)
    return {
        "condition": cond_name,
        "weights": cond["weights"],
        "freeze": cond["freeze"],
        "cls_pw": cls_pw,
        "epochs": trainer.epochs,
        "train_images": train_images,
        "image_exposures": train_images * trainer.epochs,  # 학습 이미지 노출 수
        "optimizer": trainer.args.optimizer,
        "lr0": trainer.args.lr0,
        "train_time_min": round(elapsed / 60, 2),
    }


def check_freeze(cond, best_pt):
    """백본·넥(0 ~ Detect 직전 층)의 가중치와 BatchNorm 통계가 학습 전후로 그대로인지 확인한다.
    head 조건은 바뀌지 않아야 하고, full 조건은 바뀌어야 정상이다."""
    before = YOLO(cond["weights"]).model.state_dict()
    after = YOLO(best_pt).model.state_dict()
    max_w, max_bn = 0.0, 0.0
    for k, v in before.items():
        layer = int(k.split(".")[1])  # 'model.3.conv.weight' → 3
        if layer >= cond["detect_index"] or not v.dtype.is_floating_point:
            continue
        diff = (after[k].float() - v.float()).abs().max().item()
        if "running_mean" in k or "running_var" in k:
            max_bn = max(max_bn, diff)  # BatchNorm 통계
        else:
            max_w = max(max_w, diff)    # 학습 가중치
    # 고정된 층은 변화량이 0이어야 한다 (계산 오차만 허용).
    # 1 epoch만 학습해도 full 조건의 가중치 변화가 0.001 정도라 허용치를 크게 잡으면 구분이 안 된다.
    frozen = max_w < 1e-6 and max_bn < 1e-6
    print(f"[고정 확인] 백본·넥 최대 변화량: 가중치 {max_w:.6f}, BN 통계 {max_bn:.6f} → "
          f"{'고정됨' if frozen else '학습됨'}")
    return {"backbone_max_diff": round(max_w, 6), "bn_stat_max_diff": round(max_bn, 6),
            "backbone_frozen": frozen}


def best_epoch_info(train_dir):
    """results.csv에서 mAP50-95가 가장 높은 epoch과 마지막 epoch 값을 읽는다 (과적합 확인용)."""
    with open(os.path.join(train_dir, "results.csv"), "r", encoding="utf-8") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[key]))
    return {"best_epoch": int(float(best["epoch"])),
            "best_mAP50-95_in_train": round(float(best[key]), 4),
            "last_mAP50-95_in_train": round(float(rows[-1][key]), 4)}


# ============================================================
# 검증 
# ============================================================
def validate_one(cond_name, best_pt, data_yaml, run_dir, class_names):
    """best.pt를 검증셋으로 평가해 전체 mAP와 클래스별 P, R, F1, AP를 구한다."""
    model = YOLO(best_pt)
    metrics = model.val(data=data_yaml, split="val", imgsz=IMGSZ, batch=BATCH,
                        project=run_dir, name=f"{cond_name}_val", exist_ok=True)
    box = metrics.box
    rows = []
    for i, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(i)
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0  
        rows.append({"condition": cond_name, "class_id": int(c), "class": class_names[int(c)],
                     "precision": round(float(p), 4), "recall": round(float(r), 4),
                     "f1": round(float(f1), 4), "AP50": round(float(ap50), 4),
                     "AP50-95": round(float(ap), 4)})
    return round(float(box.map50), 4), round(float(box.map), 4), rows


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {path}")


# ============================================================
# 실험 전체 실행
# ============================================================
def run_experiment(exp_name, use_repeat, cls_pw):
    run_dir = os.path.join(ROOT, "runs", exp_name)  # runs/expXXX/v8n_head 등이 여기에 생김
    os.makedirs(run_dir, exist_ok=True)

    # 1. 장치 확인 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
    print(f"[{exp_name}] 반복 학습={use_repeat}, cls_pw={cls_pw}")
    print(f"현재 사용 중인 디바이스: {device} ({gpu_name})")
    print(f"PyTorch {torch.__version__} / Ultralytics {ultralytics.__version__}")
    print(f"결과 저장 위치: {run_dir}")

    # 2. 데이터 확인 
    with open(os.path.join(DATA_DIR, "data.yaml"), "r", encoding="utf-8") as f:
        class_names = yaml.safe_load(f)["names"]  # {0: 'helmet', 1: 'gloves', ...}
    train_classes = read_split_classes("train")
    val_classes = read_split_classes("val")
    print_class_table(f"[train] 이미지 {len(train_classes)}장,", class_names, *count_classes(train_classes))
    print_class_table(f"[val] 이미지 {len(val_classes)}장,", class_names, *count_classes(val_classes))

    # 3. 학습 이미지 목록: 반복 학습이면 반복 목록 txt, 아니면 원본 학습 폴더
    if use_repeat:
        train_txt = os.path.join(run_dir, f"train_{exp_name}.txt")
        make_repeat_list(train_classes, class_names, train_txt,
                         os.path.join(run_dir, "repeat_stats.csv"))
        train_path = train_txt
    else:
        train_path = "images/train"

    # 4. data.yaml 생성 
    data_yaml = os.path.join(run_dir, f"data_{exp_name}.yaml")
    create_data_yaml(data_yaml, class_names, train_path)

    # 5. 증강 설정 기록 + 좌우 반전 확인 그림
    save_aug_check(train_classes, class_names, run_dir)

    # 실험 설정 기록
    config = {"exp_name": exp_name, "use_repeat": use_repeat, "cls_pw": cls_pw,
              "rare_ratio": RARE_RATIO if use_repeat else None,
              "epochs": EPOCHS, "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
              "torch": torch.__version__, "ultralytics": ultralytics.__version__, "gpu": gpu_name}
    with open(os.path.join(run_dir, "experiment_config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

    # 6. 네 조건 학습 → 고정 확인 → 검증
    summary, class_rows = [], []
    for cond_name, cond in MODEL_CONDITIONS.items():
        print(f"\n==================== {exp_name} / {cond_name} ====================")
        result = train_one(cond_name, cond, data_yaml, run_dir, cls_pw)
        train_dir = os.path.join(run_dir, cond_name)
        best_pt = os.path.join(train_dir, "weights", "best.pt")

        result.update(check_freeze(cond, best_pt))
        result.update(best_epoch_info(train_dir))
        map50, map5095, rows = validate_one(cond_name, best_pt, data_yaml, run_dir, class_names)
        result.update({"val_mAP50": map50, "val_mAP50-95": map5095})
        summary.append(result)
        class_rows += rows

        # 조건 하나가 끝날 때마다 저장 (중간에 멈춰도 끝난 결과는 남도록)
        save_csv(os.path.join(run_dir, "summary.csv"), summary)
        save_csv(os.path.join(run_dir, "class_metrics.csv"), class_rows)

    print(f"\n[{exp_name}] 완료")
    for r in summary:
        print(f"  {r['condition']:<10} mAP50 {r['val_mAP50']:.4f}  mAP50-95 {r['val_mAP50-95']:.4f}  "
              f"best epoch {r['best_epoch']:>2}  {r['train_time_min']}분  "
              f"{r['optimizer']}(lr={r['lr0']})  backbone_frozen={r['backbone_frozen']}")
