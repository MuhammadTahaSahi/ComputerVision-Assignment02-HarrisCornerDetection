import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np


def make_images(folder, seed, size):
    folder.mkdir(parents=True, exist_ok=True)
    if size != 384:
        raise ValueError("Use 384 for the supplied synthetic ground truth")
    image = np.full((size, size), 0.1, np.float32)
    vertices = []
    for x1, y1, x2, y2, value in [
        (40, 40, 120, 120, 0.9),
        (180, 40, 270, 115, 0.55),
        (40, 190, 120, 275, 0.25),
    ]:
        cv2.rectangle(image, (x1, y1), (x2, y2), value, -1)
        vertices.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    polygon = np.array([(190, 190), (300, 190), (300, 230),
                        (240, 230), (240, 300), (190, 300)], np.int32)
    cv2.fillPoly(image, [polygon], 0.85)
    vertices.extend(polygon.tolist())
    image[:, 340:] = 0.7
    noise = np.random.default_rng(seed).normal(0, 0.02, image.shape)
    images = {"synthetic": image,
              "noisy": np.clip(image + noise, 0, 1).astype(np.float32)}
    for name in ("building", "fruits"):
        original = cv2.imread(str(folder / f"{name}_original.jpg"))
        if original is None:
            raise FileNotFoundError(folder / f"{name}_original.jpg")
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        images[name] = cv2.resize(gray[y0:y0+side, x0:x0+side],
                                  (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    for name, img in images.items():
        np.save(folder / f"{name}.npy", img)
        cv2.imwrite(str(folder / f"{name}.png"), np.rint(img * 255).astype(np.uint8))
    truth = np.asarray(vertices, np.float64).reshape(-1, 2)
    np.savetxt(folder / "ground_truth.csv", truth, delimiter=",", header="x,y", comments="", fmt="%.1f")
    return images, truth


def response_map(image, detector, ksize, block_size, k, epsilon):
    border = cv2.BORDER_REFLECT_101
    if detector == "ST":
        return cv2.cornerMinEigenVal(image, block_size, ksize=ksize, borderType=border)
    if detector == "H":
        return cv2.cornerHarris(image, block_size, ksize, k, borderType=border)
    eig = cv2.cornerEigenValsAndVecs(image, block_size, ksize, borderType=border)
    a, b = eig[..., 0], eig[..., 1]
    trace = a + b
    out = np.zeros(image.shape, np.float32)
    np.divide(a * b, trace, out=out, where=trace > epsilon)
    return out


def select_corners(response, alpha, border, nms):
    maximum = float(response.max())
    threshold = alpha * maximum
    mask = (response > threshold) & (response > 0) if maximum > 0 else np.zeros(response.shape, bool)
    local_max = cv2.dilate(response, np.ones((nms, nms), np.uint8))
    candidates = mask & (response == local_max)
    candidates[:border] = False
    candidates[-border:] = False
    candidates[:, :border] = False
    candidates[:, -border:] = False
    count, labels = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    first = np.full(count, response.size, np.int64)
    flat = np.flatnonzero(candidates)
    np.minimum.at(first, labels.ravel()[flat], flat)
    chosen = first[1:]
    y, x = np.divmod(chosen, response.shape[1])
    points = np.column_stack((x, y)).astype(np.int32).reshape(-1, 2)
    return points, mask, threshold


def detect(image, detector, ksize, block_size, k, alpha, epsilon, border, nms):
    response = response_map(image, detector, ksize, block_size, k, epsilon)
    points, mask, threshold = select_corners(response, alpha, border, nms)
    return response, points, mask, threshold


def evaluate(points, truth, border, size, tolerance):
    valid = np.all((truth >= border) & (truth < size - border), axis=1)
    truth = truth[valid]
    distance = np.linalg.norm(points[:, None, :] - truth[None, :, :], axis=2)
    neighbors = [sorted(np.flatnonzero(row <= tolerance).tolist(), key=lambda j: (row[j], j))
                 for row in distance]
    assigned = {}

    def augment(i, seen):
        for j in neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in assigned or augment(assigned[j], seen):
                assigned[j] = i
                return True
        return False

    for i in range(len(points)):
        augment(i, set())
    tp = len(assigned)
    fp, fn = len(points) - tp, len(truth) - tp
    precision = tp / len(points) if len(points) else 0.0
    recall = tp / len(truth) if len(truth) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    error = float(np.mean([distance[i, j] for j, i in assigned.items()])) if tp else None
    return dict(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1,
                mean_error_px=error, gt_count=len(truth))


def configurations(settings):
    base = settings["baseline"]
    runs = []

    def add(group, image, detector, **changes):
        cfg = dict(base)
        cfg.update(changes)
        cfg.update(group=group, image=image, detector=detector)
        runs.append(cfg)

    for image in ("synthetic", "building", "fruits"):
        for detector in ("ST", "DT", "H"):
            add("baseline", image, detector)
    for key, group in (("ksize", "derivative"), ("block_size", "integration"), ("alpha", "threshold")):
        for value in settings["sweeps"][key]:
            if value == base[key]:
                continue
            for detector in ("ST", "DT", "H"):
                add(group, "synthetic", detector, **{key: value})
    for value in settings["sweeps"]["k"]:
        if value != base["k"]:
            add("harris_k", "synthetic", "H", k=value)
    for detector in ("ST", "DT", "H"):
        add("noise", "noisy", detector)
    return runs


def run(config_path, output):
    settings = json.loads(config_path.read_text())
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    root = config_path.parent
    images, truth = make_images(root / "images", settings["seed"], settings["size"])
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in configurations(settings):
        image = images[cfg["image"]]
        args = (image, cfg["detector"], cfg["ksize"], cfg["block_size"], cfg["k"],
                cfg["alpha"], settings["epsilon"], settings["border"], settings["nms"])
        detect(*args)
        samples = []
        for _ in range(5):
            start = time.perf_counter_ns()
            result = detect(*args)
            samples.append((time.perf_counter_ns() - start) / 1e6)
        response, points, mask, threshold = result
        name = (f"{cfg['image']}_{cfg['detector']}_d{cfg['ksize']}_b{cfg['block_size']}"
                f"_a{cfg['alpha']:.2f}_k{cfg['k']:.2f}")
        folder = output / name
        folder.mkdir(exist_ok=True)
        np.save(folder / "response.npy", response)
        cv2.imwrite(str(folder / "threshold_mask.png"), mask.astype(np.uint8) * 255)
        np.save(folder / "corners.npy", points)
        np.savetxt(folder / "corners.csv", points, delimiter=",", header="x,y", comments="", fmt="%d")
        overlay = cv2.cvtColor(np.rint(image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(folder / "overlay.png"), overlay)
        row = dict(cfg, run_id=name, threshold=threshold, maximum=float(response.max()),
                   count=len(points), runtime_ms=float(np.median(samples)),
                   nms=settings["nms"], border=settings["border"], epsilon=settings["epsilon"])
        if cfg["image"] in ("synthetic", "noisy"):
            row.update(evaluate(points, truth, settings["border"], settings["size"], settings["tolerance"]))
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "results.json").write_text(json.dumps(rows, indent=2))
    metadata = dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                    platform=platform.platform(), processor=platform.processor(), threads=1,
                    timing="One warm-up then median of five runs; response and NMS recomputed; no I/O, plotting or evaluation",
                    seed=settings["seed"], noise_std=0.02,
                    hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (root / "images").glob("*_original.jpg")})
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(dict(runs=len(rows), output=str(output), environment=metadata), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve())
