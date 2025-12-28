import argparse
import os
import cv2
import numpy as np
from cues import ms_score, ss_score, ed_score

MS_THRESHOLD = 0.2125
MS_Scale = [64]
SS_SCALE = 50
ED_THETA = 0.15

NUM_WINDOWS = 100
TOP_KEEP = 20
TOP_RED = 3
SCORE_THRESHOLD = 0.3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help="Path to test images")
    parser.add_argument('--output', required=True, help="Path to save results")
    return parser.parse_args()

def random_window(w, h):
    x1 = np.random.randint(0, w - 1)
    y1 = np.random.randint(0, h - 1)
    x2 = np.random.randint(x1 + 1, w)
    y2 = np.random.randint(y1 + 1, h)
    return [x1, y1, x2, y2]

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    image_files = [f for f in os.listdir(args.input)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Found {len(image_files)} test images")

    for img_name in image_files:
        img_path = os.path.join(args.input, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        scored_windows = []

        for _ in range(NUM_WINDOWS):
            win = random_window(w, h)
            ms = ms_score(img, win, thresh=MS_THRESHOLD, scale=MS_Scale)
            ss = ss_score(img, win, scale=SS_SCALE)
            ed = ed_score(img, win, theta_ed=ED_THETA)

            score = (ms + ss + ed) / 3.0
            scored_windows.append((score, win))

        filtered_windows = [ (s, w) for s, w in scored_windows if s >= SCORE_THRESHOLD ]
        filtered_windows.sort(key=lambda x: x[0], reverse=True)
        top_windows = filtered_windows[:TOP_KEEP]

        img_all_top = img.copy()
        for score, win in top_windows:
            cv2.rectangle(
                img_all_top,
                (win[0], win[1]),
                (win[2], win[3]),
                (0, 255, 255), 2
            )
        save_all = os.path.join(args.output, f"all_top_{img_name}")
        cv2.imwrite(save_all, img_all_top)

        img_top3 = img.copy()
        for i in range(min(TOP_RED, len(top_windows))):
            score, win = top_windows[i]
            cv2.rectangle(
                img_top3,
                (win[0], win[1]),
                (win[2], win[3]),
                (0, 0, 255), 2
            )
            text = f"{score:.3f}"
            cv2.putText(
                img_top3,
                text,
                (win[0], win[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            print(f"{img_name} | Top {i+1} Score: {score:.4f}")

        save_top3 = os.path.join(args.output, f"top3_{img_name}")
        cv2.imwrite(save_top3, img_top3)

        print(f"Processed {img_name}\n")

    print("Testing complete.")

if __name__ == "__main__":
    main()
