import cv2
import numpy as np

# SIFT
def detect_differences(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        height, width = img1.shape[:2]
        aligned_img2 = cv2.warpPerspective(img2, M, (width, height))

        diff = cv2.absdiff(img1, aligned_img2)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        mask_rgb = cv2.cvtColor(mask_diff, cv2.COLOR_GRAY2BGR)
        changed_area = cv2.bitwise_and(aligned_img2, mask_rgb)

        return aligned_img2, changed_area
    else:
        return None, None


def pixel_pairwise(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = cv2.absdiff(img1, img2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    changed_area = cv2.bitwise_and(img2, mask_rgb)

    return changed_area

def align_with_phase_correlation(ref_path, target_path, out_path=None, display=False):
    """
    Align target_path to ref_path using phase correlation.

    Args:
        ref_path (str): Path to the reference image.
        target_path (str): Path to the target image to be aligned.
        out_path (str, optional): If provided, saves the aligned image here.
        display (bool): If True, shows before/after windows.
    Returns:
        aligned (ndarray): The aligned target image.
        shift (tuple): (dx, dy) translation applied to the target.
    """

    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Проверяем размеры и обрезаем до минимального размера
    h = min(ref.shape[0], target.shape[0])
    w = min(ref.shape[1], target.shape[1])
    ref = ref[:h, :w]
    target = target[:h, :w]

    (dx, dy), _ = cv2.phaseCorrelate(ref, target)
    print(f"Shift: dx = {dx:.3f}, dy = {dy:.3f}")

    # Perform the alignment
    M = np.array([[1, 0, dx],
                  [0, 1, dy]], dtype=np.float32)
    aligned = cv2.warpAffine(target, M, (w, h),
                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # save/display
    if out_path:
        cv2.imwrite(out_path, aligned)
        print(f"Aligned image saved to: {out_path}")

    if display:
        cv2.imshow("Reference", ref.astype(np.uint8))
        cv2.imshow("Original Target", target.astype(np.uint8))
        cv2.imshow("Aligned Target", aligned.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return aligned, (dx, dy)