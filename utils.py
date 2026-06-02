import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

def pose_to_xy_yaw(pose):
    """
    pose: Tensor med shape (4,) = [tx, ty, sin(yaw), cos(yaw)]
    """
    pose = pose.detach().cpu()
    tx, ty, s, c = pose.tolist()
    yaw = np.arctan2(s, c)
    return tx, ty, yaw

def plot_relative_pose(gt_pose, pred_pose):
    gt_tx, gt_ty, gt_yaw = pose_to_xy_yaw(gt_pose)
    pr_tx, pr_ty, pr_yaw = pose_to_xy_yaw(pred_pose)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Camera A som origin
    ax.scatter(0.0, 0.0, c="black", s=60, label="Camera A")
    ax.annotate("A", (0.0, 0.0), xytext=(6, 6), textcoords="offset points")

    arrow_len = 0.5

    # GT
    ax.scatter(gt_tx, gt_ty, c="green", s=70, label="GT B")
    ax.quiver(
        gt_tx, gt_ty,
        arrow_len * np.cos(gt_yaw),
        arrow_len * np.sin(gt_yaw),
        angles="xy", scale_units="xy", scale=1,
        color="green", width=0.006
    )
    ax.annotate("GT B", (gt_tx, gt_ty), xytext=(6, 6), textcoords="offset points", color="green")

    # Prediction
    ax.scatter(pr_tx, pr_ty, c="red", s=70, label="Pred B")

    ax.quiver(
        pr_tx, pr_ty,
        arrow_len * np.cos(pr_yaw), 
        arrow_len * np.sin(pr_yaw),
        angles="xy", scale_units="xy", scale=1,
        color="red", width=0.006
    )
    ax.annotate("Pred B", (pr_tx, pr_ty), xytext=(6, 6), textcoords="offset points", color="red")

    xs = [0.0, gt_tx, pr_tx]
    ys = [0.0, gt_ty, pr_ty]
    pad = 0.5
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Relative pose: GT vs prediction")
    ax.legend()
    plt.show()


def pose_to_text(pose_4):
    """
    pose_4: tensor [tx, ty, sin(yaw), cos(yaw)]
    """
    tx, ty, s, c = pose_4.tolist()
    yaw_deg = np.degrees(np.arctan2(s, c))
    return f"tx={tx:.3f}, ty={ty:.3f}, yaw={yaw_deg:.1f}°"

def show_image_pair(img_a, img_b, title=None):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img_a.permute(1, 2, 0).cpu().clamp(0, 1))
    axes[0].set_title("Image A")
    axes[0].axis("off")

    axes[1].imshow(img_b.permute(1, 2, 0).cpu().clamp(0, 1))
    axes[1].set_title("Image B")
    axes[1].axis("off")

    if title is not None:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def _as_numpy_image(img):
    if hasattr(img, "detach"):
        img = img.detach().cpu()

    if hasattr(img, "permute") and img.dim() == 3 and img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)

    img = np.asarray(img)

    if img.ndim == 2:
        return np.clip(img, 0.0, 1.0)

    if img.shape[-1] == 1:
        img = img[..., 0]

    return np.clip(img, 0.0, 1.0)


def _as_numpy_vision(vision):
    if hasattr(vision, "detach"):
        vision = vision.detach().cpu()
    return np.asarray(vision, dtype=float)


def _selected_cylinder_bins(vision, occ_threshold, group_adjacent, max_cylinders):
    occ = vision[:, 0]
    active = np.flatnonzero(occ > occ_threshold)

    if active.size == 0:
        return active

    if group_adjacent:
        breaks = np.where(np.diff(active) > 1)[0] + 1
        groups = np.split(active, breaks)
        active = np.array([group[np.argmax(occ[group])] for group in groups], dtype=int)

    if max_cylinders is not None and active.size > max_cylinders:
        strongest = np.argsort(occ[active])[-max_cylinders:]
        active = active[strongest]

    return np.sort(active)


def _cylinder_span_pixels(
    bin_idx,
    radius,
    depth,
    num_bins,
    image_width,
    fov_degrees,
    min_width_px,
    flip_x=True,
):
    if depth <= 0.0:
        return None

    fov = np.deg2rad(fov_degrees)
    theta = np.linspace(-0.5 * fov, 0.5 * fov, num_bins)[bin_idx]
    focal_px = image_width / (2.0 * np.tan(0.5 * fov))

    rel_radius = np.clip(radius / max(depth, 1e-6), 0.0, 0.98)
    half_angle = np.arcsin(rel_radius)

    image_sign = -1.0 if flip_x else 1.0
    x_center = image_width / 2.0 + image_sign * focal_px * np.tan(theta)
    x_left = image_width / 2.0 + image_sign * focal_px * np.tan(theta - half_angle)
    x_right = image_width / 2.0 + image_sign * focal_px * np.tan(theta + half_angle)

    if x_left > x_right:
        x_left, x_right = x_right, x_left

    if x_right - x_left < min_width_px:
        half_width = 0.5 * min_width_px
        x_left = x_center - half_width
        x_right = x_center + half_width

    x_left = np.clip(x_left, 0.0, image_width - 1.0)
    x_right = np.clip(x_right, 0.0, image_width - 1.0)
    x_center = np.clip(x_center, 0.0, image_width - 1.0)

    return x_left, x_center, x_right


def plot_estimated_cylinders_on_images(
    images,
    visions,
    titles=None,
    occ_threshold=0.4,
    fov_degrees=90.0,
    flip_x=True,
    color="tab:orange",
    alpha=0.22,
    line_width=1.8,
    min_width_px=2.0,
    group_adjacent=True,
    max_cylinders=None,
    figsize=None,
):
    """
    Plotta estimerade cylindrar från vision-output ovanpå en eller flera bilder.

    images: en bild (C,H,W) eller lista med bilder.
    visions: motsvarande vision tensor/array med shape (N,3) eller (N,4):
        [occupancy, radius, depth, optional_id].
    flip_x: True när bin 0 ligger till höger i kamerabilden.
    """
    if not isinstance(images, (list, tuple)):
        images = [images]
        visions = [visions]

    if titles is None:
        titles = [None] * len(images)

    if len(images) != len(visions):
        raise ValueError("images och visions måste ha samma längd")

    if len(titles) != len(images):
        raise ValueError("titles måste vara None eller ha samma längd som images")

    if figsize is None:
        figsize = (5 * len(images), 4)

    fig, axes = plt.subplots(1, len(images), figsize=figsize, squeeze=False)
    axes = axes[0]

    for ax, img, vision, title in zip(axes, images, visions, titles):
        img_np = _as_numpy_image(img)
        vision_np = _as_numpy_vision(vision)

        if vision_np.ndim != 2 or vision_np.shape[1] < 3:
            raise ValueError("vision måste ha shape (N,3) eller (N,4)")

        height, width = img_np.shape[:2]
        ax.imshow(img_np)

        active_bins = _selected_cylinder_bins(
            vision_np,
            occ_threshold=occ_threshold,
            group_adjacent=group_adjacent,
            max_cylinders=max_cylinders,
        )

        for bin_idx in active_bins:
            occ, radius, depth = vision_np[bin_idx, :3]
            span = _cylinder_span_pixels(
                bin_idx=bin_idx,
                radius=radius,
                depth=depth,
                num_bins=vision_np.shape[0],
                image_width=width,
                fov_degrees=fov_degrees,
                min_width_px=min_width_px,
                flip_x=flip_x,
            )

            if span is None:
                continue

            x_left, x_center, x_right = span
            rect_width = max(x_right - x_left, min_width_px)

            ax.add_patch(
                Rectangle(
                    (x_left, 0.0),
                    rect_width,
                    height - 1.0,
                    facecolor=color,
                    edgecolor="none",
                    alpha=alpha,
                )
            )
            ax.add_patch(
                Rectangle(
                    (x_left, 0.0),
                    rect_width,
                    height - 1.0,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=line_width,
                    alpha=0.95,
                )
            )
            ax.axvline(x_center, color=color, linewidth=1.0, alpha=0.9)
            ax.text(
                x_center,
                2.0,
                f"{occ:.2f}",
                color=color,
                fontsize=8,
                ha="center",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.5},
            )

        if title is not None:
            ax.set_title(title)

        ax.set_xlim(0, width - 1)
        ax.set_ylim(height - 1, 0)
        ax.axis("off")

    fig.tight_layout()
    plt.show()

    return fig, axes
