import matplotlib.pyplot as plt
import numpy as np

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