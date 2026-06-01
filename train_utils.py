from losses import supervised_loss, matched_radius_consistency_loss, matched_reprojection_loss_2d

NUM_EPOCHS = 500
VAL_INTERVAL = 5
FOREST_START_EPOCH = 50

LAMBDA_OCC = 10.0
LAMBDA_RADIUS = 10.0
OCC_THRESH = 0.5

LAMBDA_REPROJ = 5.0
LAMBDA_RAD = 10.0

def move_two_pair_batch_to_device(batch, device):
    return tuple(x.to(device, non_blocking=True) for x in batch)


def compute_two_pair_losses(batch, model, device):
    (
        img_a1,
        vision_a1,
        img_b1,
        vision_b1,
        pose_ab1,
        img_a2,
        vision_a2,
        img_b2,
        vision_b2,
        pose_ab2,
    ) = move_two_pair_batch_to_device(batch, device)

    pred_vision_a1, pred_vision_b1, pred_pose1 = model(img_a1, img_b1)
    pred_vision_a2, pred_vision_b2, pred_pose2 = model(img_a2, img_b2)

    loss_out1 = supervised_loss(
        pred_vision_a1, vision_a1,
        pred_vision_b1, vision_b1,
        pred_pose1, pose_ab1,
        occ_thresh=OCC_THRESH,
        lambda_occ=LAMBDA_OCC,
        lambda_radius=LAMBDA_RADIUS,
    )

    loss_out2 = supervised_loss(
        pred_vision_a2, vision_a2,
        pred_vision_b2, vision_b2,
        pred_pose2, pose_ab2,
        occ_thresh=OCC_THRESH,
        lambda_occ=LAMBDA_OCC,
        lambda_radius=LAMBDA_RADIUS,
    )

    radius_cons1 = matched_radius_consistency_loss(
        pred_vision_a1, vision_a1,
        pred_vision_b1, vision_b1,
        relative_pose_pred=pred_pose1,
        matching_mode="sinkhorn",
        occ_thresh=OCC_THRESH,
    )

    radius_cons2 = matched_radius_consistency_loss(
        pred_vision_a2, vision_a2,
        pred_vision_b2, vision_b2,
        relative_pose_pred=pred_pose2,
        matching_mode="sinkhorn",
        occ_thresh=OCC_THRESH,
    )

    reproj1 = matched_reprojection_loss_2d(
        pred_vision_a1, vision_a1,
        pred_vision_b1, vision_b1,
        pred_pose1,
        matching_mode="sinkhorn",
        occ_thresh=OCC_THRESH,
        fov_degrees=90.0,
    )

    reproj2 = matched_reprojection_loss_2d(
        pred_vision_a2, vision_a2,
        pred_vision_b2, vision_b2,
        pred_pose2,
        matching_mode="sinkhorn",
        occ_thresh=OCC_THRESH,
        fov_degrees=90.0,
    )

    sup_loss = (loss_out1["total"] + loss_out2["total"]) / 2.0
    reproj = (reproj1 + reproj2) / 2.0
    radius_cons = (radius_cons1 + radius_cons2) / 2.0

    loss = sup_loss

    metrics = {
        "total": loss.item(),
        "supervised": sup_loss.item(),
        "vision": ((loss_out1["vision_a"] + loss_out2["vision_a"]) / 2.0).item(),
        "pose": ((loss_out1["pose"] + loss_out2["pose"]) / 2.0).item(),
        "radius_consistency": radius_cons.item(),
        "reprojection": reproj.item(),
    }

    return loss, metrics


def compute_forest_losses(batch, model, device):
    img_a, img_b, _ = move_two_pair_batch_to_device(batch, device)

    pred_vision_a, pred_vision_b, pred_pose = model(img_a, img_b)

    radius_cons = matched_radius_consistency_loss(
    pred_vision_a,
    pred_vision_b=pred_vision_b,
    relative_pose_pred=pred_pose,
    matching_mode="sinkhorn",
    occ_thresh=OCC_THRESH,
    )

    reproj = matched_reprojection_loss_2d(
        pred_vision_a,
        pred_vision_b=pred_vision_b,
        relative_pose_pred=pred_pose,
        matching_mode="sinkhorn",
        occ_thresh=OCC_THRESH,
        fov_degrees=90.0,
    )

    loss = LAMBDA_RAD * radius_cons + LAMBDA_REPROJ * reproj

    metrics = {
        "total": loss.item(),
        "supervised": 0.0,
        "vision": 0.0,
        "pose": 0.0,
        "radius_consistency": radius_cons.item(),
        "reprojection": reproj.item(),
    }

    return loss, metrics