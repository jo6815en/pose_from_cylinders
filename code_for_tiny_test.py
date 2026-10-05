# ============================================================
# TINY-DATASET OVERFIT TEST
# Ändra endast TINY_N mellan 4 och 16
# ============================================================

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

TINY_N = 4          # <-- kör först 4, sedan 16
TINY_EPOCHS = 800
TINY_LR = 3e-4
TINY_BATCH_SIZE = 4 #min(TINY_N, 16)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

tiny_dataset = Subset(
    train_loader.dataset,
    range(TINY_N),
)

tiny_loader = DataLoader(
    tiny_dataset,
    batch_size=TINY_BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

# ------------------------------------------------------------
# Ny modell
# ------------------------------------------------------------

tiny_model = PairImageCylinderModel(
    img_size=128,
    patch_size=(16, 8),
    in_chans=3,
    embed_dim=192,
    depth=4,
    num_heads=4,
    num_bins=NUM_BINS,
    dropout=0.0,
).to(device)

tiny_optimizer = torch.optim.AdamW(
    tiny_model.parameters(),
    lr=TINY_LR,
    weight_decay=0.0,
)

# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

for epoch in range(TINY_EPOCHS):
    tiny_model.train()
    epoch_loss = 0.0

    for batch in tiny_loader:
        batch = [
            x.to(device, non_blocking=True)
            for x in batch
        ]

        # Datasetet kan returnera ett eller två pairs
        if len(batch) == 10:
            pairs = [
                (batch[0], batch[1], batch[2], batch[3]),
                (batch[5], batch[6], batch[7], batch[8]),
            ]
        else:
            pairs = [
                (batch[0], batch[1], batch[2], batch[3])
            ]

        tiny_optimizer.zero_grad(set_to_none=True)

        pair_losses = []

        for img_a, gt_a, img_b, gt_b in pairs:

            pred_a, pred_b, _ = tiny_model(
                img_a,
                img_b,
                compute_pose=False,
            )

            loss_a, *_ = vision_loss(
                pred_a,
                gt_a,
                occ_thresh=OCC_THRESH,
                lambda_occ=LAMBDA_OCC,
                lambda_radius=LAMBDA_RADIUS,
                lambda_depth=1.0,
            )

            loss_b, *_ = vision_loss(
                pred_b,
                gt_b,
                occ_thresh=OCC_THRESH,
                lambda_occ=LAMBDA_OCC,
                lambda_radius=LAMBDA_RADIUS,
                lambda_depth=1.0,
            )

            pair_losses.append(
                0.5 * (loss_a + loss_b)
            )

        loss = torch.stack(pair_losses).mean()

        loss.backward()
        tiny_optimizer.step()

        epoch_loss += loss.item()

    if epoch == 0 or (epoch + 1) % 20 == 0:
        print(
            f"N={TINY_N:2d} | "
            f"Epoch {epoch+1:3d}/{TINY_EPOCHS} | "
            f"loss={epoch_loss / len(tiny_loader):.4f}"
        )


# ------------------------------------------------------------
# Evaluate på EXAKT samma samples
# ------------------------------------------------------------

tiny_model.eval()

depth_pred = []
depth_gt = []
radius_pred = []
radius_gt = []

with torch.no_grad():

    eval_loader = DataLoader(
        tiny_dataset,
        batch_size=TINY_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    for batch in eval_loader:
        batch = [x.to(device) for x in batch]

        if len(batch) == 10:
            pairs = [
                (batch[0], batch[1], batch[2], batch[3]),
                (batch[5], batch[6], batch[7], batch[8]),
            ]
        else:
            pairs = [
                (batch[0], batch[1], batch[2], batch[3])
            ]

        for img_a, gt_a, img_b, gt_b in pairs:

            pred_a, pred_b, _ = tiny_model(
                img_a,
                img_b,
                compute_pose=False,
            )

            for pred, gt in [
                (pred_a, gt_a),
                (pred_b, gt_b),
            ]:
                mask = gt[..., 0] > OCC_THRESH

                depth_pred.append(
                    pred[..., 2][mask].cpu()
                )
                depth_gt.append(
                    gt[..., 2][mask].cpu()
                )

                radius_pred.append(
                    pred[..., 1][mask].cpu()
                )
                radius_gt.append(
                    gt[..., 1][mask].cpu()
                )


dp = torch.cat(depth_pred).numpy()
dg = torch.cat(depth_gt).numpy()

rp = torch.cat(radius_pred).numpy()
rg = torch.cat(radius_gt).numpy()


print(f"\nTINY-SET OVERFIT — N={TINY_N}")
print("=" * 50)

print("\nDEPTH")
print("-" * 50)
print(f"MAE:              {np.mean(np.abs(dp - dg)):.4f}")
print(f"Correlation:      {np.corrcoef(dp, dg)[0, 1]:.4f}")
print(f"Pred std:         {dp.std():.4f}")
print(f"GT std:           {dg.std():.4f}")

print("\nRADIUS")
print("-" * 50)
print(f"MAE:              {np.mean(np.abs(rp - rg)):.4f}")
print(f"Correlation:      {np.corrcoef(rp, rg)[0, 1]:.4f}")
print(f"Pred std:         {rp.std():.4f}")
print(f"GT std:           {rg.std():.4f}")