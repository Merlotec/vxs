# losses.py
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss, TverskyLoss, FocalLoss
def make_recon_loss(name: str, **kwargs):
    name = name.lower()
    
    if name == "ce":
        # knobs (can be overridden per-call via extra)
        label_smoothing = float(kwargs.get("label_smoothing", 0.05))  # small but helpful
        default_gamma   = kwargs.get("focal_gamma", None)             # e.g. 1.0–2.0 or None
        ignore_index    = kwargs.get("ignore_index", -100)            # only used if you pass such labels

        def f(logits, target, extra=None):
            """
            logits: [B,C,...], target: [...], int64
            extra:
              - class_weights: Tensor[C]
              - mask: optional bool/float tensor broadcastable to target (ignored voxels = 0)
              - focal_gamma: optional float to enable focal modulation
            """
            extra = extra or {}
            w     = extra.get("class_weights", None)
            mask  = extra.get("mask", None)
            gamma = extra.get("focal_gamma", default_gamma)

            if w is not None:
                w = w.to(logits.device, dtype=logits.dtype)

            # compute per-element CE so we can apply focal/mask safely
            ce = F.cross_entropy(
                logits, target,
                weight=w,
                label_smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction="none",
            )

            # optional focal modulation on top of weighted CE
            if gamma is not None and gamma > 0:
                with torch.no_grad():
                    # p_t = softmax(logits)[range, target]
                    pt = logits.softmax(dim=1).gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1).clamp_min(1e-6)
                ce = ((1.0 - pt) ** float(gamma)) * ce

            # optional mask (e.g., to ignore unlabeled voxels)
            if mask is not None:
                m = mask.to(logits.device, dtype=ce.dtype)
                return (ce * m).sum() / m.sum().clamp_min(1.0)
            else:
                return ce.mean()

        return f


# ===================== helpers =====================
def _soft_occupancy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B, 3, D, H, W]
    Return soft occupancy prob ∈ [0,1]. Treat class2=filled, class1=sparse (half weight).
    """
    p = logits.softmax(dim=1)
    return (p[:, 2] + 0.5 * p[:, 1]).clamp(0, 1)  # [B,D,H,W]


def _avg_downsample_3d(x: torch.Tensor, scale: int) -> torch.Tensor:
    if scale == 1:
        return x
    return F.avg_pool3d(x.unsqueeze(1), kernel_size=scale, stride=scale).squeeze(1)


def _pairwise_dist2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # a: [Na,3], b: [Nb,3] -> [Na,Nb]
    return ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)


# ===================== shape-aware losses =====================
def soft_iou_loss(logits: torch.Tensor, target_occ: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """
    Soft IoU/Jaccard on occupancy.
    target_occ: [B,D,H,W] float in [0,1]
    """
    pred = _soft_occupancy_from_logits(logits)
    inter = (pred * target_occ).sum(dim=(1, 2, 3))
    union = (pred + target_occ - pred * target_occ).sum(dim=(1, 2, 3)).clamp_min(eps)
    return (1.0 - inter / union).mean()


def tv_smoothness_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    3D total variation on soft occupancy to encourage smooth surfaces.
    """
    occ = _soft_occupancy_from_logits(logits)
    dx = occ[:, 1:, :, :] - occ[:, :-1, :, :]
    dy = occ[:, :, 1:, :] - occ[:, :, :-1, :]
    dz = occ[:, :, :, 1:] - occ[:, :, :, :-1]
    return dx.abs().mean() + dy.abs().mean() + dz.abs().mean()


def boundary_loss(logits: torch.Tensor, *, labels: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Distance-transform-weighted CE at boundaries.
    labels: [B,D,H,W] Long {0/1 or 0/1/2}. If 3-class, 1|2 => occupied.
    dt:     [B,D,H,W] Float distance-to-occupied (0 at boundary/occupied).
    """
    # map 3-class → binary
    if labels.max() > 1:
        labels = ((labels == 1) | (labels == 2)).long()

    # build 2-class logits: empty vs occupied (merge classes 1 and 2)
    occ_logit = logits[:, 1:].logsumexp(dim=1)
    logits2 = torch.stack([logits[:, 0], occ_logit], dim=1)  # [B,2,D,H,W]

    w = 1.0 / (1.0 + dt)  # emphasize near boundary
    ce = F.cross_entropy(logits2, labels, reduction="none")
    return (ce * w).mean()


def center_of_mass_loss(logits: torch.Tensor, target_com: torch.Tensor) -> torch.Tensor:
    """
    Align predicted COM with target COM.
    target_com: [B,3] in voxel coords (x,y,z).
    """
    occ = _soft_occupancy_from_logits(logits)  # [B,D,H,W]
    B, D, H, W = occ.shape
    z = torch.linspace(0, D - 1, D, device=occ.device)
    y = torch.linspace(0, H - 1, H, device=occ.device)
    x = torch.linspace(0, W - 1, W, device=occ.device)

    sum_occ = occ.sum(dim=(1, 2, 3)).clamp_min(1e-6)
    zc = (occ.sum(dim=(2, 3)) * z).sum(dim=1) / sum_occ
    yc = (occ.sum(dim=(1, 3)) * y).sum(dim=1) / sum_occ
    xc = (occ.sum(dim=(1, 2)) * x).sum(dim=1) / sum_occ

    pred_com = torch.stack([xc, yc, zc], dim=1)
    return F.smooth_l1_loss(pred_com, target_com)


def class_balance_loss(logits: torch.Tensor, prior: torch.Tensor = None) -> torch.Tensor:
    """
    Encourage reasonable class proportions across the volume.
    prior: optional [3] tensor that sums to 1. If None, maximize entropy.
    """
    p = logits.softmax(dim=1)            # [B,3,D,H,W]
    freq = p.mean(dim=(0, 2, 3, 4))      # [3]
    if prior is None:
        entropy = -(freq * freq.clamp_min(1e-6).log()).sum()
        return -entropy
    prior = prior / prior.sum()
    return F.kl_div(freq.log(), prior, reduction="batchmean")


def multiscale_occ_loss(logits: torch.Tensor, target_low: torch.Tensor, *, scale: int = 4) -> torch.Tensor:
    """
    Match low-res occupancy via average pooling.
    target_low: [B, D/scale, H/scale, W/scale] float ∈ [0,1]
    """
    pred = _soft_occupancy_from_logits(logits)
    pred_low = _avg_downsample_3d(pred, scale)
    return F.binary_cross_entropy(pred_low, target_low)


def chamfer_points_loss(
    logits: torch.Tensor,
    target_points: list,
    *,
    topk: int = 2048
) -> torch.Tensor:
    """
    Chamfer between top-K predicted occupied voxels and GT points per sample.
    target_points: list of length B with tensors [Ni,3] in (x,y,z) voxel coords.
    NOTE: Top-K selection is non-differentiable over indices; gradients still flow to chosen voxels.
    """
    occ = _soft_occupancy_from_logits(logits)  # [B,D,H,W]
    B, D, H, W = occ.shape
    flat = occ.view(B, -1)
    k = min(topk, flat.shape[1])
    vals, idx = flat.topk(k, dim=1)  # [B,k]

    # decode flat indices to (z,y,x)
    z = (idx // (H * W))
    y = (idx % (H * W)) // W
    x = (idx % W)
    pred_pts_list = [torch.stack([x[i], y[i], z[i]], dim=1).float() for i in range(B)]

    per_sample = []
    for pred_pts, tgt_pts in zip(pred_pts_list, target_points):
        if pred_pts.numel() == 0 or tgt_pts.numel() == 0:
            per_sample.append(torch.tensor(0.0, device=logits.device))
            continue
        tgt_pts = tgt_pts.to(logits.device).float()
        d2 = _pairwise_dist2(pred_pts, tgt_pts)
        per_sample.append(d2.min(dim=1).values.mean() + d2.min(dim=0).values.mean())
    return torch.stack(per_sample).mean()


# ===================== factory =====================
def make_aux_loss(name: str, **kwargs):
    """
    Returns a function `fn(logits, targets, extra=None) -> loss`.
    You decide what keys to put in `targets` based on the loss.

    Built-ins (name → expected targets):
      - "soft_iou"        → targets["occ_dense"]
      - "tv"              → (no targets)
      - "boundary"        → targets["labels"], targets["dt"]
      - "com"             → targets["com"]
      - "class_balance"   → optional prior via extra["prior"]
      - "ms_occ"          → targets["occ_low"], kw["scale"]
      - "chamfer"         → targets["points"] (list of [Ni,3]), kw["topk"]
    """
    name = name.lower()

    if name == "soft_iou":
        return lambda logits, targets, extra=None: soft_iou_loss(
            logits, targets["occ_dense"]
        )

    if name == "tv":
        return lambda logits, targets=None, extra=None: tv_smoothness_loss(logits)

    if name == "boundary":
        return lambda logits, targets, extra=None: boundary_loss(
            logits, labels=targets["labels"], dt=targets["dt"]
        )

    if name == "com":
        return lambda logits, targets, extra=None: center_of_mass_loss(
            logits, targets["com"]
        )

    if name == "class_balance":
        return lambda logits, targets=None, extra=None: class_balance_loss(
            logits, prior=None if extra is None else extra.get("prior")
        )

    if name == "ms_occ":
        scale = int(kwargs.get("scale", 4))
        return lambda logits, targets, extra=None: multiscale_occ_loss(
            logits, targets["occ_low"], scale=scale
        )

    if name == "chamfer":
        topk = int(kwargs.get("topk", 2048))
        return lambda logits, targets, extra=None: chamfer_points_loss(
            logits, targets["points"], topk=topk
        )

    raise ValueError(f"Unknown aux loss name: {name}")