import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


def normalized_confidence(probabilities):
    num_classes = probabilities.size(1)
    entropy_value = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)
    return (1.0 - entropy_value / np.log(num_classes)).clamp(0.0, 1.0)


def js_stability(first, second):
    midpoint = 0.5 * (first + second)
    first_kl = (first * (torch.log(first + 1e-8) - torch.log(midpoint + 1e-8))).sum(dim=1)
    second_kl = (second * (torch.log(second + 1e-8) - torch.log(midpoint + 1e-8))).sum(dim=1)
    return (1.0 - 0.5 * (first_kl + second_kl) / np.log(2.0)).clamp(0.0, 1.0)


def _section(cfg, section_name=None):
    if section_name is not None and hasattr(cfg, section_name):
        return getattr(cfg, section_name)
    return getattr(cfg, 'DIFF_ALIGN', None) or getattr(cfg, 'ProDe', cfg)


def adaptive_mix_weight(i, mi, q_target, q_clip, target_output, clip_history,
                        clip_history_seen, observation_count, target_loss,
                        clip_loss, cfg, section_name=None):
    """Return a detached per-sample target mixing weight.

    q_target, q_clip and target_output must be probability distributions.  The
    function keeps the current diff-align gate convention so existing callers
    can migrate without changing their training behavior.
    """
    section = _section(cfg, section_name)
    target_confidence = normalized_confidence(q_target)
    clip_confidence = normalized_confidence(q_clip)
    target_stability = js_stability(q_target, target_output)
    clip_reference = torch.where(clip_history_seen.unsqueeze(1), clip_history, q_clip)
    clip_stability = js_stability(q_clip, clip_reference)

    confidence_weight = getattr(section, 'MERGE_CONFIDENCE_WEIGHT', 0.2)
    stability_weight = getattr(section, 'MERGE_STABILITY_WEIGHT', 0.8)
    weight_sum = confidence_weight + stability_weight
    target_reliability = (
        confidence_weight * target_confidence + stability_weight * target_stability
    ) / weight_sum
    clip_reliability = clip_confidence

    # Keep the current implementation's scale: reliability and loss are
    # combined in a bounded logit.  The old observation prior is represented
    # explicitly and can be enabled by setting MERGE_USE_OBSERVATION_PRIOR=1.
    use_prior = False#int(getattr(section, 'MERGE_USE_OBSERVATION_PRIOR', 0)) == 1
    if use_prior:
        initial_target_weight = getattr(section, 'MERGE_INITIAL_TARGET_WEIGHT', 0.1)
        initial_logit = np.log(initial_target_weight / (1.0 - initial_target_weight))
        prior_bias = initial_logit * torch.exp(
            -observation_count / getattr(section, 'MERGE_PRIOR_DECAY', 2.0)
        )
    else:
        prior_bias = torch.zeros_like(target_reliability)

    prior_logit = prior_bias + target_reliability - clip_reliability
    min_weight = getattr(section, 'MERGE_MIN_WEIGHT', 0.05)
    prior = torch.sigmoid(prior_logit).clamp(min_weight, 1.0 - min_weight)

    loss_gap = clip_loss.detach() - target_loss.detach()
    normalized_loss_gap = loss_gap.clamp(-5.0, 5.0)
    robust_temperature = getattr(section, 'MERGE_ROBUST_TEMPERATURE', 5.0)
    robust_logit = (torch.logit(prior) + normalized_loss_gap) / robust_temperature
    alpha = torch.sigmoid(robust_logit).clamp(min_weight, 1.0 - min_weight)

    return (
        alpha.detach(), prior.detach(), target_confidence.detach(),
        clip_confidence.detach(), target_stability.detach(),
        clip_stability.detach(), prior_bias.detach(),
        (
            (target_reliability - clip_reliability)
            / getattr(section, 'MERGE_GATE_TEMPERATURE', 0.25)
        ).detach(),
        (normalized_loss_gap / robust_temperature).detach(),
        loss_gap.detach(), observation_count.detach(),
        target_loss.detach(), clip_loss.detach()
    )


class TargetLogitBank:
    """Direct (non-EMA) per-sample bank of target logits.

    Stores the most recent target logits for each sample, overwriting them on
    every update -- this is the target prediction history, not the source
    bank.  Seed it with source logits so the first visit of each sample has a
    valid history.
    """

    def __init__(self, num_samples, num_classes, device, init_logits=None):
        self.num_samples = num_samples
        self.num_classes = num_classes
        if init_logits is not None:
            self.bank = init_logits.detach().clone()
        else:
            self.bank = torch.zeros(num_samples, num_classes, device=device)

    @property
    def device(self):
        return self.bank.device

    def update(self, idx, logits):
        # Direct store of the target logit history -- no EMA smoothing.
        self.bank[idx] = logits.detach().clone()

    def __getitem__(self, idx):
        return self.bank[idx]


def adaptive_alpha_from_logits(target_logits, clip_logits, cfg,
                               target_history_logits=None,
                               observation_count=None, step=0, max_step=1,
                               section_name=None, return_details=False):
    """Compute alpha for blending two logits tensors with shape ``[B, K]``."""
    target_logits = target_logits.detach()
    clip_logits = clip_logits.detach().to(target_logits.device)
    q_clip = torch.softmax(clip_logits, dim=1)

    if target_history_logits is None:
        q_target = torch.softmax(target_logits, dim=1)
    else:
        q_target = torch.softmax(
            target_history_logits.detach().to(target_logits.device), dim=1
        )
    target_output = torch.softmax(target_logits, dim=1)

    batch_size = target_logits.size(0)
    clip_history = q_clip
    clip_history_seen = torch.zeros(
        batch_size, dtype=torch.bool, device=target_logits.device
    )
    if observation_count is None:
        observation_count = torch.zeros(batch_size, device=target_logits.device)
    else:
        observation_count = observation_count.to(
            device=target_logits.device, dtype=target_logits.dtype
        )

    log_target = F.log_softmax(target_logits, dim=1)
    target_loss = -(q_target * log_target).sum(dim=1)
    clip_loss = -(q_clip * log_target).sum(dim=1)
    result = adaptive_mix_weight(
        step, max_step, q_target, q_clip, target_output, clip_history,
        clip_history_seen, observation_count, target_loss, clip_loss, cfg,
        section_name=section_name
    )
    return result if return_details else result[0]


def gate_diagnostics(alpha_bank, alpha_seen, prior_bank, target_prediction,
                     clip_prediction, label_bank, target_confidence,
                     clip_confidence, target_stability, clip_stability,
                     prior_term_bank, reliability_term_bank, loss_term_bank,
                     loss_gap_bank, target_loss_bank, clip_loss_bank,
                     observation_count_bank, cfg, section_name=None):
    """Summarize adaptive-alpha gate behavior over observed samples.

    All arguments are per-sample tensors indexed by the dataset sample id.
    Predictions and labels should use ``-1`` for unavailable entries.  The
    returned string is suitable for appending to a training log line.
    """
    valid = alpha_seen & (label_bank >= 0) & (target_prediction >= 0) & (clip_prediction >= 0)
    if not valid.any():
        return 'Gate: no observations'

    section = _section(cfg, section_name)
    alpha = alpha_bank[valid]
    target_prediction = target_prediction[valid]
    clip_prediction = clip_prediction[valid]
    labels = label_bank[valid]
    target_correct = target_prediction == labels
    clip_correct = clip_prediction == labels
    disagreement = target_prediction != clip_prediction
    selected_prediction = torch.where(alpha >= 0.5, target_prediction, clip_prediction)
    quantiles = torch.quantile(alpha, torch.tensor([0.1, 0.5, 0.9], device=alpha.device))

    target_only = target_correct & ~clip_correct
    clip_only = clip_correct & ~target_correct
    target_only_alpha = alpha[target_only].mean().item() if target_only.any() else float('nan')
    clip_only_alpha = alpha[clip_only].mean().item() if clip_only.any() else float('nan')
    if disagreement.any():
        target_disagreement_acc = target_correct[disagreement].float().mean().item() * 100
        clip_disagreement_acc = clip_correct[disagreement].float().mean().item() * 100
        selected_disagreement_acc = (
            selected_prediction[disagreement] == labels[disagreement]
        ).float().mean().item() * 100
    else:
        target_disagreement_acc = clip_disagreement_acc = selected_disagreement_acc = float('nan')

    low_confidence = float(getattr(section, 'MERGE_LOW_CONFIDENCE', 0.3))
    both_low = ((target_confidence[valid] < low_confidence) &
                (clip_confidence[valid] < low_confidence)).float().mean().item() * 100
    min_weight = float(getattr(section, 'MERGE_MIN_WEIGHT', 0.05))
    saturated_low = (alpha <= min_weight + 1e-6).float().mean().item() * 100
    saturated_high = (alpha >= 1.0 - min_weight - 1e-6).float().mean().item() * 100

    prior_term = prior_term_bank[valid]
    reliability_term = reliability_term_bank[valid]
    loss_term = loss_term_bank[valid]
    loss_gap = loss_gap_bank[valid]
    target_loss = target_loss_bank[valid]
    clip_loss = clip_loss_bank[valid]
    observation_count = observation_count_bank[valid]

    def summary(values):
        q = torch.quantile(values, torch.tensor([0.1, 0.5, 0.9], device=values.device))
        return values.mean().item(), q[0].item(), q[1].item(), q[2].item()

    prior_stats = summary(prior_term)
    reliability_stats = summary(reliability_term)
    loss_stats = summary(loss_term)
    gap_stats = summary(loss_gap)
    target_loss_stats = summary(target_loss)
    clip_loss_stats = summary(clip_loss)
    observation_stats = summary(observation_count)
    loss_saturated = (loss_term.abs() >= 1.0 - 1e-6).float().mean().item() * 100

    return (
        'Gate: n={}; alpha mean={:.3f}, p10/50/90={:.3f}/{:.3f}/{:.3f}; prior={:.3f}; '
        'conf target/clip={:.3f}/{:.3f}; stability target/clip={:.3f}/{:.3f}; '
        'sat low/high={:.1f}%/{:.1f}%; disagree={:.1f}%; '
        'disagree acc target/clip/selected={:.2f}%/{:.2f}%/{:.2f}%; '
        'oracle alpha target-only/clip-only={:.3f}/{:.3f}; both-low-conf={:.1f}%; '
        'terms prior/reliability/loss mean={:.3f}/{:.3f}/{:.3f}; '
        'terms p50={:.3f}/{:.3f}/{:.3f}; p90={:.3f}/{:.3f}/{:.3f}; '
        'loss sat={:.1f}%; gap mean/p50={:.4f}/{:.4f}; obs mean/p50={:.1f}/{:.1f}; '
        'scores targetCE/clipCE mean={:.3f}/{:.3f}; p50={:.3f}/{:.3f}'
    ).format(
        alpha.numel(), alpha.mean().item(), quantiles[0].item(), quantiles[1].item(),
        quantiles[2].item(), prior_bank[valid].mean().item(),
        target_confidence[valid].mean().item(), clip_confidence[valid].mean().item(),
        target_stability[valid].mean().item(), clip_stability[valid].mean().item(),
        saturated_low, saturated_high, disagreement.float().mean().item() * 100,
        target_disagreement_acc, clip_disagreement_acc, selected_disagreement_acc,
        target_only_alpha, clip_only_alpha, both_low,
        prior_stats[0], reliability_stats[0], loss_stats[0],
        prior_stats[2], reliability_stats[2], loss_stats[2],
        prior_stats[3], reliability_stats[3], loss_stats[3], loss_saturated,
        gap_stats[0], gap_stats[2], observation_stats[0], observation_stats[2],
        target_loss_stats[0], clip_loss_stats[0], target_loss_stats[2], clip_loss_stats[2]
    )




def test_on_source(cfg, netF, netB, netC, dset_loaders):
    ## evaluate the adapted model on the source split (GSFDA)
    netF.eval()
    netB.eval()
    netC.eval()
    start_test = True
    with torch.no_grad():
        iter_test = iter(dset_loaders['source_test'])
        for _ in range(len(dset_loaders['source_test'])):
            data = next(iter_test)
            inputs = data[0][0].cuda()
            labels = data[1]
            outputs = netC(netB(netF(inputs)))[:, :cfg.class_num]
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    acc_s = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) * 100

    if cfg.SETTING.DATASET == 'VISDA-C':
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        acc_list = ' '.join(str(np.round(i, 2)) for i in acc)
        log_str = 'GSFDA: Task: {}, Source Accuracy = {:.2f}%'.format(cfg.name_src, acc_s) + '\n' + acc_list
    else:
        log_str = 'GSFDA: Task: {}, Source Accuracy = {:.2f}%'.format(cfg.name_src, acc_s)
    logger.info(log_str)
    netF.train()
    netB.train()
    netC.train()
    return acc_s
