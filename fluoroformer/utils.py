import torch
from torchmetrics import Metric


def _get_concordant(event_times, predicted_scores, event_observed=None, drop_nans=True):
    device = event_times.device

    if event_observed is None:
        event_observed = torch.ones(
            event_times.shape[0], dtype=torch.bool, device=device
        )

    if drop_nans:
        valid = ~torch.isnan(event_times) & ~torch.isnan(predicted_scores)
        event_times = event_times[valid]
        predicted_scores = predicted_scores[valid]
        event_observed = event_observed[valid]

    # Ensure 1D tensors
    event_times = event_times.view(-1)
    predicted_scores = predicted_scores.view(-1)
    event_observed = event_observed.view(-1)

    # Create 2D matrices for all pairs of event times and predictions
    event_times_matrix = event_times.view(-1, 1).repeat(1, len(event_times))
    predicted_scores_matrix = predicted_scores.view(-1, 1).repeat(
        1, len(predicted_scores)
    )

    # Compute differences between pairs
    event_time_diffs = event_times_matrix - event_times_matrix.t()
    predicted_score_diffs = predicted_scores_matrix - predicted_scores_matrix.t()

    # Identify permissible pairs (only consider pairs (i, j) where either i or j is uncensored)
    permissible_mask = event_observed.unsqueeze(1) | event_observed.unsqueeze(0)

    # Identify concordant pairs (excluding ties)
    concordant = (event_time_diffs * predicted_score_diffs > 0) & permissible_mask

    # Exclude diagonal elements (self-pairs)
    diagonal_mask = torch.eye(len(event_times), dtype=torch.bool, device=device)
    concordant = concordant & ~diagonal_mask
    permissible_mask = permissible_mask & ~diagonal_mask

    num_pairs = permissible_mask.sum().float()
    num_concordant = concordant.sum().float()

    return num_pairs, num_concordant


def concordance_index(
    event_times, predicted_scores, event_observed=None, drop_nans=True
):
    num_pairs, num_concordant = _get_concordant(
        event_times, predicted_scores, event_observed, drop_nans=drop_nans
    )
    if num_pairs == 0:
        return torch.tensor(0.0, device=event_times.device)

    c_index = num_concordant / num_pairs
    return c_index


class ConcordanceIndex(Metric):
    def __init__(self, drop_nans=True, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.drop_nans = drop_nans
        self.add_state("num_pairs", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("num_concordant", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, event_times, predicted_scores, event_observed=None):
        num_pairs, num_concordant = _get_concordant(
            event_times, predicted_scores, event_observed, drop_nans=self.drop_nans
        )
        self.num_pairs += num_pairs.long()
        self.num_concordant += num_concordant.long()

    def compute(self):
        # Final computation to get the concordance index
        return (
            self.num_concordant / self.num_pairs
            if self.num_pairs > 0
            else torch.tensor(0.0, device=self.device)
        )

    def reset(self):
        self.num_pairs = torch.tensor(0, device=self.device)
        self.num_concordant = torch.tensor(0, device=self.device)
