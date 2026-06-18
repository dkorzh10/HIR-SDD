from .base import Trainer
from ..epochs.sft_epoch import SFTTrainEpoch


def _compute_dataset_label_proportions(train_loader):
    """Compute label proportions from the training dataset (once per epoch).
    Expects dataset to have .samples with 'is_bonafide' (bool or 0/1).
    Returns dict mapping label index 0/1 -> proportion, e.g. {0: 0.6, 1: 0.4}.
    """
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None:
        return None
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    counts = {0: 0, 1: 0}
    for s in samples:
        ib = s.get("is_bonafide")
        label = 1 if (ib is True or ib == 1) else 0
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return None
    return {k: v / total for k, v in counts.items()}


class SFTTrainer(Trainer):
    def train(self):
        # if self.val_loader:
        #     print("Running pre-training validation (epoch -1)", flush=True)
        #     self.validate(-1)

        loss_weight_based_on_dataset_label = self.config.get("loss_weight_based_on_dataset_label", False)

        for epoch in range(self.num_epochs):
            print(f"Starting Epoch {epoch}", flush=True)
            dataset_label_proportions = None
            if loss_weight_based_on_dataset_label:
                dataset_label_proportions = _compute_dataset_label_proportions(self.train_loader)
                if dataset_label_proportions and (epoch == 0 or not hasattr(self, "_logged_proportions")):
                    print(f"Dataset label proportions (epoch {epoch}): {dataset_label_proportions}", flush=True)
                    self._logged_proportions = True

            opt_cfg = self.config.get("optimizator", {})
            bonafide_weight = opt_cfg.get("bonafide_weight")
            model_name = self.config.get("model_name")

            train_epoch = SFTTrainEpoch(
                self.model, self.train_loader, self.logger,
                self.optimizer, self.scheduler, self.scaler,
                device=self.device,
                amp=self.amp,
                iters_per_epoch=self.config.get("iters_per_epoch"),
                accum_grad_iters=self.config.get("accum_grad_iters", 1),
                grounding_debug_max_examples=self.config.get("grounding_debug_max_examples", 0),
                max_grad_norm=self.max_grad_norm,
                dataset_label_proportions=dataset_label_proportions,
                bonafide_weight=bonafide_weight,
                model_name=model_name,
            )
            train_epoch.run(epoch_num=epoch)

            val_accuracy = 0.0
            if self.val_loader:
                val_accuracy = self.validate(epoch)

            self.save_checkpoint(epoch, val_accuracy, name="sft")
