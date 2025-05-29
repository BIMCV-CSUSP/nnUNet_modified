import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import nnUNetTrainerNoDeepSupervision
from nnunetv2.training.loss.compound_losses import FL_and_CE_and_DC_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
import numpy as np



class nnUNetTrainerFocalCeDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = FL_and_CE_and_DC_loss(
            soft_dice_kwargs = {
                "batch_dice": self.configuration_manager.batch_dice,
                "do_bg": True,
                "smooth": 1e-5,
                "ddp": self.is_ddp,
            },
            dice_class=MemoryEfficientSoftDiceLoss
        )

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array(
                [1 / (2**i) for i in range(len(deep_supervision_scales))]
            )
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerFocalCeDiceLoss_5epochs(nnUNetTrainerFocalCeDiceLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 5

class nnUNetTrainerFocalCeDiceLoss_200epochs(nnUNetTrainerFocalCeDiceLoss):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        """used for debugging plans etc"""
        super().__init__(
            plans, configuration, fold, dataset_json, unpack_dataset, device
        )
        self.num_epochs = 200
