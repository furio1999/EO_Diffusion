import timeit
import math
from typing import Sequence, Mapping, Literal, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import Callback, ModelCheckpoint
from utils import ExponentialMovingAverage

from timm.utils.model import get_state_dict, unwrap_model
#from gale.utils.logger import log_main_process

# https://betterprogramming.pub/a-keyframe-style-learning-rate-scheduler-for-pytorch-b889110dcde8


class KeyframeLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        frames: Sequence[Mapping | Sequence[float] | str],
        end: float,
        units: Literal["percent", "steps", "time"] = "percent",
    ):
        """
        Define a PyTorch LR scheduler with keyframes

        Parameters
        ----------
        optimizer
            torch.optim optimizer

        frames
            A sequence of mappings (e.g. list of dicts), each one either specifying a
            position/lr or transition.

            Positions should be defined like `{"position": 0.2, "lr": 0.1}`.
            As a shorthand, you can also provide a list or tuple with the position/lr

            When units are `"steps"`, define the position in steps, else define the position as
            a float in the interval [0, 1].

            Transitions can optionally be inserted between positions, e.g. `{"transform": "cos"}`
            If no transition is defined between two positions, `linear` will be used.
            Options are `"linear"` and `"cos"`, or a function with the signature:
            `func(last_lr, start_frame, end_frame, position, scheduler)`
            As a shorthand, you can also provide just the string or callable

        end
            When `units` are `"time"`, this should be the expected run-time in seconds
            Otherwise, this should be the maximum number of times you plan to call .step()

        units
            "percent", "steps", or "time". Default is "percent"
        """
        self.end = end
        self.units = units
        self.frames = self.parse_frames(frames)
        self.last_lr = 0
        self.start_time = timeit.default_timer() if units == "time" else None

        super().__init__(optimizer=optimizer)

    def parse_frames(self, user_frames):
        frames = []
        previous_pos = -1
        end_pos = self.end if self.units == "steps" else 1

        unpacked_frames = []
        for frame in user_frames:
            # Allow shorthand for position
            if isinstance(frame, Sequence) and len(frame) == 2:
                frame = {"position": frame[0], "lr": frame[1]}

            # Allow shorthand for transition
            if isinstance(frame, (str, Callable)):
                frame = {"transition": frame}

            # Allow for "position": "end"
            if frame.get("position", None) == "end":
                frame["position"] = end_pos
            unpacked_frames.append(frame)

        for i, frame in enumerate(unpacked_frames):
            first_frame = i == 0
            last_frame = i == len(unpacked_frames) - 1
            if first_frame:
                if "position" in frame and frame["position"] != 0:
                    frames.append({"position": 0, "lr": 0})
                    frames.append({"transition": "linear"})
                if "transition" in frame:
                    frames.append({"position": 0, "lr": 0})

            frames.append(frame)

            if "position" in frame:
                position = frame["position"]
                assert (
                    position >= previous_pos
                ), f"position {position!r} is not bigger than {previous_pos}"
                assert (
                    position <= end_pos
                ), f"position {position} is bigger than end value {end_pos}"
                previous_pos = position

                if not last_frame:
                    next_frame = unpacked_frames[i + 1]
                    if "position" in next_frame:
                        frames.append({"transition": "linear"})

            if last_frame:
                if "position" in frame and frame["position"] < end_pos:
                    frames.append({"transition": "linear"})
                    frames.append({"position": end_pos, "lr": 0})
                if "transition" in frame:
                    frames.append({"position": end_pos, "lr": 0})

        return frames

    @staticmethod
    def interpolate(a, b, pct):
        return (1 - pct) * a + pct * b

    def interpolate_frames(self, start_frame, transition, end_frame, position):
        pos_range = end_frame["position"] - start_frame["position"]
        pct_of_range = (position - start_frame["position"]) / pos_range

        if transition == "linear":
            return self.interpolate(
                start_frame["lr"],
                end_frame["lr"],
                pct_of_range,
            )
        if transition == "cos":
            pct_of_range_cos = 1 - (1 + math.cos(pct_of_range * math.pi)) / 2
            return self.interpolate(
                start_frame["lr"],
                end_frame["lr"],
                pct_of_range_cos,
            )

        if isinstance(transition, Callable):
            return transition(self.last_lr, start_frame, end_frame, position, self)

        raise ValueError(f"Unknown transition: {transition!r}")

    def get_lr_at_pos(self, position):
        start_frame = None
        transition = None
        end_frame = None
        lr = None

        for frame in self.frames:
            if "position" in frame:
                if frame["position"] == position:
                    lr = frame["lr"]
                    # Direct match, we're done
                    break
                if frame["position"] < position:
                    start_frame = frame

            if start_frame is not None and "transition" in frame:
                transition = frame["transition"]

            if (
                transition is not None
                and "position" in frame
                and frame["position"] >= position
            ):
                end_frame = frame
                break

        if lr is None:
            if start_frame is None or end_frame is None:
                print(f"No matching frames at position {position}, using last LR.")
                return self.last_lr

            lr = self.interpolate_frames(start_frame, transition, end_frame, position)

        # We store last_lr here so that custom transitions work with .sample_lrs()
        self.last_lr = lr
        return lr

    @property
    def progress(self):
        if self.units == "time":
            return (timeit.default_timer() - self.start_time) / self.end
        return self.last_epoch / self.end

    def get_lr(self):
        if self.units == "percent":
            position = self.last_epoch / self.end
        elif self.units == "steps":
            position = self.last_epoch
        elif self.units == "time":
            position = (timeit.default_timer() - self.start_time) / self.end
        else:
            raise TypeError(f"Unknown units {self.units}")

        lr = self.get_lr_at_pos(position)

        return [lr for _ in self.optimizer.param_groups]

    def sample_lrs(self, n=100):
        """
        Get a sample of the LRs that would be produced, for visualization.
        This might not work well with custom transitions.
        """
        # We don't want to generate a huge number of steps or affect optimizer state
        # so don't use the scheduler.step() machinery.
        # Instead, we loop manually and call get_lr_at_pos() directly
        lrs = []

        for i in range(n):
            pos = i / n
            if self.units == "steps":
                pos *= self.end
            lrs.append(self.get_lr_at_pos(pos))

        self.last_lr = 0

        return lrs

    def print_frames(self):
        for frame in self.frames:
            print(frame)

class CustomModelCheckpoint(ModelCheckpoint):
    def format_checkpoint_name(self, metrics, filename=None, ver=None):
        increm_metrics = copy.deepcopy(metrics)
        increm_metrics['epoch'] += 1
        increm_metrics['step'] += 1
        return super().format_checkpoint_name(increm_metrics, filename, ver)

class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    """

    def __init__(self, decay=0.9999, use_ema_weights: bool = True):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        adjust = 1* trainer.batch_size * len(trainer.train_dataloader) / trainer.max_epochs
        alpha = 1.0 - self.decay
        alpha = min(1.0, alpha * adjust)
        self.ema = ExponentialMovingAverage(pl_module, device=pl_module.device, decay=1.0 - alpha)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        self.ema.update(pl_module)

    def on_validation_epoch_start(self, trainer, pl_module):
        "do validation using the stored parameters"
        # save original parameters before replacing with EMA version
        self.store(pl_module.parameters())

        # update the LightningModule with the EMA weights
        # ~ Copy EMA parameters to LightningModule
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        # update the LightningModule with the EMA weights
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            msg = "Model weights replaced with the EMA version."
            #log_main_process(_logger, logging.INFO, msg)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}

    def on_load_checkpoint(self, callback_state):
        if self.ema is not None:
            self.ema.module.load_state_dict(callback_state["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
