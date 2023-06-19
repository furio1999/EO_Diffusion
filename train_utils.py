import timeit
import math
from typing import Sequence, Mapping, Literal, Callable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

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
