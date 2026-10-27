"""Steering controller producing simulation-only motor PWM proposals."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from pathdemo.config import ControllerConfig


@dataclass
class SteeringDecision:
    """Simulation-only steering decision and motor proposal."""
    state: str  # 'STRAIGHT', 'LEFT', 'RIGHT', 'PATH_LOST', 'STOP'
    steering_command: float  # Proportional + derivative effort
    left_pwm: int  # Bounded to [0, 255]
    right_pwm: int  # Bounded to [0, 255]
    normalized_error: float
    proportional_term: float
    derivative_term: float
    reason: str


class PathSteeringController:
    """Pure Python PD steering controller for differential-drive simulation.

    Differential Steering Sign Convention:
    - Path is centered (|e| <= deadband): STRAIGHT -> Left=Base, Right=Base
    - Path to Left (e < -deadband): Vehicle turns counter-clockwise -> Left slows down, Right speeds up -> State: LEFT
    - Path to Right (e > +deadband): Vehicle turns clockwise -> Left speeds up, Right slows down -> State: RIGHT
    - Path Lost (not valid): Immediate safety stop -> Left=0, Right=0 -> State: PATH_LOST / STOP

    All motor values are SIMULATION-ONLY proposals.
    """

    def __init__(self, config: ControllerConfig):
        self.config = config
        self.config.validate()
        self._prev_error: Optional[float] = None

    def reset(self) -> None:
        """Reset internal controller history."""
        self._prev_error = None

    def update_config(self, config: ControllerConfig) -> None:
        """Update controller parameters dynamically."""
        config.validate()
        self.config = config

    def compute(
        self,
        path_valid: bool,
        normalized_error: float,
        dt: Optional[float] = None,
    ) -> SteeringDecision:
        """Computes differential steering command and PWM outputs.

        Args:
            path_valid: Perception flag indicating if a path is actively visible.
            normalized_error: Lateral displacement in [-1.0, 1.0].
            dt: Time elapsed since last control step (seconds). Handled safely if dt <= 0 or None.
        """
        # Safety Gate: If path is missing, immediately stop all simulated drive proposals
        if not path_valid:
            self._prev_error = None
            return SteeringDecision(
                state="PATH_LOST",
                steering_command=0.0,
                left_pwm=0,
                right_pwm=0,
                normalized_error=0.0,
                proportional_term=0.0,
                derivative_term=0.0,
                reason="Path lost or not detected. Safety stop activated.",
            )

        # Sanitize error to [-1.0, 1.0]
        error = float(np.clip(normalized_error, -1.0, 1.0))

        # Proportional term
        p_term = self.config.kp * error

        # Derivative term with numerical protection against dt <= 0, inf, or NaN
        d_term = 0.0
        if self._prev_error is not None and dt is not None and dt > 1e-4 and np.isfinite(dt):
            d_term = self.config.kd * ((error - self._prev_error) / dt)

        self._prev_error = error

        # Deadband handling: tiny errors within tolerance do not oscillate steering
        if abs(error) <= self.config.deadband:
            state = "STRAIGHT"
            command = 0.0
            left_pwm = self.config.base_speed
            right_pwm = self.config.base_speed
            reason = f"Error ({error:+.3f}) within deadband (±{self.config.deadband}). Driving straight."
        else:
            command = p_term + d_term
            if error < 0:
                state = "LEFT"
                reason = f"Path on left (e={error:+.2f}). Steering left."
            else:
                state = "RIGHT"
                reason = f"Path on right (e={error:+.2f}). Steering right."

            # Differential drive PWM mixing:
            # When command > 0 (turn right): left wheel accelerates, right wheel decelerates
            # When command < 0 (turn left): left wheel decelerates, right wheel accelerates
            left_val = self.config.base_speed + command
            right_val = self.config.base_speed - command

            # Clamp output PWM proposals strictly within bounds [min_pwm, max_pwm]
            left_pwm = int(np.clip(round(left_val), self.config.min_pwm, self.config.max_pwm))
            right_pwm = int(np.clip(round(right_val), self.config.min_pwm, self.config.max_pwm))

        return SteeringDecision(
            state=state,
            steering_command=float(command),
            left_pwm=left_pwm,
            right_pwm=right_pwm,
            normalized_error=error,
            proportional_term=float(p_term),
            derivative_term=float(d_term),
            reason=reason,
        )
