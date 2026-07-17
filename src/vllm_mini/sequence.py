from dataclasses import dataclass, field
from enum import Enum


class SequenceStatus(Enum):
    """The status of a sequence."""

    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_LENGTH = "finished_length"
    FINISHED_EOS = "finished_eos"

_FINISHED_STATUSES = {
    SequenceStatus.FINISHED_LENGTH,
    SequenceStatus.FINISHED_EOS,
}

_FINISHED_REASONS = {
    SequenceStatus.FINISHED_LENGTH: "length",
    SequenceStatus.FINISHED_EOS: "eos",
}


@dataclass
class Sequence:
    """A sequence of tokens to be generated."""

    seq_id: int
    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    finish_reason: str | None = None

    @property
    def token_ids(self) -> list[int]:
        """Get the token ids of the sequence, including the prompt and generated tokens."""
        return self.prompt_token_ids + self.generated_token_ids

    @property
    def num_generated_tokens(self) -> int:
        """Get the number of generated tokens."""
        return len(self.generated_token_ids)

    @property
    def is_finished(self) -> bool:
        """Check if the sequence is finished."""
        return self.status in _FINISHED_STATUSES

    def mark_running(self) -> None:
        """Mark the sequence as running."""
        if self.is_finished:
            raise RuntimeError("cannot mark a finished sequence as running")
        self.status = SequenceStatus.RUNNING

    def append_token(self, token_id: int) -> None:
        """Append a generated token to the sequence."""
        if self.is_finished:
            raise RuntimeError("cannot append token to a finished sequence")
        self.generated_token_ids.append(token_id)

    def finish(self, status: SequenceStatus) -> None:
        """Mark the sequence as finished with the given status."""
        if status not in _FINISHED_STATUSES:
            raise ValueError("finish status must be FINISHED_LENGTH or FINISHED_EOS")
        self.status = status
        self.finish_reason = _FINISHED_REASONS[status]
