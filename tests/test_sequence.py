# tests/test_sequence.py

import pytest

from vllm_mini.sequence import Sequence, SequenceStatus


def test_sequence_initial_state():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])

    assert sequence.seq_id == 0
    assert sequence.prompt == "Hello"
    assert sequence.prompt_token_ids == [15496]
    assert sequence.generated_token_ids == []
    assert sequence.token_ids == [15496]
    assert sequence.num_generated_tokens == 0
    assert sequence.status == SequenceStatus.WAITING
    assert sequence.finish_reason is None
    assert sequence.is_finished is False


def test_mark_running_updates_status():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])

    sequence.mark_running()

    assert sequence.status == SequenceStatus.RUNNING
    assert sequence.is_finished is False


def test_append_token_tracks_generated_and_full_token_ids():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])

    sequence.append_token(995)
    sequence.append_token(0)

    assert sequence.generated_token_ids == [995, 0]
    assert sequence.token_ids == [15496, 995, 0]
    assert sequence.num_generated_tokens == 2


def test_finish_length_sets_status_and_reason():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])

    sequence.finish(SequenceStatus.FINISHED_LENGTH)

    assert sequence.status == SequenceStatus.FINISHED_LENGTH
    assert sequence.finish_reason == "length"
    assert sequence.is_finished is True


def test_finish_eos_sets_status_and_reason():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])

    sequence.finish(SequenceStatus.FINISHED_EOS)

    assert sequence.status == SequenceStatus.FINISHED_EOS
    assert sequence.finish_reason == "eos"
    assert sequence.is_finished is True


def test_finish_rejects_non_finished_status():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])

    with pytest.raises(ValueError, match="finish status"):
        sequence.finish(SequenceStatus.RUNNING)


def test_finished_sequence_rejects_new_tokens():
    sequence = Sequence(seq_id=0, prompt="Hello", prompt_token_ids=[15496])
    sequence.finish(SequenceStatus.FINISHED_LENGTH)

    with pytest.raises(RuntimeError, match="finished sequence"):
        sequence.append_token(995)
