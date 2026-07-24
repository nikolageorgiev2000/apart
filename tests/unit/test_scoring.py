from apart.models.logprobs import build_scoring_batch


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        content = "|".join(message["content"] for message in messages)
        return content + "|assistant:"

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [ord(character) % 31 + 1 for character in text]}


def test_response_mask_ignores_different_teacher_prefix_lengths() -> None:
    import torch

    tokenizer = FakeTokenizer()
    student = build_scoring_batch(
        tokenizer,
        ["hello"],
        [[7, 8, 2]],
        max_sequence_length=128,
        device="cpu",
    )
    teacher = build_scoring_batch(
        tokenizer,
        ["hello"],
        [[7, 8, 2]],
        system_prompts=["a much longer teacher instruction"],
        max_sequence_length=128,
        device="cpu",
    )
    assert torch.equal(
        student.input_ids[student.response_mask],
        teacher.input_ids[teacher.response_mask],
    )
    assert student.response_mask.sum() == teacher.response_mask.sum() == 3
