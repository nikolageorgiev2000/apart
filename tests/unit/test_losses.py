import torch

from apart.training.rl_self_distill import sampled_token_policy_loss
from apart.training.subliminal import completion_cross_entropy


def test_rl_advantage_sign_and_detachment() -> None:
    student = torch.tensor([[-2.0, -3.0]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -4.0]], requires_grad=True)
    mask = torch.tensor([[True, True]])
    loss, advantage = sampled_token_policy_loss(student, teacher, mask)
    assert torch.allclose(advantage, torch.tensor([[1.0, -1.0]]))
    assert not advantage.requires_grad
    loss.backward()
    assert torch.allclose(student.grad, torch.tensor([[-0.5, 0.5]]))
    assert teacher.grad is None


def test_subliminal_loss_is_standard_masked_causal_ce() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, -100, 2]])
    loss = completion_cross_entropy(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[:, 1, :], torch.tensor([2]))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
