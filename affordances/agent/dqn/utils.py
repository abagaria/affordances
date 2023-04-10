import torch
from pfrl.utils import batch_states


def batch_experiences(experiences, device, phi, gamma, batch_states=batch_states):
    """Takes a batch of k experiences each of which contains j

    consecutive transitions and vectorizes them, where j is between 1 and n.

    Args:
        experiences: list of experiences. Each experience is a list
            containing between 1 and n dicts containing
              - state (object): State
              - action (object): Action
              - reward (float): Reward
              - is_state_terminal (bool): True iff next state is terminal
              - next_state (object): Next state
        device : GPU or CPU the tensor should be placed on
        phi : Preprocessing function
        gamma: discount factor
        batch_states: function that converts a list to a batch
    Returns:
        dict of batched transitions
    """

    batch_exp = {
        "state": batch_states([elem[0]["state"] for elem in experiences], device, phi),
        "action": torch.as_tensor(
            [elem[0]["action"] for elem in experiences], device=device
        ),
        "reward": torch.as_tensor(
            [
                sum((gamma ** i) * exp[i]["reward"] for i in range(len(exp)))
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "next_state": batch_states(
            [elem[-1]["next_state"] for elem in experiences], device, phi
        ),
        "is_state_terminal": torch.as_tensor(
            [
                any(transition["is_state_terminal"] for transition in exp)
                for exp in experiences
            ],
            dtype=torch.float32,
            device=device,
        ),
        "discount": torch.as_tensor(
            [(gamma ** len(elem)) for elem in experiences],
            dtype=torch.float32,
            device=device,
        ),
        "extra_info": [elem[-1]["extra_info"] for elem in experiences]
    }
    if all(elem[-1]["next_action"] is not None for elem in experiences):
        batch_exp["next_action"] = torch.as_tensor(
            [elem[-1]["next_action"] for elem in experiences], device=device
        )
    return batch_exp