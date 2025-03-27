import torch


def save_model(
        model : torch.nn.Module,
        path : str
):
    torch.save(model.state_dict(), path)


def load_model(
        model : torch.nn.Module,
        path : str
):
    model.load_state_dict(torch.load(path))

