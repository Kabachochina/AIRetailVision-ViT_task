import torch


class PGDAttack:
    def __init__(
            self,
            model : torch.nn.Module,
            epsilon : float,
            steps : int,
            step_size : float,
            random_start : bool = False
    ):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size
        self.rand = random_start
        self.loss_func = self._cross_entropy

    def _cross_entropy(self, x, y):
        logits = self.model(x).logits
        return torch.nn.functional.cross_entropy(logits, y)

    def perturb(
            self,
            images : torch.Tensor,
            labels : torch.Tensor
    ) -> torch.Tensor:
        """
        Do pgd attack on images.
        Turn eval mode for model
        :param images: images for attack
        :param labels: labels of images
        :return: attacked images
        """

        self.model.eval()
        if self.rand:
            x = images + torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            x = torch.clamp(x, 0, 1)
        else:
            x = images.clone()

        # Обеспечиваем вычисление градиентов для x
        for _ in range(self.steps):
            x.requires_grad_()
            self.model.zero_grad()

            loss = self.loss_func(x, labels)
            loss.backward()
            # Обновляем x в направлении знака градиента

            with torch.no_grad():
                grad_sign = x.grad.sign()
                x = x + self.step_size * grad_sign
                x = torch.max(torch.min(x, images + self.epsilon), images - self.epsilon)
                x = torch.clamp(x, 0, 1)
            # Обнуляем градиенты перед следующим шагом
            x = x.detach()

        return x