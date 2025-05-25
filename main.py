import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from dataset import (
    gaussian_mixture_pdf,
    make_gaussian_mixture
)
from model import DDPM


def main():
    # Generate dataset
    X, y, pdf_values = make_gaussian_mixture(
        n_samples=10_000,
        n_features=2,
        means=np.array([[-5, -5], [0, 5], [5, -2]]),
        covariances=np.array([[[2, 0], [0, 2]],
                              [[1, 0], [0, 3]],
                              [[3, 1], [1, 1]]]),
        weights=np.array([0.3, 0.4, 0.3]),
        return_pdf=True,
        random_state=42
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = DataLoader(dataset=TensorDataset(torch.from_numpy(X).to(torch.float32)),
                            batch_size=128,
                            shuffle=True)

    model = DDPM(input_dim=2,
                 hidden_dim=32,
                 T=200,
                 beta_start=1e-4,
                 beta_end=0.02).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # train
    epochs = 5_000
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(device)
            loss = model(batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} - Loss: {total_loss:.4f}")

    test_samples = model.sample(num_samples=1000, device=device).cpu().numpy()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(test_samples[:, 0], test_samples[:, 1], alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.title(f'Generated samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    # # Visualize the data
    # plt.figure(figsize=(12, 5))
    #
    # # Plot samples
    # plt.subplot(1, 2, 1)
    # scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
    # plt.colorbar(scatter, label='Class')
    # plt.title('Generated samples')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    #
    # # Plot PDF contour
    # plt.subplot(1, 2, 2)
    # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
    #                      np.linspace(y_min, y_max, 100))
    # grid_points = np.c_[xx.ravel(), yy.ravel()]
    #
    # # Calculate PDF values on the grid
    # grid_pdf = gaussian_mixture_pdf(
    #     grid_points,
    #     means=np.array([[-5, -5], [0, 5], [5, -2]]),
    #     covariances=np.array([[[2, 0], [0, 2]],
    #                           [[1, 0], [0, 3]],
    #                           [[3, 1], [1, 1]]]),
    #     weights=np.array([0.3, 0.4, 0.3])
    # )
    # grid_pdf = grid_pdf.reshape(xx.shape)
    #
    # plt.contourf(xx, yy, grid_pdf, cmap='viridis', alpha=0.7)
    # plt.colorbar(label='PDF value')
    # plt.scatter(X[:, 0], X[:, 1], c='white', alpha=0.1, s=5)
    # plt.title('Probability Density Function')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
