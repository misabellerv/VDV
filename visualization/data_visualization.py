import matplotlib.pyplot as plt

class Visualization:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_imgs(images_list, grid_size, title):
        """
        Plots a grid of images.

        Parameters:
        images_list (list): List of images to plot.
        grid_size (int): Size of the grid (e.g., 3 for a 3x3 grid).
        title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 10))
        for idx, img in enumerate(images_list):
            if idx < grid_size * grid_size:
                plt.subplot(grid_size, grid_size, idx + 1)
                plt.suptitle(title)
                plt.axis('off')
                plt.imshow(img, cmap='gray')
        plt.show()
