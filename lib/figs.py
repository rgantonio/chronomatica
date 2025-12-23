"""
    Python library for figure generation
    - In this library, we will define classes and functions to generate figures
"""

import matplotlib.pyplot as plt


class figStyleParams:
    """
    Class for the line style parameters
    """

    def __init__(self):

        # --------------------------------------------
        # Style configurations
        # --------------------------------------------
        self.style = {
            # Font sizes
            "title_size": 13,
            "label_size": 12,
            "tick_size": 11,
            "legend_size": 11,
            # Line & marker
            "line_width": 2.2,
            "marker_size": 6,
            # Grid
            "grid_alpha": 0.6,
            "grid_linewidth": 0.6,
            # Figure
            "figsize": (7, 4.2),
            # Colors (color-blind friendly)
            "colors": [
                "#1f77b4",  # blue
                "#ff7f0e",  # orange
                "#2ca02c",  # green
                "#d62728",  # red
                "#9467bd",  # purple
                "#8c564b",  # brown
                "#e377c2",  # pink
                "#7f7f7f",  # gray
            ],
            "markers": [
                "o",  # circle
                "s",  # square
                "D",  # diamond
                "^",  # triangle up
                "v",  # triangle down
                "<",  # triangle left
                ">",  # triangle right
                "x",  # x
            ],
        }


class multiLinePlot(figStyleParams):
    """
    Class for multi-line plot
    Args:
        x_data: list of x data points
        y_data_list: list of lists of y data points
        legend_list: list of legend labels
        x_label: label for x axis
        y_label: label for y axis
        title: title of the figure
    """

    def __init__(self, x_data, y_data_list, legend_list, x_label, y_label, title):
        super().__init__()
        self.x_data = x_data
        self.y_data_list = y_data_list
        self.legend_list = legend_list
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

    def plot_fig(self):
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=self.style["figsize"])

        # Plot each line
        for i, (y_data, label) in enumerate(zip(self.y_data_list, self.legend_list)):
            self.ax.plot(
                self.x_data,
                y_data,
                label=label,
                color=self.style["colors"][i % len(self.style["colors"])],
                marker=self.style["markers"][i % len(self.style["markers"])],
                linewidth=self.style["line_width"],
                markersize=self.style["marker_size"],
            )

        # Set labels and title
        self.ax.set_xlabel(self.x_label, fontsize=self.style["label_size"])
        self.ax.set_ylabel(self.y_label, fontsize=self.style["label_size"])
        self.ax.set_title(self.title, fontsize=self.style["title_size"])

        # Set tick parameters
        self.ax.tick_params(
            axis="both", which="major", labelsize=self.style["tick_size"]
        )

        # Add grid
        self.ax.grid(
            True,
            which="both",
            linestyle="--",
            alpha=self.style["grid_alpha"],
            linewidth=self.style["grid_linewidth"],
        )

        # Add legend
        self.ax.legend(fontsize=self.style["legend_size"], frameon=False)

        # Plot
        plt.tight_layout()
        plt.show()
