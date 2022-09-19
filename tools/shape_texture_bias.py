from modelvshuman import Plot, Evaluate
from modelvshuman import constants as c
from modelvshuman.plotting.colors import *
from modelvshuman.plotting.decision_makers import DecisionMaker

def plotting_definition_template(df):
    """Decision makers to compare a few models with human observers.

    This exemplary definition can be adapted for the
    desired purpose, e.g. by adding more/different models.

    Note that models will need to be evaluated first, before
    their data can be plotted.

    For each model, define:
    - a color using rgb(42, 42, 42)
    - a plotting symbol by setting marker;
      a list of markers can be found here:
      https://matplotlib.org/3.1.0/api/markers_api.html
    """

    decision_makers = []

    decision_makers.append(DecisionMaker(name_pattern="tv_resnet50",
                           color=rgb(65, 90, 140), marker="o", df=df,
                           plotting_name="ResNet-50"))
    decision_makers.append(DecisionMaker(name_pattern="subject-*",
                           color=rgb(165, 30, 55), marker="D", df=df,
                           plotting_name="humans"))
    return decision_makers

def run_evaluation():
    models = ["tv_resnet50"]
    datasets = ["cue-conflict"] # or e.g. ["cue-conflict", "uniform-noise"]
    params = {"batch_size": 64, "print_predictions": True, "num_workers": 20}
    Evaluate()(models, datasets, **params)


def run_plotting():
    plot_types = ["shape-bias"] # or e.g. ["accuracy", "shape-bias"]
    plotting_def = plotting_definition_template
    figure_dirname = "example-figures/"
    Plot(plot_types = plot_types, plotting_definition = plotting_def,
         figure_directory_name = figure_dirname, crop_PDFs=False)

    # In examples/plotting_definition.py, you can edit
    # plotting_definition_template as desired: this will let
    # the toolbox know which models to plot, and which colours to use etc.


if __name__ == "__main__":
    # 1. evaluate models on out-of-distribution datasets
    run_evaluation()
    # 2. plot the evaluation results
    run_plotting()
