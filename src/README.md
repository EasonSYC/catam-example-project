# Information and Instructions for Running the Code

Program was coded in a `.ipynb` Jupyter Notebook (`notebook.ipynb`, a fresh version `notebook-fresh.ipynb` without outputs) using Python 3.15.0, utilising packages `numpy`, `pandas` and `matplotlib`.

In addition, a custom `macros.tex` is used for producing the LaTeX formulae in titles/labels in the diagrams, but is itself irrelevant to the code and the mathematics. If code is to be ran **without** `macros.tex`, the setup component of `matplotlib` for `plt.rcParams` would need the option `text.latex.preamble` to be removed, and certain labels/titles for the graphs adjusted, removing the custom LaTeX commands. This will only adjust the labels/title of the diagrams and nothing else.

There are two ways to run the code, after setting up Python and the relevant packages.

- **Preferred.** Install Jupyter Notebook and run `notebook.ipynb` or `notebook-fresh.ipynb`. This will include all intermediate outputs, and diagrams will be displayed as in-line outputs, as well as exported to separate files.
- *Alternative.* Run `notebook.py` using Python. In this case, some intermediate outputs of values will be omitted, but all data spreadsheets and diagrams will still be produced.

Note that data sheets and figures will be output to `../data/*.csv` and `../fig/*.png` respectively and the directories would need to be created in advance.
