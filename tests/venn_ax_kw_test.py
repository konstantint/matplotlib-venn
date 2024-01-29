"""
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from matplotlib_venn import *


def test_ax_kw():
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2)

    venn2(subsets={"10": 1, "01": 1, "11": 1}, set_labels=("A", "B"), ax=axes[0][0])
    venn2_circles((1, 2, 3), ax=axes[0][1])
    venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels=("A", "B", "C"), ax=axes[1][0])
    venn3_circles(
        {"001": 10, "100": 20, "010": 21, "110": 13, "011": 14}, ax=axes[1][1]
    )
