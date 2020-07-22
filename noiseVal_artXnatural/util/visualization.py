import matplotlib.pyplot as plt
import numpy as np

# We'll use matplotlib for graphics.
import matplotlib.patheffects as PathEffects
import matplotlib
# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
# sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def scatter(x, colors, labels, legend=True, chosen_palette='hls'):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette(chosen_palette, 10))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in np.unique(colors):
        sc = ax.scatter(x[colors == i,0], x[colors == i,1], lw=0, s=40,
                        c=np.expand_dims(palette[i],0), label=labels[i])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')
    if legend:
        ax.legend(bbox_to_anchor=(-0.18, 1.18),loc=2,fontsize='large')
    # We add the labels for each digit.
    txts = []
    # for i in range(10):
    #     # Position of each label.
    #     xtext, ytext = np.median(x[colors == i, :], axis=0)
    #     txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #     txt.set_path_effects([
    #         PathEffects.Stroke(linewidth=5, foreground="w"),
    #         PathEffects.Normal()])
    #     txts.append(txt)

    return f, ax, sc, txts