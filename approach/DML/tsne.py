import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 2)
import json


def plot_embedding(X_org, y, title=None):
    X, _, Y, _ = train_test_split(X_org, y, test_size=0.5)
    X, Y = np.asarray(X), np.asarray(Y)
    # X = X[:10000]
    # Y = Y[:10000]
    # y_v = ['Vulnerable' if yi == 1 else 'Non-Vulnerable' for yi in Y]
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    print('Fitting TSNE!')
    X = tsne.fit_transform(X)
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    file_ = open("ll" + str(title) + '-tsne-features.json', 'w')
    if isinstance(X, np.ndarray):
        _x = X.tolist()
        _y = Y.tolist()
    else:
        _x = X
        _y = Y
    json.dump([_x, _y], file_)
    file_.close()
    fig, ax = plt.subplots()

    plt.ylim(0, 1.025)
    plt.xlim(0, 1.02)
    # plt.figure("zcy")
    ax.text(0.3, 1.05, '0:',
            color=plt.cm.Set1(2),
            fontdict={'weight': 'bold', 'size': 20})
    ax.text(0.35, 1.05, 'Non-Vul',
            color=plt.cm.Set1(2),
            fontdict={'weight': 'bold', 'size': 20})
    ax.text(0.55, 1.05, '1:',
            color=plt.cm.Set1(0),  # red
            fontdict={'weight': 'bold', 'size': 20})
    ax.text(0.6, 1.05, 'Vul',
            color=plt.cm.Set1(0),
            fontdict={'weight': 'bold', 'size': 20})
    # sns.scatterplot(X[:, 0], X[:, 1], hue=y_v, palette=['red', 'green'])
    for i in range(X.shape[0]):
        if Y[i] == 0:
            ax.text(X[i, 0], X[i, 1], '0',
                    color=plt.cm.Set1(2),  # hui color
                    fontdict={'size': 6})
            ax.set_facecolor('white')
        else:
            ax.text(X[i, 0], X[i, 1], '1',
                    color=plt.cm.Set1(0),  # red
                    fontdict={'weight': 'bold', 'size': 9})
            ax.set_facecolor('white')

    ax.spines['top'].set_linewidth('1.5')
    ax.spines['top'].set_linestyle("-")
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_linewidth('1.0')
    ax.spines['bottom'].set_linestyle("-")
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_linewidth('1.0')
    ax.spines['right'].set_linestyle("-")
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_linewidth('1.0')
    ax.spines['left'].set_linestyle("-")
    ax.spines['left'].set_color('black')

    plt.xticks([]), plt.yticks([])
    # plt.grid(b=False)
    plt.savefig("zcy" + str(title) + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
