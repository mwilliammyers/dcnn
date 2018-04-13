import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas

for i, (file_path, lab) in enumerate(zip(sys.argv[1::2], sys.argv[2::2])):
    data = pandas.read_csv(file_path)

    sns.set_style('darkgrid')
    sns.set_context('paper')

    fig = plt.figure(i, (3, 2))
    ax = fig.add_subplot(111)

    b = sns.countplot(x=lab, data=data, order=data[lab].value_counts().index, ax=ax)
    b.set_xlabel('')
    fig.tight_layout()
    for p, label in zip(b.patches, data[lab].value_counts().index):
        b.annotate(
            f'{data[lab].value_counts()[label]/len(data)*100:.0f}%',
            (p.get_x() + p.get_width() / 2, p.get_y() + p.get_height() / 2),
            fontsize=10,
            color='white',
            ha='center',
        )

plt.show()
