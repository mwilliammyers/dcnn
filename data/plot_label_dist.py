import matplotlib.pyplot as plt
import seaborn as sns
import pandas


data = pandas.read_csv('twitter_airlines.csv')
lab = 'airline_sentiment'

sns.set_style('darkgrid')
sns.set_context('paper')

fig = plt.figure(1, (3,2))
ax = fig.add_subplot(111)

b = sns.countplot(x=lab, data=data, order=data[lab].value_counts().index, ax=ax)
b.set_xlabel('')
fig.tight_layout()
for p, label in zip(b.patches, data[lab].value_counts().index):
    b.annotate(
        f'{data[lab].value_counts()[label]/len(data)*100:.0f}%', 
        (p.get_x()+p.get_width()/2, p.get_y()+p.get_height()/2),
        fontsize=10,
        color='white',
        ha='center',
    )

plt.show()