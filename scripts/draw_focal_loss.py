from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

# Values
gammas = [0, 2.0, 5.0, 10.0]
line_styles = ['solid', 'dashed', 'dashdot', 'dotted']
probabilities = np.linspace(start=0.001, stop=1.0, num=1000)
focal_loss = np.vectorize(lambda p, gamma: - np.power(1 - p, gamma) * np.log(p))

# Draw graph
rc('font', **{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

_, ax = plt.subplots(figsize=(6, 3))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

for gamma, line_style in zip(gammas, line_styles):
    handle = plt.plot(probabilities, focal_loss(probabilities, gamma), label=f'$\gamma$ = {gamma}', linestyle=line_style)

plt.xlim((-0.01, 1.01))
plt.ylim((-0.01, 5))
plt.xlabel('probability of ground truth class')
plt.ylabel('loss')
plt.legend()

plt.tight_layout()
plt.savefig('focal_loss.pdf')
plt.show()
