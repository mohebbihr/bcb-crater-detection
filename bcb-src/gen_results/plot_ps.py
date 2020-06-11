"""
============================
Bachelor's degrees by gender
============================

A graph of multiple time series which demonstrates extensive custom
styling of plot frame, tick lines and labels, and line graph properties.

Also demonstrates the custom placement of text labels along the right edge
as an alternative to a conventional legend.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
import pandas as pd

if __name__ == "__main__":

	epochs = [i for i in range(0,525,25)]
	# validation accuracy of progressive models
	acc_1 = [0.44, 0.46, 0.48, 0.48, 0.5, 0.49, 0.51, 0.53, 0.56, 0.54, 0.55, 0.58, 0.58, 0.59, 0.58, 0.63, 0.64, 0.69, 0.70, 0.72, 0.72] # 12 x 12 images
	acc_2 = [0.42, 0.54, 0.56, 0.58, 0.57, 0.56, 0.56, 0.53, 0.59, 0.64, 0.65, 0.66, 0.65, 0.71, 0.74, 0.73, 0.77, 0.80, 0.86, 0.85, 0.846] # 24 x 24 images
	acc_3 = [0.51, 0.55, 0.56, 0.59, 0.60, 0.58, 0.61, 0.63, 0.62, 0.66, 0.68, 0.70, 0.71, 0.73, 0.74, 0.78, 0.79, 0.84, 0.85, 0.89, 0.892] # 48 x 48 images
	acc_list = [acc_1, acc_2, acc_3]

	# These are the colors that will be used in the plot
	color_sequence = ['#00ff00', '#0000ff', '#ff0000']

	# You typically want your plot to be ~1.33x wider than tall. This plot
	# is a rare exception because of the number of lines being plotted on it.
	# Common sizes: (10, 7.5) and (12, 9)
	fig, ax = plt.subplots(1, 1, figsize=(14, 9))

	# Ensure that the axis ticks only show up on the bottom and left of the plot.
	# Ticks on the right and top of the plot are generally unnecessary.
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xlim(0, 500)

	# Provide tick lines across the plot to help your viewers trace along
	# the axis ticks. Make sure that the lines are light and small so they
	# don't obscure the primary data lines.
	plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)

	# Remove the tick marks; they are unnecessary with the tick lines we just
	# plotted.
	plt.tick_params(axis='both', which='both', bottom=False, top=False,
					labelbottom=True, left=False, right=False, labelleft=True)

	# Now that the plot is prepared, it's time to actually plot the data!
	# Note that I plotted the majors in order of the highest % in the final year.
	progressive_resizing = ['12 x 12 ','24 x 24 ','48 x 48 ']
	options = ['--', '+', 's']
	
	for i, step in enumerate(progressive_resizing):
		print("i: " + str(i))
		plt.plot(epochs, acc_list[i], options[i], lw=2.5, color=color_sequence[i])
		y_pos = acc_list[i][-1]
		
		plt.text(510, y_pos, step, fontsize=14, color=color_sequence[i])
	
	
	plt.show()
	plt.savefig('progressive_resizing_val_acc_2.png', bbox_inches='tight', dpi=400)

