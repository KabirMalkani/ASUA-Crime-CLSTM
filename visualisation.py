import numpy as np
from scipy import sparse
import seaborn as sns
import matplotlib.pyplot as plt

# # HEAT MAP

truth = np.load('test_truth.npy')
pred = np.load('test_predict.npy')

for i in range(5):
    sns.set()
    ax = sns.heatmap(truth[15, :, :, i])
    plt.show()
    sns.set()
    ax = sns.heatmap(pred[15, :, :, i])
    plt.show()


# # PLOT SINGLE

# sp_mat = sparse.dok_matrix(mat[4, :, :]) 

# x = []
# y = []
# marker_size = []
# for loc, count in sp_mat.items():
# 	x.append(loc[0])
# 	y.append(loc[1])
# 	marker_size.append(10*int(count))

	
# fig = go.Figure(data=[go.Scatter(
#     x=x, y=y,
#     mode='markers',
#     marker_size=marker_size)
# ])

# fig.show()


# # # PLOT OVER TIME
# # Create figure
# fig = go.Figure()

# # Add traces, one for each slider step
# for step in range(mat.shape[0]):
# 	sp_mat = sparse.dok_matrix(mat[step, :, :]) 
# 	x = []
# 	y = []
# 	marker_size = []
# 	for loc, count in sp_mat.items():
# 		x.append(loc[0])
# 		y.append(loc[1])
# 		marker_size.append(10*int(count))

# 	fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker_size=marker_size))
# # # Make 10th trace visible
# # fig.data[10].visible = True

# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#     step = dict(
#         method="restyle",
#         args=["visible", [False] * len(fig.data)],
#     )
#     step["args"][1][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)

# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "Hour: "},
#     pad={"t": 50},
#     steps=steps
# )]

# fig.update_layout(
#     sliders=sliders
# )

# fig.update_xaxes(range=[0, 32])
# fig.update_yaxes(range=[0, 32])
# fig.show()
