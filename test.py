import numpy as np
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

import matplotlib.pylab as pl
from src.utils import Hausdorff_dist_two_convex_hulls


np.random.seed(0)

####################################################################################### 
# Parameters
M = 1000 # num of samples
dim_input = 2 # dimension of input space
dim_outpu = 2 # dimension of output space
# ----------------------

# Function to be used
def f(x):
    # x - (2, M) with M number of samples
    A = np.array([[2.,3.5],[-1.5,1.]])
    return A @ x

"""
    If X ⊂ Rn and F is an affine transformation of Rn, then F maps Conv (X) onto Conv(F[X]).
"""
input_points = np.random.normal(0., 1., (100, dim_input))
input_set = ConvexHull(input_points)
output_set = ConvexHull(f(input_points.T).T)

#######################################################################################
def eps_RAND_UP(input_set, M, distribution):
    """
        Function that deploys epsilon-Rand-Up algorithm without the epsilon-padding (theorem 2)
        input_set: list of scipy.spatial.convexHull type whose union is X
        M: number of samples
        eps: desired padding
        distribution: sampling distribution
    """
    i = 1
    ys = np.zeros((dim_input, M))

    if distribution["type"] == "generic":
        while i < M:
            sample = np.random.normal(distribution["mean"], distribution["std"], (1, dim_input))
            for set in input_set:
                verts_bef = set.vertices
                new_set = ConvexHull(set.points, incremental=True)
                new_set.add_points(sample, restart=False)
                vert_after = new_set.vertices

                if np.array_equal(set.points[verts_bef, :], new_set.points[vert_after, :]):
                    ys[:, i-1] = f(sample.T).reshape(-1,)
                    i = i+1
                    break

    elif distribution["type"] == "border":
        num_faces = 0
        for set in input_set:
            num_faces += set.equations.shape[0]

        num_samples_per_face = int(np.floor(M/num_faces))

        for set in input_set:
            for idx in range(len(set.vertices)):
                step = 1/num_samples_per_face
                for _ in range(num_samples_per_face):
                    alpha = _*step
                    sample = (1-alpha)*set.points[set.vertices[idx], :] + alpha*set.points[set.vertices[(idx+1)%len(set.vertices)], :]
                    ys[:, i-1] = f(sample.reshape(-1,1)).reshape(-1,)
                    i = i+1

        ys = ys[:, :i-1]

    output_set = ConvexHull(ys.T)

    return output_set

distr = {}
distr["type"] = "border"
distr["mean"] = 0.
distr["std"] = 1.

y = eps_RAND_UP([input_set], M, distr)

hausdorff_distance = Hausdorff_dist_two_convex_hulls(y, output_set)
print("The Hausdorff distance between the true output set and the estimated output set is: "+str(hausdorff_distance) + " for the sampling method " + str(distr["type"]))
convex_hull_plot_2d(y)
convex_hull_plot_2d(output_set)
plt.show()




