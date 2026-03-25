from collections import deque
from typing import Optional
import sympy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import seaborn as sns
from scipy.spatial import KDTree

from pyneutube.core.io.swc_parser import Neuron
from pyneutube.tracers.pyNeuTube.config import ConnectorType
from pyneutube.core.visualization import image_enhancer


DEBUG = False

def inspect_image(image: np.ndarray, debug=DEBUG):
    if not debug: return

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(image[image.shape[0]//2])
    axs[1].imshow(image.mean(axis=0))
    axs[2].imshow(image.max(axis=0))
    plt.show()

    return

def inspect_seed(image: np.ndarray, seed, debug=DEBUG):
    if not debug: return

    center = seed.coord[::-1]

    slices = []
    for dim, (c, half) in enumerate(zip(center, (11,11,11))):
        start = max(0, c - half)
        stop  = min(image.shape[dim], c + half + 1)
        slices.append(slice(start, stop))

    cropped = image[tuple(slices)]

    fig, axs = plt.subplots(1,3)
    axs[0].imshow(cropped[cropped.shape[0]//2])
    axs[1].imshow(cropped.mean(axis=0))
    axs[2].imshow(cropped.max(axis=0))
    plt.show()


def inspect_seg(image: np.ndarray, seg, filter_args=None, debug=DEBUG):
    if not debug:
        return

    if filter_args is not None:
        filter_coords_3d, _, weights_3d = filter_args
        filter_coords_3d -= np.mean(filter_coords_3d, axis=0)

    # 1) Don’t lose the sub‐pixel center by casting to int yet!
    true_center = seg.center_coord[::-1]  # (z,y,x) → (row, col) floats

    padding = 11
    img_pad = np.pad(image, pad_width=padding, mode='constant', constant_values=0)

    # 2) Compute integer crop bounds around the true center
    crop_slices = []
    for dim, c in enumerate(true_center):
        c_int = int(np.floor(c)) + padding
        start = max(0, c_int - padding)
        stop  = min(img_pad.shape[dim], c_int + padding + 1)
        crop_slices.append(slice(start, stop))

    cropped = img_pad[tuple(crop_slices)]

    # 3) Now compute where your TRUE center sits in the cropped block:
    #    center_in_crop = true_center + padding - slice.start
    center_in_crop = []
    for dim, c in enumerate(true_center):
        c_int = int(np.floor(c)) + padding
        center_in_crop.append(c_int - crop_slices[dim].start)
    # center_in_crop is now [row_center, col_center, slice_center]

    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        # get the 2D slice of interest
        slc = np.take(cropped, cropped.shape[i] // 2, axis=i)
        if i in (1, 2):
            slc = slc.T

        ax = axs[i, 2]
        ax.imshow(slc, cmap='gray')

        # projected center in this 2D view:
        # axes_map[i] tells you which two dims remain
        axes_map = [(1, 2), (0, 2), (0, 1)]
        ax1, ax2 = axes_map[i]
        cx = center_in_crop[ax2]  # x‐axis in image
        cy = center_in_crop[ax1]  # y‐axis in image

        # draw your direction vector (half‐length = padding)
        dx = (padding / 2 * seg.dir_v[ax2])
        dy = (padding / 2 * seg.dir_v[ax1])


        length = 2 * np.hypot(dx, dy)            # full length along segment
        height1 = 2 * seg.radius                     # mimics lw=radius*2
        height2 = 2 * seg.radius * seg.scale             # mimics lw=radius*scale*2
        angle  = np.degrees(np.arctan2(dy, dx))  # segment angle in degrees

        def draw_rect(length, height, color, alpha):
            # lower‐left corner before rotation:
            ll_x = cx - length/2
            ll_y = cy - height/2

            rect = Rectangle((ll_x, ll_y),
                            width=length,
                            height=height,
                            facecolor=color,
                            alpha=alpha,
                            edgecolor=None)

            # rotate around the center (cx,cy)
            trans = (Affine2D()
                    .rotate_deg_around(cx, cy, angle)
                    + ax.transData)
            rect.set_transform(trans)
            ax.add_patch(rect)

        draw_rect(length, height1, color='green', alpha=0.3)
        draw_rect(length, height2, color='red', alpha=0.3)

        if filter_args is not None:
            # scatter your filter points in the same coordinate frame:
            # filter_coords_3d are in segment‐local coords, so translate by
            # (center_in_crop - true_center_fractional_offset)…
            # but if filter_coords_3d was generated around (0,0,0), just
            # add cx,cy appropriately
            wmax = np.max(np.abs(weights_3d))
            pts_x = filter_coords_3d[:,ax2] + cx
            pts_y = filter_coords_3d[:,ax1] + cy

            ax.scatter(pts_x, pts_y,
                    c=weights_3d,
                    cmap='coolwarm',
                    vmin=-wmax,
                    vmax=+wmax,
                    edgecolor=None,
                    marker='.',
                    s=1,
                    alpha=0.3)
            

    plt.tight_layout()
    plt.show()


def inspect_chain(image: np.ndarray, chain, debug=DEBUG):
    if not debug: return

    for seg in chain:
        inspect_seg(image, seg, debug=debug)


def inspect_chain_graph(image: np.ndarray, edge, chains, conn, debug=DEBUG):
    if not debug: return
    
    plt.figure(figsize=(10,10))
    plt.imshow(image.max(axis=0), cmap='gray')
    u,v = edge
    chain_u_coords = chains[u].to_coords()
    chain_v_coords = chains[v].to_coords()
    plt.plot(chain_u_coords[:,0], chain_u_coords[:,1], 'red')
    plt.plot(chain_v_coords[:,0], chain_v_coords[:,1], 'green')
    x0, y0, _ = chain_u_coords[-conn.info[0]]
    if conn.mode.value == ConnectorType.NEUROCOMP_CONN_LINK.value:
        x1, y1, _ = chain_v_coords[-conn.info[1]]
    else:
        x1, y1, _ = chain_v_coords[conn.info[1]]
    plt.scatter(x0,y0, edgecolor='black',color='red')
    plt.scatter(x1,y1, edgecolor='black',color='green')
    plt.scatter(conn.pos[0],conn.pos[1], edgecolor='black',color='blue')
    
    plt.title(f'{conn.mode}'+'\n'+str(conn.info))
    plt.show()


def plot_neuron(image: Optional[np.ndarray], neuron: Neuron, debug=DEBUG):
    if not debug: return

    plt.figure(figsize=(10,10))
    if image is not None:
        plt.imshow(image.max(axis=0), cmap='gray',origin='lower')
    for node in neuron.swc:
        pid = node[6]
        pidx = neuron.nidHash.get(pid)
        if pidx is not None:
            pnode = neuron.swc[pidx]
            plt.plot([node[2], pnode[2]], [node[3], pnode[3]], 'dodgerblue', 
                    #  marker='o',
                     linewidth=1,)
    plt.show()


def compare_neurons(image: np.ndarray, neuron: Neuron, ref_neuron: Neuron, debug=DEBUG, figname=None, enhance=True):
    if not debug: return

    coords = neuron.coords
    ref_coords = ref_neuron.coords

    kdtree = KDTree(ref_coords)
    _, neighbors_list = kdtree.query(coords, 20)

    def dist_point_2_nearest_lines(point, neighbors):

        def point_to_line_distance(point, line_start, line_end):
            line_vec = line_end - line_start
            point_vec = point - line_start
            line_len2 = np.dot(line_vec, line_vec)
            if line_len2 == 0:
                return np.linalg.norm(point_vec)
            t = np.clip(np.dot(point_vec, line_vec) / line_len2, 0, 1)
            proj = line_start + t * line_vec
            return np.linalg.norm(point - proj)

        min_dist = np.inf
        for nb in neighbors:
            ref_node = ref_neuron.swc[nb]
            ref_node_pidx = ref_neuron.nidHash.get(ref_node[6])
            if ref_node_pidx is not None and ref_node_pidx in neighbors:
                # dist = float(sympy.Line3D(ref_coords[nb], ref_coords[ref_node_pidx]).distance(sympy.Point3D(point)))
                dist = point_to_line_distance(point, ref_coords[nb], ref_coords[ref_node_pidx])
                min_dist = min(min_dist, dist)
        return min_dist


    cmap = sns.color_palette("RdYlGn_r", n_colors=256)
    max_dist = 2.0

    plt.figure(figsize=(10,10))
    img2d = image.max(axis=0)
    if enhance:
        img2d = image_enhancer.soft_standardize(img2d, k_sigma=1.5)
    plt.imshow(img2d, cmap='gray')

    stack = deque()
    for soma in neuron.get_soma(allow_multiple=True):
        soma_idx = neuron.nidHash[soma[0]]
        stack.extend(neuron.indexChildren[soma_idx])

        while stack:
            cur_idx = stack.pop()
            cur_node = neuron.swc[cur_idx]

            parent_idx = neuron.nidHash[cur_node[6]]

            min_dist = dist_point_2_nearest_lines(coords[cur_idx], neighbors_list[cur_idx])

            normalized_dist = int(np.clip(min_dist/max_dist, 0, 1) * 255)
            color = cmap[normalized_dist]

            plt.plot([coords[cur_idx][0], coords[parent_idx][0]],
                    [coords[cur_idx][1], coords[parent_idx][1]],
                    color=color,)

            stack.extend(neuron.indexChildren[cur_idx])

    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[2])

    if figname is not None:
        plt.tight_layout()
        plt.savefig(figname, dpi=300); plt.close()
    else:
        plt.show()
    
