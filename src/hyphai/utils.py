import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from PIL import Image
import datetime
import matplotlib.patches as mpatches
# Cloud types
CLOUD_TYPE_12 = {'0': 'No Cloud',
                 '1': 'Very low clouds',
                 '2': 'Low clouds',
                 '3': 'Mid-level clouds',
                 '4': 'High opaque clouds',
                 '5': 'Very high opaque clouds',
                 '6': 'Fractional clouds',
                 '7': 'High semi transparent thin clouds',
                 '8': 'High semi transparent meanly thick clouds',
                 '9': 'High semi transparent thick clouds',
                 '10': 'High semi transparent above low/medium clouds',
                 '11': 'High semi transparent above snow/ice'
                 }
# Color map for the 12 cloud types
CT_12_COLOR_MAP = [[0.39215686, 0.39215686, 0.39215686],
                   [1.0, 0.39215686, 0.0],
                   [1.0, 0.70588235, 0.0],
                   [0.94117647, 0.94117647, 0.0],
                   [0.84313725, 0.84313725, 0.58823529],
                   [0.90196078, 0.90196078, 0.90196078],
                   [0.78431373, 0.0, 0.78431373],
                   [0.0, 0.31372549, 0.84313725],
                   [0.0, 0.70588235, 0.90196078],
                   [0.0, 0.94117647, 0.94117647],
                   [0.35294118, 0.78431373, 0.62745098],
                   [1, 0.6, 1]]

def load_data(name: str, t: int, bounds: tuple = (280, 1740, 256, 256)) -> np.ndarray:
    """Get satellite data at time t

    Args:
        name (str): datetime of the initial state
        t (int): time step
        bounds (tuple): starting point coordinates of the bounding box and the shape of the image
            (x_min, y_min, *IN_SHAPE)

    Returns:
        np.ndarray: satellite image at time t
    """
    x_min, y_min, x_max, y_max = bounds
    x_max += x_min
    y_max += y_min

    # Convert the string to a datetime object
    dtime = datetime.datetime.strptime(name, '%Y%m%d%H%M')

    try:
        # Get the file name of the observations at time t
        # Convert the datetime object to a string
        file = (dtime + t * datetime.timedelta(minutes=15)).strftime('%Y%m%d%H%M')
        # Load the file
        print(f"Loading data at {dtime + t * datetime.timedelta(minutes=15)} ... ",end="")
        data = np.load(f"./data/{file}.npz")['arr_0'][x_min:x_max, y_min:y_max]
        print(f"done")
        # Merge the labels(0:'no data', 1:'cloud free land', 2:'cloud free sea',
        # 3:'snow over land' and 4:'sea ice') into one label (0:'no cloud')
        data[data < 5] = 0
        data[data != 0] -= 4
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file} does not exist", "red")
    
    return data

def one_hot(x: np.ndarray, num_classes=12, axis=-1) -> np.ndarray:
    r"""One-hot encoding

    Args:
        x (np.ndarray): Input array
        num_classes (int, optional): Number of classes. Defaults to 12.
        axis (int, optional): The class axis is placed last by defaults.

    Raises:
        NotImplementedError: Only axis=-1 or axis=0 cases are implemented

    Returns:
        np.ndarray: A binary matrix representation of the input
    """
    if axis == -1:
        cat_x = np.empty(x.shape + (num_classes,))
        for i in range(num_classes):
            cat_x[..., i] = 1 * (x == i)
    elif axis == 0:
        cat_x = np.empty((num_classes,) + x.shape)
        for i in range(num_classes):
            cat_x[i, ...] = 1 * (x == i)
    else:
        raise NotImplementedError(
            "Only axis=-1 or axis=0 cases are implemented")
    return cat_x.astype(int)


def make_gif(inputs, outputs, ground_truth, save_to) -> None:
    """Make an animated plot of the model's predictions over multiple time steps

    Args:
        inputs (np.ndarray): Context observations
        outputs (np.ndarray): Model's predictions
        ground_truth (np.ndarray): Ground truth
        save_to (str): Path to save the animation
    """
    
    print("Making animation ...")
    y_list = 2*[inputs]
    titles = 2*["Observations"]
    context_size = inputs.shape[0]
    max_lead_time = outputs.shape[0]
    cmap = colors.ListedColormap(CT_12_COLOR_MAP)
    norm = colors.BoundaryNorm([i for i in range(cmap.N + 1)], cmap.N)
    patches = [mpatches.Patch(color=CT_12_COLOR_MAP[i], label=CLOUD_TYPE_12[str(i)]) for i in range(cmap.N)]
    images = []
    for t in range(-context_size, max_lead_time):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs = axs.flatten()
        if t == 0:
            y_list = [ground_truth, outputs]
            titles = ["Ground truth","Predictions"]
        for i in range(len(y_list)):
            ind = t if t >= 0 else t + context_size
            ax = axs[i].imshow(y_list[i][ind], cmap=cmap, norm=norm, interpolation='none', 
                        origin='lower')
            axs[i].set_title(f"{titles[i]} : {'+' if t >= 0 else ''}{(t + 1) * 15} min")
            axs[i].grid(False)
            axs[i].tick_params(left = False, right = False, labelleft = False, 
                               labelbottom = False, bottom = False)
        # color legend        
        cbar_ax = fig.add_axes([0.35, -0.05, 0.3, 0.05])
        fig.colorbar(ax, cax=cbar_ax, orientation='horizontal',pad=0.03).set_label('Cloud Type (CT)')
        plt.legend(handles=patches, loc='lower center',ncol=3, 
                   bbox_to_anchor=(0.12, -6.5, 0.8, 0.4),fontsize = 'small')

        # Reduce the white space between the plots
        plt.subplots_adjust(top=0.95, bottom=0.05,
                            right=0.98, left=0.05, hspace=0.05, wspace=0.05)            
        plt.margins(0.015, tight=True)
        tmp = "tmp.png"
        plt.savefig(tmp, dpi=300, bbox_inches='tight')
        if t in [0, max_lead_time-1]:
            plt.show()
        plt.close()
        images += [Image.open(tmp)]
        os.remove(tmp)
    print("Done")
    # Save the animation
    images[0].save(save_to, save_all=True, append_images=images[1:], optimize=False, 
                   duration=200, loop=0)
    print(f"Saved to {save_to}")

def plot_velocity(v_field, save_to="velocity_field"):
    """Plot the estimated velocity field

    Args:
        v_field (np.ndarray): Estimated velocity field of shape (2, 256, 256)
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.grid(False)
    #plt.axis('off')
    arrowsize = 1
    step = 10
    density = 2.5
    
    extent = [-5.5, 10.5, 39.5, 53.5]
    lons = np.linspace(extent[0], extent[1], v_field.shape[-2])
    lats = np.linspace(extent[2], extent[3], v_field.shape[-1])
    # Plot quiver
    u = 4 * 4.5 * v_field[0, ::step, ::step]
    v = 4 * 4.5 * v_field[1, ::step, ::step]
    v_max = 300
    magnitude = (u**2 + v**2)**0.5
    magnitude[magnitude>v_max] = v_max
    stream = plt.streamplot(lons[::step], lats[::step], u, v, 
                        density=density, color=magnitude, arrowsize=arrowsize,linewidth=1)#, cmap='plasma')

    cbar = plt.colorbar(stream.lines, ax=ax, orientation='vertical', fraction=0.046, pad=0.1, extend='max')
    cbar.set_label('Velocity ($km$ $h^{-1}$)')
    cbar.set_ticklabels([f'{val}' if val < v_max else f'$\geq${v_max}' for val in cbar.get_ticks()])    
    plt.title("Estimated velocity field")   
    ext = 'pdf'
    plt.savefig(save_to + f'.{ext}', dpi=300, format=ext, bbox_inches='tight') 
    plt.show()
    print(f"Figure saved to {save_to}.{ext}")  