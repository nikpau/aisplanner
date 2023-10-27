import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

import numpy as np
from descriptives import ShipDims
from scipy.stats import gaussian_kde

import pickle

# Load pickled data
def load_data(file) -> tuple[ShipDims,ShipDims,ShipDims]:
    with open(file,"rb") as f:
        try:
            CargoDims, TankerDims, PassengerDims = pickle.load(f)
        except:
            raise ValueError("File does not contain the expected data.")
    return CargoDims, TankerDims, PassengerDims

    
# Create a kde plot for each ship category
def plot_hist(data: tuple[ShipDims,ShipDims,ShipDims]) -> None:
    """
    Plots a histogram for each ship category.
    """
    for ship_type in data:
        fig, ax = plt.subplots(figsize=(8,5))
        ax2 = ax.twinx()
        # Create a numpy array from the set of tuples
        _,lengths,widths = np.array(list(ship_type.dims)).T
        # Plot the kde
        kde_length = gaussian_kde(lengths)
        kde_width = gaussian_kde(widths)
        xx = np.linspace(0, max(max(lengths),max(widths)), 500)
        
        ax.set_title(ship_type.name)
        ax.set_xlabel("Length")
        ax.set_ylabel("Count")
        ax2.set_ylabel("Density")
        
        ax.hist(
            [lengths,widths], 
            bins=200, 
            histtype = "barstacked", 
            density=False,
            rwidth=1.8,
            label=["Length","Width"],
            color=["#2a9d8f","#e76f51"]
        )
        # Add kde to plot
        ax2.plot(xx, kde_length(xx), color="#c1121f", label="Length KDE")    
        ax2.plot(xx, kde_width(xx), color="#003049", label="Width KDE")
        
        #ax.set_ylim(0, max(max(lengths),max(widths)))
        ax.set_xlim(-2, 350)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"aisstats/out/{ship_type.name}_lenwidth-hist.png",dpi=300)
        plt.close()
    return None

if __name__ == "__main__":
    # Load data
    CargoDims, TankerDims, PassengerDims = load_data("aisstats/extracted/dims.pickle")
    # Plot histogram
    plot_hist((CargoDims, TankerDims, PassengerDims))