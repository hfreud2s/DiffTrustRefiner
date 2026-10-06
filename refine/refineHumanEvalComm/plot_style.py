import matplotlib as mpl

DARK_BLUE       = '#576AAC'
LIGHT_BLUE      = '#A2A9ED'    
DARK_SALMON     = '#FA5A50'
LIGHT_SALMON    = '#FF998A'
NEUTRAL_COLOR   = '#C8C4CC'
SAGE_GREEN      = '#BECD63'
MEDIUM_GREEN    = '#87A878'
TOMATO_RED      = '#D61D2F'
DARK_GREEN      = '#003D3D'
WARM_GRAY       = "#666666"

mpl.rcParams.update({
    # --- Figure
    "figure.dpi":            100,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",

    # --- Font
    "font.family":           "sans-serif",
    "font.size":             11,
    "axes.titlesize":        12,
    "axes.labelsize":        10,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       10,

    # --- Axes
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "axes.linewidth":        0.8,

    # --- Grid
    "axes.grid":             True,
    "grid.color":            NEUTRAL_COLOR,
    "grid.linewidth":        0.4,
    "grid.linestyle":        "--",
    "grid.alpha":            0.6,

    # --- Ticks
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.color":           WARM_GRAY,
    "ytick.color":           WARM_GRAY,
    "axes.labelcolor":       WARM_GRAY,
    "axes.edgecolor":        WARM_GRAY,

    # --- Lines & markers
    "lines.linewidth":       1.2,
    "lines.markersize":      5,
    "lines.linestyle":       '--',

    # --- Color cycle
    "axes.prop_cycle": mpl.cycler("color", [
        DARK_BLUE, DARK_SALMON, SAGE_GREEN,
        TOMATO_RED, DARK_GREEN, MEDIUM_GREEN,
    ]),
    "axes.titlecolor":       WARM_GRAY,

    # --- Legend
    "legend.frameon":        True,
    "legend.borderpad":      0.6,
    "legend.labelcolor":     WARM_GRAY,

    # --- Layout
    "figure.constrained_layout.use": True,
})