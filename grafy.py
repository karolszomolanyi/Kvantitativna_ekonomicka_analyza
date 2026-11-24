import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def set_theory_style():

    plt.rcParams.update({
        "figure.figsize": (8,6),
        "figure.dpi": 120,
        "font.size": 12,
        "lines.linewidth": 2,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def set_data_style():

    plt.rcParams.update({
        "figure.figsize": (8,4.5),
        "figure.dpi": 120,
        "font.size": 11,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })

def f(k,alpha=1/3):
    return k**alpha
   
def v(c):
    return np.log(c)


def uzitocnost(c, cp, beta=None, r=None):

    if beta is None:
        beta = 1/(1+r)

    return v(c) + beta*v(cp)

def optimum(y, yp, r, t=0, tp=0, beta=None):

    if beta is None:
        beta = 1/(1+r)

    we = y - t + (yp - tp)/(1+r)

    c = we/(1+beta)

    cp = beta*(1+r)*c

    u = uzitocnost(c, cp, beta)

    return {
        "we": we,
        "c": c,
        "cp": cp,
        "u": u,
        "beta": beta
    }


def ekonomicke_osi(ax, xlabel, ylabel,
                   xmin=0, ymin=0):

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(left=xmin)
    ax.set_ylim(bottom=ymin)

    ax.margins(0)

    ax.text(
        1.02, -0.02, xlabel,
        transform=ax.transAxes,
        ha="right",
        va="top"
    )

    ax.text(
        -0.02, 1.02, ylabel,
        transform=ax.transAxes,
        ha="right",
        va="top"
    )

    return ax

def nakresli_rozpocet_intertemporal(
    ax,
    r,
    y,
    yp,
    t=0,
    tp=0,
    color="green",
    linestyle="-"
):

    y1 = y - t
    y2 = yp - tp

    c_max = y1 + y2/(1+r)
    cp_max = (1+r)*y1 + y2

    c = np.linspace(0, c_max, 200)

    cp = cp_max - (1+r)*c

    ax.plot(
        c,
        cp,
        color=color,
        linestyle=linestyle
    )

    return {
        "y1": y1,
        "y2": y2,
        "c_max": c_max,
        "cp_max": cp_max
    }

def nakresli_indiferencnu_intertemporal(
    ax,
    u,
    beta,
    c_min=1,
    c_max=100,
    color="blue",
    linestyle="-"
):

    c = np.linspace(c_min, c_max, 500)

    cp = np.exp((u - np.log(c))/beta)

    ax.plot(
        c,
        cp,
        color=color,
        linestyle=linestyle
    )

    return {"u": u}


def nakresli_bod(
    ax,
    x,
    y,
    nazov="E",
    vodiace_ciary=False,
    dx=5,
    dy=5,
    color="black"
):

    ax.scatter(
        [x],
        [y],
        color=color,
        zorder=10
    )

    ax.annotate(
        nazov,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points"
    )

    if vodiace_ciary is True:

        ax.plot(
            [0, x],
            [y, y],
            "--",
            color="gray",
            linewidth=1
        )

        ax.plot(
            [x, x],
            [0, y],
            "--",
            color="gray",
            linewidth=1
        )

    elif vodiace_ciary == "horizontal":

        ax.plot(
            [0, x],
            [y, y],
            "--",
            color="gray",
            linewidth=1
        )

    elif vodiace_ciary == "vertical":

        ax.plot(
            [x, x],
            [0, y],
            "--",
            color="gray",
            linewidth=1
        )

    return {"x": x, "y": y}

def vyznac_na_osi(
    ax,
    x=None,
    y=None,
    popis_x=None,
    popis_y=None,
    color="black",
    dx_x=0,
    dy_x=-15,
    dx_y=-10,
    dy_y=0
):

    if x is not None and popis_x is not None:

        ax.annotate(
            popis_x,
            xy=(x, 0),
            xytext=(dx_x, dy_x),
            textcoords="offset points",
            ha="center",
            color=color
        )

    if y is not None and popis_y is not None:

        ax.annotate(
            popis_y,
            xy=(0, y),
            xytext=(dx_y, dy_y),
            textcoords="offset points",
            ha="right",
            va="center",
            color=color
        )

def nakresli_sipku(
    ax,
    x1,y1,
    x2,y2,
    color="black"
):

    ax.annotate(
        "",
        xy=(x2,y2),
        xytext=(x1,y1),
        arrowprops=dict(
            arrowstyle="->",
            color=color
        )
    )

def deep_parameters():
    alpha = 1/3 # podiel kapitálu na HDP
    phi = 1/2 # elasticita užitočnosti podľa voľného času
    g_share = 0.2 # podiel vládnych nákupov na HDP
    h = 1 + phi*(1-g_share)/((1-phi)*(1-alpha)) # voľný čas kalibrovaný tak, aby Y=N=A=1
    return alpha, phi, g_share, h

def deep_parameters_linear():
    alpha = 0 # podiel kapitálu na HDP
    phi = 1/2 # elasticita užitočnosti podľa voľného času
    g_share = 0.6 # podiel vládnych nákupov na HDP
    h = 1 + phi*(1-g_share)/((1-phi)*(1-alpha)) # voľný čas kalibrovaný tak, aby Y=N=A=1
    return alpha, phi, g_share, h

def volny_cas(w, phi, h, pi_T):
    """
    Optimálny voľný čas
    """
    return phi*h + phi* pi_T/w


def spotreba(w, phi, h, pi_T):
    """
    Optimálna spotreba
    """
    return (1-phi)*(w*h +  pi_T)


def ponuka_prace(w, phi, h, pi_T):
    """
    Ponuka práce
    """
    l = volny_cas(w, phi, h, pi_T=pi_T)
    return h - l

def produkcna_funkcia(N, A, alpha):
    """
    Produkčná funkcia
    """
    return A*N**(1-alpha)

def MPN(N, A, alpha):
    """
    Hraničný produkt práce
    """
    return (1-alpha)*A*N**(-alpha)

def ekonomicke_osi(ax, xlabel, ylabel,
                   xmin=0, ymin=0):

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(left=xmin)
    ax.set_ylim(bottom=ymin)

    ax.margins(0)

    ax.text(
        1.02, -0.02, xlabel,
        transform=ax.transAxes,
        ha="right",
        va="top"
    )

    ax.text(
        -0.02, 1.02, ylabel,
        transform=ax.transAxes,
        ha="right",
        va="top"
    )

    return ax


def nakresli_ponuku_prace(ax, phi, h, pi_T,
                          color="green", linestyle="-",label = "$N^s$"
                         ):
   
    
    w_grid = np.linspace(0.2,1.,500)
    N = ponuka_prace(w_grid,phi,h,pi_T)
    ax.plot(N,w_grid,
        color=color,linestyle=linestyle,label=label
    )

def nakresli_dopyt_po_praci(ax, A, alpha,
                            color="blue", linestyle="-",label=r"$N^d$"):
    N_grid = np.linspace(0.7, 1.3, 500)
    w = MPN(N_grid, A, alpha)

    ax.plot(
        N_grid, w,
        color=color,
        linestyle=linestyle,
        label=label
    )

def nakresli_produkcnu_f(ax,A,alpha,h,
                            color="green", linestyle="-"):
    N_grid = np.linspace(0,h,500)
    Y = produkcna_funkcia(N_grid,A=A,alpha=alpha)
    ax.plot(N_grid, Y,color=color,linestyle=linestyle)

#Ako zobraziť dotyčnice ku krivke?
#Dotyčnica je priamka y = a + bx
#sklon dotyčnice je prvá derivácia g(x): b = g'(x)
#Dotyčnica musí prechádzať bodom dotyku, v ktorom platí: g(x) = a + bx
#Teda: g(x) = a + g'(x) x
#Takže a = g(x) - g'(x) x

def dotycnica(g,gder,x):
    """
    Príklad: dotycnica(produkcna_funkcia,MPN,0.5)

    """
    b = gder(x) #sklon dotyčnice
    a = g(x) - gder(x)*x #intercept dotyčnice
    return a,b

def nakresli_dotycnicu(ax,g,gder,bod,rozsah,
                      color="black",linestyle="-",label=None,linewidth=None):
    a,b = dotycnica(g,gder,bod)
    x = np.linspace(bod-rozsah/2,bod+rozsah/2,500)
    y=a+b*x
    ax.plot(x,y,color=color,linestyle=linestyle,label=label,linewidth=linewidth)


def ekonomicke_osi_HVM(ax, xlabel, ylabel,G):

    ax.set_xticks([])
    ax.set_yticks([])

    # schovaj hornú a pravú os
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # x-os presuň na y=0
    ax.spines["bottom"].set_position(("data", 0))

    # y-os nech zostane v x=0
    ax.spines["left"].set_position(("data", 0))

    # názov osi x na koniec osi
    ax.annotate(xlabel,xy=(ax.get_xlim()[1], 0),xytext=(6, -4),
                textcoords="offset points",ha="left",va="top")
    ax.set_ylim(bottom=-G-0.05)
    ax.set_xlim(left=0)
    # názov osi y hore
    ax.text(-0.02,1.00,ylabel,transform=ax.transAxes,ha="right",va="top")
    

    return ax

def HVM(l,A,alpha,h,G):
    return A*(h-l)**(1-alpha) - G
# táto funkca poslúži pre dotyčnicu v prostrednom 
def HVM0(l,A,alpha,h): # HVM s nulovými vládnymi nákupmi
    return HVM(l,A,alpha,h,G=0)
def HVM_der(l,A,alpha,h):
    return -MPN(h-l,A,alpha)

def nakresli_HVM(ax,
                 A, alpha, h, G,
                 l_max=None,
                 color="green",
                 linestyle="-",
                 prerus_pre_C_zaporne=False):

    if l_max is None:
        l_max = h

    l_grid = np.linspace(0, l_max, 500)

    C = HVM(l_grid,A,alpha,h,G)

    if not prerus_pre_C_zaporne:

        ax.plot(l_grid, C,
            color=color,linestyle=linestyle
        )

    else:

        kladne = C >= 0
        zaporne = C < 0

        ax.plot(l_grid[kladne],C[kladne],
            color=color,linestyle="-"
        )

        ax.plot(
            l_grid[zaporne],C[zaporne],
            color=color,linestyle="--"
        )

def U_lC(l,C,phi):
    return l**phi * C**(1-phi)

def nakresli_indiferencnu_lC(ax, U, phi, h, l_max, l_min=0.64,
                             color="blue",linestyle="-",label=None):

    l_grid = np.linspace(l_min, l_max, 500)

    C = (U / l_grid**phi)**(1/(1-phi))

    ax.plot(l_grid,C,
        color=color,linestyle=linestyle,label=label)

def konkurencna_rovnovaha(phi,h,G,A,alpha):

    def vr(endog):
        l,C,Y,N,w = endog
        return(C*phi/(l*(1-phi)) - A*(1-alpha)*(h-l)**(-alpha), #MRS=MRT
              C - A*(h-l)**(1-alpha) + G, #PPF
              Y-C-G,# Y
              N-h+l,
              w-(1-alpha)*A*(h-l)**(-alpha))

    start = (h-1,1-G,1.,1.,1-alpha)
    
    x = fsolve(vr,start)
    return x

def hicks_SE(A,U,
             phi,h,alpha):

    def rovnica(l):

        C = (U/l**phi)**(1/(1-phi))

        return (
            phi*C/(l*(1-phi))
            - (1-alpha)*A*(h-l)**(-alpha)
        )

    l_SE = fsolve(rovnica,h-1)[0]

    C_SE = (U/l_SE**phi)**(1/(1-phi))

    return l_SE,C_SE

def nakresli_rozpocet_klin(
    ax,
    r1,
    r2,
    y,
    yp,
    t=0,
    tp=0,
    color="green",
    linestyle="-",
    aux_linestyle="--"
):
    
    y1 = y - t
    y2 = yp - tp

    # maximálny rozsah grafu
    c_max = y1 + y2/(1+r1)
    cp_max = (1+r1)*y1 + y2

    c = np.linspace(0, c_max, 500)

    # obe rozpočtové priamky cez prídelový bod E=(y1,y2)
    cp_r1 = y2 + (1+r1)*(y1 - c)
    cp_r2 = y2 + (1+r2)*(y1 - c)

    lavy = c <= y1
    pravy = c >= y1

    # ľavá časť: r1 plná
    ax.plot(
        c[lavy],
        cp_r1[lavy],
        color=color,
        linestyle=linestyle
    )

    ax.plot(
        c[lavy],
        cp_r2[lavy],
        color=color,
        linestyle=aux_linestyle,
        linewidth=1.0
    )

    # pravá časť: r2 plná
    ax.plot(
        c[pravy],
        cp_r2[pravy],
        color=color,
        linestyle=linestyle
    )

    ax.plot(
        c[pravy],
        cp_r1[pravy],
        color=color,
        linestyle=aux_linestyle,
        linewidth=1.0
    )

def nakresli_rozpocet_kolateral(
    ax,
    r,
    y,
    yp,
    B,
    t=0,
    tp=0,
    color="green"
):

    """
    B je pH/(1+r)
    """

    y1 = y - t
    y2 = yp - tp + B*(1+r)

    c_grid = np.linspace(0, y1 + B, 300)

    cp_grid = (
        (1+r)*y1
        + y2
        - (1+r)*c_grid
    )

    ax.plot(
        c_grid,
        cp_grid,
        color=color
    )

    # bod zlomu G

    c_G = y1 + B

    cp_G = yp-tp

    # vertikála GD

    ax.plot(
        [c_G, c_G],
        [cp_G, 0],
        color=color
    )


    return {
        "E": (y1, y2),
        "G": (c_G, cp_G)
    }

    return {
        "E": (y1, y2),
        "c_max": c_max,
        "cp_max": cp_max
    }

def plot_ts(
    data,
    filename=None,
    labels=None,
    title=None,
    ylabel=None,
    common_sample=True,
    normalize=False,
    zero_line=False,
    recessions=None,
    figsize=None,
    YearLocator=5
):
    """
    Graf priebehu jedného alebo viacerých časových radov.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame alebo list[pd.Series]
        Vstupné časové rady.

    labels : list[str], optional
        Názvy do legendy.

    title : str, optional
        Názov grafu.

    ylabel : str, optional
        Popis osi y.

    common_sample : bool, default=True
        Oreže rady na spoločné obdobie.

    normalize : bool, default=False
        Normalizuje prvé pozorovanie na 100.

    zero_line : bool, default=False
        Nakreslí horizontálnu čiaru v nule.

    recessions : bool, default=False
        Zobrazí vybrané recesie.

    figsize : tuple, optional
        Prepíše predvolenú veľkosť grafu.
    
    YearLocator : interval rokov zobrazovaných na osi x 

    Returns
    -------
    fig, ax, df
    """

    # --------------------------------------------------
    # Vytvorenie DataFrame
    # --------------------------------------------------

    if isinstance(data, pd.Series):
        df = data.to_frame()

    elif isinstance(data, pd.DataFrame):
        df = data.copy()

    elif isinstance(data, (list, tuple)):
        df = pd.concat(data, axis=1)

    else:
        raise TypeError(
            "data musí byť Series, DataFrame alebo list Series"
        )

    # --------------------------------------------------
    # Datetime index
    # --------------------------------------------------

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError(
                "Index nie je datetime a nepodarilo sa ho previesť."
            )

    df = df.sort_index()

    # --------------------------------------------------
    # Legendy
    # --------------------------------------------------

    if labels is not None:

        if len(labels) != len(df.columns):
            raise ValueError(
                "Počet labels sa nerovná počtu radov."
            )

        df.columns = labels

    # --------------------------------------------------
    # Spoločná vzorka
    # --------------------------------------------------

    if common_sample:

        start = max(
            df[c].first_valid_index()
            for c in df.columns
        )

        end = min(
            df[c].last_valid_index()
            for c in df.columns
        )

        df = df.loc[start:end]

    # --------------------------------------------------
    # Normalizácia
    # --------------------------------------------------

    if normalize:

        df = (
            df
            .div(df.iloc[0])
            .mul(100)
        )

    # --------------------------------------------------
    # Graf
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    for col in df.columns:
        ax.plot(df.index, df[col], label=col)

    # --------------------------------------------------
    # Voľby
    # --------------------------------------------------

    if title:
        ax.set_title(title)

    if ylabel:
        ax.set_ylabel(ylabel)

    if zero_line:
        ax.axhline(
            0,
            color="black",
            lw=0.8
        )

    if recessions:

        for peak, trough in recessions:

            ax.axvspan(
                pd.Timestamp(peak),
                pd.Timestamp(trough),
                color="lightgray",
                alpha=0.3
            )

    # --------------------------------------------------
    # Os x
    # --------------------------------------------------

    
    ax.xaxis.set_major_locator(
        mdates.YearLocator(YearLocator)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

    fig.autofmt_xdate()

    if len(df.columns) > 1:
        ax.legend()
    
    plt.tight_layout()

    if filename is not None:
        plt.savefig(
            f"Data/{filename}.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    return fig, ax, df

def plot_model_path(
    data,
    filename=None,
    labels=None,
    title=None,
    ylabel=None,
    steady_states=None,
    markers=True,
    markersize=4,
    zero_line=False,
    figsize=None,
    shock_period=None,
    series_color=None
):
    """
    Graf modelovej trajektórie jednej alebo viacerých veličín.

    Vhodné najmä pre diskrétne dynamické modely, kde index
    predstavuje obdobia t = 0, 1, ..., T.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, array-like alebo list[pd.Series]
        Modelová trajektória jednej alebo viacerých veličín.

    filename : str, optional
        Názov súboru bez prípony. Graf sa uloží do priečinka Data.

    labels : list[str], optional
        Názvy jednotlivých trajektórií do legendy.

    title : str, optional
        Názov grafu.

    ylabel : str, optional
        Popis osi y.

    steady_state : float, optional
        Hodnota ustáleného stavu. Ak je zadaná, zobrazí sa
        ako horizontálna prerušovaná čiara.

    markers : bool, default=True
        Ak True, vyznačí hodnotu modelu v každom období bodom.

    markersize : float, default=4
        Veľkosť markerov.

    zero_line : bool, default=False
        Ak True, zobrazí horizontálnu čiaru v nule.

    figsize : tuple, optional
        Prepíše predvolenú veľkosť grafu.

    Returns
    -------
    fig, ax, df
    """
    
    set_data_style()

    # --------------------------------------------------
    # Vytvorenie DataFrame
    # --------------------------------------------------

    if isinstance(data, pd.Series):
        df = data.to_frame()

    elif isinstance(data, pd.DataFrame):
        df = data.copy()

    elif isinstance(data, (list, tuple)):

        if all(isinstance(x, pd.Series) for x in data):
            df = pd.concat(data, axis=1)

        else:
            df = pd.DataFrame(data)

    elif isinstance(data, np.ndarray):

        if data.ndim == 1:
            df = pd.DataFrame(data)

        elif data.ndim == 2:
            df = pd.DataFrame(data)

        else:
            raise ValueError(
                "NumPy array musí byť jednorozmerný alebo dvojrozmerný."
            )

    else:
        raise TypeError(
            "data musí byť Series, DataFrame, array alebo list Series"
        )

    # --------------------------------------------------
    # Legendy
    # --------------------------------------------------

    if labels is not None:

        if len(labels) != len(df.columns):
            raise ValueError(
                "Počet labels sa nerovná počtu trajektórií."
            )

        df.columns = labels

    # --------------------------------------------------
    # Graf
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    for j, col in enumerate(df.columns):
    
        if isinstance(series_color, (list, tuple)):
            if len(series_color) != len(df.columns):
                raise ValueError(
                    "Počet farieb v series_color sa nerovná počtu trajektórií."
                )
            color = series_color[j]
        else:
            color = series_color
    
        ax.plot(
            df.index,
            df[col],
            label=col,
            marker="o" if markers else None,
            markersize=markersize,
            color=color
        )

    # --------------------------------------------------
    # Ustálený stav
    # --------------------------------------------------

    if steady_states is not None:

        for label, item in steady_states.items():

            if isinstance(item, tuple):
                value, color = item
            else:
                value = item
                color = "black"

            ax.axhline(
                y=value,
                color=color,
                linestyle="--",
                linewidth=1.0,
                label=label
            )

    # --------------------------------------------------
    # Voľby
    # --------------------------------------------------

    if title:
        ax.set_title(title)

    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xlabel("Obdobie")

    if zero_line:
        ax.axhline(
            0,
            color="black",
            lw=0.8
        )

    # --------------------------------------------------
    # Legenda
    # --------------------------------------------------

    if len(df.columns) > 1 or steady_states is not None:
        ax.legend()
    
    # --------------------------------------------------
    # Označenie šoku
    # --------------------------------------------------
        
    if shock_period is not None:
        ax.axvline(
            shock_period,
            color="gray",
            linestyle=":",
            linewidth=1.0
        )
    #--------------------
    # výška y osi
    #---------------------
    
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(
        ymin,
        ymax + 0.08 * (ymax - ymin)
    )

    # --------------------------------------------------
    # Výstup
    # --------------------------------------------------

    plt.tight_layout()

    if filename is not None:

        plt.savefig(
            f"images/{filename}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    return fig, ax, df

#základná rovnica Solowovho modelu
def phi(k,A,alpha,n,s,delta):
    return s * A * f(k,alpha) / (1 + n) + (1 - delta) * k / (1 + n)

def rovnice(k,A,alpha,s):
    y = A * f(k, alpha)
    i = s * y
    c = (1 - s) * y
    return y, i, c

 
def us(A,alpha,n,s,delta):
    k_us = (s * A / (delta + n))**(1 / (1 - alpha))
    y_us, i_us, c_us = rovnice(k_us, A, alpha, s)
 
    return k_us, y_us, i_us, c_us

# výpočet radu kapitál
def rady(k0,N0,A0,alpha,n,s,delta,g=0,T=100):
    k = np.zeros(T)
    N = np.zeros(T)
    A = np.zeros(T)
    k[0] = k0
    N[0] = N0
    A[0] = A0
    index = range(T)

    for t in index[:-1]:
        A[t+1] = (g+1)*A[t]
        k[t+1] = phi(k[t],A[t+1],alpha,n,s,delta)
        N[t+1] = (n+1)*N[t]
    
    return (k, index, N, A)

def plot_cobweb(
    g,
    xmin,
    xmax,
    x0,
    num_arrows=4,
    var="x",
    steady_state=None,
    title=None,
    filename=None,
    figsize=None
):
    """
    Pavučinový diagram pre dynamickú rovnicu

        x_{t+1} = g(x_t)
    """

    # --------------------------------------------------
    # Graf
    # --------------------------------------------------
    
    set_theory_style()

    xgrid = np.linspace(xmin, xmax, 300)

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    # --------------------------------------------------
    # Dynamická rovnica a 45-stupňová priamka
    # --------------------------------------------------

    ax.plot(
        xgrid,
        g(xgrid),
        label=rf"${var}_{{t+1}}=𝜙({var}_t)$"
    )

    ax.plot(
        xgrid,
        xgrid,
        color="black",
        lw=1,
        label=r"$45^\circ$"
    )

    # --------------------------------------------------
    # Štartovacia hodnota
    # --------------------------------------------------

    x = x0

    vyznac_na_osi(
        ax,
        x=x,
        popis_x=rf"${var}_0$"
    )

    # --------------------------------------------------
    # Pavučina
    # --------------------------------------------------

    for i in range(num_arrows):

        gx = g(x)

        # Prvý vertikálny pohyb začína na osi x
        if i == 0:
            nakresli_sipku(
                ax,
                x, xmin,
                x, gx
            )
        else:
            nakresli_sipku(
                ax,
                x, x,
                x, gx
            )

        # Bod (x_i, x_{i+1}) na dynamickej krivke
        nakresli_bod(
            ax,
            x,
            gx,
            nazov="",
            vodiace_ciary=True
        )
        
        # Nová hodnota na oboch osiach
        vyznac_na_osi(
            ax,
            x=gx,
            y=gx,
            popis_x=rf"${var}_{i+1}$",
            popis_y=rf"${var}_{i+1}$"
        )

        # Horizontálny pohyb na 45-stupňovú priamku
        nakresli_sipku(
            ax,
            x, gx,
            gx, gx
        )

        x = gx
    
    # posledná vodiaca čiara
    ax.plot(
        [x, x],
        [0, x],
        "--",
        color="gray",
        linewidth=1
    )

    # --------------------------------------------------
    # Ustálený stav
    # --------------------------------------------------

    if steady_state is not None:

        nakresli_bod(
            ax,
            steady_state,
            steady_state,
            nazov="",
            vodiace_ciary=True
        )

        vyznac_na_osi(
            ax,
            x=steady_state,
            y=steady_state,
            popis_x=rf"${var}^*$",
            popis_y=rf"${var}^*$"
        )


    # --------------------------------------------------
    # Ekonomické osi
    # --------------------------------------------------

    ekonomicke_osi(
        ax,
        xlabel=rf"${var}_t$",
        ylabel=rf"${var}_{{t+1}}$",
        xmin=xmin,
        ymin=xmin
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)

    # --------------------------------------------------
    # Názov a legenda
    # --------------------------------------------------

    if title:
        ax.set_title(title)

    ax.legend(
        ncol=2,
        frameon=False,
        loc="upper left"
    )

    plt.tight_layout()

    # --------------------------------------------------
    # Uloženie
    # --------------------------------------------------

    if filename is not None:
                
        plt.savefig(
            f"images/{filename}.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()
        
    return fig, ax

def nakresli_funkciu(
    ax,
    funkcia,
    xmin,
    xmax,
    nazov=None,
    color=None,
    linestyle="-",
    linewidth=None,
    n=300
):
    """
    Nakreslí funkciu y = f(x) do existujúcich osí.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Osi, do ktorých sa funkcia kreslí.

    funkcia : callable
        Kreslená funkcia.

    xmin, xmax : float
        Rozsah osi x.

    nazov : str, optional
        Názov funkcie v legende.

    color : str, optional
        Farba čiary.

    linestyle : str, default="-"
        Štýl čiary.

    linewidth : float, optional
        Hrúbka čiary.

    n : int, default=300
        Počet bodov mriežky.

    Returns
    -------
    line
    """

    x = np.linspace(xmin, xmax, n)
    y = funkcia(x)

    line, = ax.plot(
        x,
        y,
        label=nazov,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )

    return line

def nakresli_dvojsipku(
    ax,
    x1,
    y1,
    x2,
    y2,
    nazov=None,
    dx=5,
    dy=0,
    color="gray",
    nazov_color="black",
    linewidth=0.5,
    alpha=0.6
):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="<->",
            color=color,
            linewidth=linewidth,
            alpha=alpha
        )
    )

    if nazov is not None:
        ax.annotate(
            nazov,
            xy=((x1 + x2) / 2, (y1 + y2) / 2),
            xytext=(dx, dy),
            textcoords="offset points",
            va="center",
            color=nazov_color
        )

def prepoj_body(
    fig,
    axes,
    value,
    points,
    direction="vertical",
    labels=None,
    value_label=None,
    axis_labels=None,
    point_color="black",
    point_size=30,
    line_color="gray",
    linestyle="--",
    linewidth=1,
    dx=5,
    dy=5,
    value_label_shift=[-7,0],
    axis_label_dx=[-7,-7],
    axis_label_dy=None
):
    """
    Vyznačí body vo viacerých axes, nakreslí vodiace čiary
    k osiam, označí hodnoty na osiach a prepojí body jednou
    súvislou vodiacou čiarou naprieč panelmi.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure obsahujúca všetky axes.

    axes : list
        Zoznam axes, cez ktoré má viesť spoločná čiara.

    value : float
        Spoločná hodnota:
        - x pri direction="vertical"
        - y pri direction="horizontal"

    points : list[float]
        Druhá súradnica bodu v jednotlivých axes:
        - hodnoty y pri direction="vertical"
        - hodnoty x pri direction="horizontal"

    direction : {"vertical", "horizontal"}, default="vertical"
        Smer spoločnej vodiacej čiary.

    labels : list[str], optional
        Popisy jednotlivých bodov.

    value_label : str, optional
        Označenie spoločnej hodnoty na osi.
        Pri vertical sa zobrazí na x-osi posledného axes,
        pri horizontal na y-osi prvého axes.

    axis_labels : list[str], optional
        Označenia jednotlivých hodnôt points na osiach.

    point_color : str, default="black"
        Farba bodov.

    point_size : float, default=30
        Veľkosť bodov.

    line_color : str, default="gray"
        Farba vodiacich čiar.

    linestyle : str, default="--"
        Štýl vodiacich čiar.

    linewidth : float, default=1
        Hrúbka vodiacich čiar.

    dx, dy : float, default=5
        Posun popisov bodov.

    axis_label_dx, axis_label_dy : list[float], optional
        Dodatočný posun označení na osiach v bodoch.
        Hodnoty sú v points.

    Returns
    -------
    points_out : list
        Zoznam objektov bodov.

    line : Line2D
        Spoločná vodiaca čiara.
    """

    n = len(axes)

    if len(axes) != len(points):
        raise ValueError(
            "Počet axes sa musí rovnať počtu bodov."
        )

    if labels is not None and len(labels) != len(axes):
        raise ValueError(
            "Počet labels sa musí rovnať počtu bodov."
        )

    if axis_labels is not None and len(axis_labels) != len(axes):
        raise ValueError(
            "Počet axis_labels sa musí rovnať počtu bodov."
        )

    if direction not in ("vertical", "horizontal"):
        raise ValueError(
            'direction musí byť "vertical" alebo "horizontal".'
        )

    n = len(axes)

    if axis_label_dx is None:
        axis_label_dx = [0] * n

    if axis_label_dy is None:
        axis_label_dy = [0] * n

    if len(axis_label_dx) != n or len(axis_label_dy) != n:
        raise ValueError(
            "axis_label_dx a axis_label_dy musia mať rovnakú "
            "dĺžku ako axes."
        )

    points_out = []

    # --------------------------------------------------
    # Body, vodiace čiary a označenia osí
    # --------------------------------------------------

    for j, (ax, point) in enumerate(zip(axes, points)):

        if direction == "vertical":
            x, y = value, point

            # vodiaca čiara od y-osi k bodu
            ax.hlines(
                y,
                xmin=ax.get_xlim()[0],
                xmax=x,
                color=line_color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=1
            )

        else:
            x, y = point, value

            # vodiaca čiara od x-osi k bodu
            ax.vlines(
                x,
                ymin=ax.get_ylim()[0],
                ymax=y,
                color=line_color,
                linestyle=linestyle,
                linewidth=linewidth,
                zorder=1
            )

        # bod
        obj = ax.scatter(
            [x],
            [y],
            color=point_color,
            s=point_size,
            zorder=10
        )

        points_out.append(obj)

        # popis bodu
        if labels is not None:
            ax.annotate(
                labels[j],
                xy=(x, y),
                xytext=(dx, dy),
                textcoords="offset points"
            )

        # označenie hodnoty point na osi
        if axis_labels is not None:

            if direction == "vertical":
                ax.annotate(
                    axis_labels[j],
                    xy=(0, y),
                    xycoords=("axes fraction", "data"),
                    xytext=(
                        axis_label_dx[j],
                        axis_label_dy[j]
                    ),
                    textcoords="offset points",
                    ha="right",
                    va="center"
                )

            else:
                ax.annotate(
                    axis_labels[j],
                    xy=(x, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(
                        axis_label_dx[j],
                        axis_label_dy[j]
                    ),
                    textcoords="offset points",
                    ha="center",
                    va="top"
                )
    if value_label_shift is None:
        value_label_shift = [0] * n
    elif np.isscalar(value_label_shift):
        value_label_shift = [value_label_shift] * n

    # --------------------------------------------------
    # Označenie spoločnej hodnoty
    # --------------------------------------------------
    
    if value_label is not None:
    
        for j, ax in enumerate(axes):
    
            if direction == "vertical":
    
                # označenie spoločnej hodnoty na x-osi
                ax.annotate(
                    value_label,
                    xy=(value, 0),
                    xycoords=("data", "axes fraction"),
                    xytext=(value_label_shift[j], -5),
                    textcoords="offset points",
                    ha="center",
                    va="top"
                )
        
    # --------------------------------------------------
    # Spoločná vodiaca čiara
    # --------------------------------------------------

    if direction == "vertical":

        x_display = axes[0].transData.transform(
            (value, points[0])
        )[0]

        x_fig = fig.transFigure.inverted().transform(
            (x_display, 0)
        )[0]

        # od x-osi posledného axes
        y_start_display = axes[-1].transData.transform(
            (value, 0)
        )[1]

        y_start_fig = fig.transFigure.inverted().transform(
            (0, y_start_display)
        )[1]

        # po bod prvého axes
        y_end_display = axes[0].transData.transform(
            (value, points[0])
        )[1]

        y_end_fig = fig.transFigure.inverted().transform(
            (0, y_end_display)
        )[1]

        line = plt.Line2D(
            [x_fig, x_fig],
            [y_start_fig, y_end_fig],
            transform=fig.transFigure,
            color=line_color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=5
        )

    else:

        y_display = axes[0].transData.transform(
            (points[0], value)
        )[1]

        y_fig = fig.transFigure.inverted().transform(
            (0, y_display)
        )[1]

        # od y-osi prvého axes
        x_start_display = axes[0].transData.transform(
            (0, value)
        )[0]

        x_start_fig = fig.transFigure.inverted().transform(
            (x_start_display, 0)
        )[0]

        # po bod posledného axes
        x_end_display = axes[-1].transData.transform(
            (points[-1], value)
        )[0]

        x_end_fig = fig.transFigure.inverted().transform(
            (x_end_display, 0)
        )[0]

        line = plt.Line2D(
            [x_start_fig, x_end_fig],
            [y_fig, y_fig],
            transform=fig.transFigure,
            color=line_color,
            linestyle=linestyle,
            linewidth=linewidth,
            zorder=5
        )

    fig.add_artist(line)

    return points_out, line

def nakresli_zvislu_ciaru(
    ax,
    x,
    nazov=None,
    color=None,
    linestyle="-",
    linewidth=None,
):
    return ax.axvline(
        x=x,
        label=nazov,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
    )