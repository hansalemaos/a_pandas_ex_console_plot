import operator
from functools import reduce
from PIL import Image
import numpy as np
import pandas as pd
from a_pandas_ex_columns_and_index import rename_columns
from a_pandas_ex_less_memory_more_speed import optimize_only_int
from ansi.colour.rgb import rgb256
from ansi.colour import fg, bg, fx
import matplotlib
from pandas.core.frame import DataFrame, Series

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def convert_img_to_pil(image):

    if isinstance(image, str):
        colourImg = Image.open(image)

    elif "PIL." in str(type(image)):
        colourImg = image
    elif isinstance(image, (np.ndarray, pd.DataFrame)):
        if isinstance(image, pd.DataFrame):
            image = convert2np(image)
        colourImg = Image.fromarray(image)
    else:
        colourImg = image
    return colourImg


def image2df(image, convert, check_invert=True):

    convertto = convert
    colourImg = convert_img_to_pil(image)

    if convertto is not None:
        if convertto == "L" or convertto == "1" or convertto == "LA":
            colourImg = colourImg.convert("RGB")
            colourImg = Image.fromarray(
                np.array(colourImg, dtype=np.uint8)[..., [2, 1, 0]]
            )

        colourImg = colourImg.convert(convertto)
    mode = colourImg.mode
    df2 = pd.DataFrame()
    if (
        mode != "RGBA"
        and mode != "RGB"
        and mode != "L"
        and mode != "1"
        and mode != "LA"
    ):
        colourImg = colourImg.convert("RGB")
        mode = "RGB"
    if mode == "RGB":
        colourArray = np.array(colourImg.getdata(), dtype=np.uint8).reshape(
            colourImg.size + (3,)
        )
        indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
        allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 5))
        df2 = pd.DataFrame(allArray, columns=["y", "x", "blue", "green", "red"]).copy()

    elif mode == "RGBA":
        colourArray = np.array(colourImg.getdata(), dtype=np.uint8).reshape(
            colourImg.size + (4,)
        )

        indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
        allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 6))
        df2 = pd.DataFrame(
            allArray, columns=["y", "x", "blue", "green", "red", "alpha"]
        ).copy()
    elif mode == "L" or mode == "1":
        colourArray = np.array(colourImg.getdata(), dtype=np.uint8).reshape(
            colourImg.size + (1,)
        )
        indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
        allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 3))
        df2 = pd.DataFrame(allArray, columns=["y", "x", "bw"]).copy()
    elif mode == "LA":
        colourArray = np.array(colourImg.getdata(), dtype=np.uint8).reshape(
            colourImg.size + (2,)
        )
        indicesArray = np.moveaxis(np.indices(colourImg.size), 0, 2)
        allArray = np.dstack((indicesArray, colourArray)).reshape((-1, 4))
        df2 = pd.DataFrame(allArray, columns=["y", "x", "bw", "alpha"]).copy()
    if not df2.empty:
        df2 = optimize_only_int(df2,verbose=False)

    return df2


def convert2np(dframe):
    df = dframe

    maxlen = df.x.max()
    convertedimage = None

    if df.shape[1] == 6:
        blue = np.array(np.array_split(df.blue.to_numpy(), maxlen + 1))

        green = np.array(np.array_split(df.green.to_numpy(), maxlen + 1))
        red = np.array(np.array_split(df.red.to_numpy(), maxlen + 1))
        alpha = np.array(np.array_split(df.alpha.to_numpy(), maxlen + 1))
        convertedimage = np.dstack((blue, green, red, alpha))
    elif df.shape[1] == 5:
        blue = np.array(np.array_split(df.blue.to_numpy(), maxlen + 1))

        green = np.array(np.array_split(df.green.to_numpy(), maxlen + 1))
        red = np.array(np.array_split(df.red.to_numpy(), maxlen + 1))

        convertedimage = np.dstack((blue, green, red))
    elif df.shape[1] == 3:
        convertedimage = np.array(np.array_split(df.bw.to_numpy(), maxlen + 1))
    elif df.shape[1] == 4:
        bw = np.array(np.array_split(df.bw.to_numpy(), maxlen + 1))
        alpha = np.array(np.array_split(df.alpha.to_numpy(), maxlen + 1))
        convertedimage = np.dstack((bw, alpha))
    return convertedimage


def print_full_col(text, colour):
    return "".join(
        list(
            map(
                str,
                (
                    fx.bold,
                    bg.brightwhite,
                    fg.brightwhite,
                    rgb256(colour[0], colour[1], colour[2]),
                    text,
                    bg.brightwhite,
                    fg.brightwhite,
                    fx.bold,
                    fx.reset,
                ),
            )
        )
    )


def console_plot(dframe, sizepic=70, *args, **kwargs):
    dfxa = dframe
    fig = dfxa.plot(*args, **kwargs).get_figure()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    imagetoconvert = Image.fromarray(data)
    imagetoconvert = imagetoconvert.resize(
        (sizepic * 3, int(sizepic / imagetoconvert.size[0] * imagetoconvert.size[1]))
    )
    df = image2df(image=imagetoconvert, convert="RGB")
    df = rename_columns(df, red="blue", blue="red")
    setwiederholen = "█"

    df["bw"] = df.apply(
        lambda x: reduce(
            lambda x_, y_: operator.add(str(x_), str(y_)),
            (
                bg.white,
                fg.white,
                rgb256(x.red, x.green, x.blue),
                setwiederholen,
                bg.white,
                fg.white,
                fx.reset,
            ),
        ),
        axis=1,
    )

    df4 = df.drop(columns=["red", "green", "blue"])
    pd.DataFrame(convert2np(dframe=df4))

    print(
        pd.DataFrame(convert2np(dframe=df4))
        .to_string(
            col_space=1, index=False, header=False, max_colwidth=1, justify="center"
        )
        .replace(" ", "")
    )
    z = plt.legend()
    tm = z.legendHandles
    print("")
    for ini, axa in enumerate(plt.gcf().get_axes()):
        labl = print_full_col("labels  ", (0, 120, 0))
        print(labl, end=" ")

        for colour_, label_ in zip(tm, (axa.get_legend_handles_labels()[-1])):
            colo = np.array(
                [x * 255 for x in colour_.get_facecolor()[:3]], dtype="uint8"
            )
            text = label_
            try:
                forjust = dfxa.columns.map(len).max() + 1
            except:
                forjust = 5
            print(print_full_col(f" {str(text).ljust(forjust)}", colo), end=" ")

        print("")
        yax = print_full_col("y-axis  ", (0, 0, 255))
        print(yax, end=" ")
        for ay_ in [yas.__dict__.get("_text") for yas in axa.get_yticklabels()]:
            print((print_full_col(str(f"{ay_}").ljust(10), (130, 0, 130))), end="")
        print("")
        xax = print_full_col("x-axis  ", (255, 0, 0))
        print(xax, end=" ")
        for ax_ in [xas.__dict__.get("_text") for xas in axa.get_xticklabels()]:
            print(print_full_col(str(f"{ax_}").ljust(10), (0, 0, 0)), end="")
    del fig
    return


def pd_add_console_plot():
    DataFrame.ds_console_plot = console_plot
    Series.ds_console_plot = console_plot


