# -*- coding: utf-8 -*-
from tkinter import *
import tkFileDialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import plotly.express as px
import plotly.graph_objects as go
import scipy.misc


class Window(Frame):


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()


    # initiating the application window and menu
    def init_window(self):
        self.master.title("Algorytm K-Średnich")
        self.pack(fill=BOTH, expand=1)

        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Wgraj zdjęcie", command=self.foto)
        file.add_command(label="O algorytmie", command=self.info)
        file.add_command(label="Wyjście", command=self.client_exit)

        menu.add_cascade(label="Menu", menu=file)
        w = Label(self.master, text="Wizualizacja działania metody K-Średnich",font=("Helvetica", 16))
        w.place(x=205, y=10)


    # window with information about algorithm
    def info(self):
        path = r'info.jpg'
        image = cv2.imread(path)
        window_name = "Informacje o algorytmie"
        cv2.imshow(window_name, image)


    # exiting application
    def client_exit(self):
        exit()



    # getting image from user folder and displaying it
    def foto(self):

        # getting path of an image
        path = tkFileDialog.askopenfilename()
        img = Image.open(path)

        # changing the size of an image
        size=img.size
        width = size[0]
        ratio=width/225
        heigth=size[1]/ratio
        img=img.resize((225,heigth))
        img_send = np.array(img)

       # displaying image on window
        render = ImageTk.PhotoImage(img)
        img = Label(self, image=render)
        img.grid(row=20, column=0, sticky=W, pady=2)
        img.image = render
        img.place(x=50, y=80)

        l1 = Label(self.master, text = "K = ")
        l1.place(x=80, y=40)

        # pixel graph of the input image
        def plotIn(img):

            # OpenCV split() - divides the image into component channels
            r, g, b = cv2.split(img)

            # spliting the image and configuration of the 3D graph
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1, projection="3d")

            # setting the color of pixels
            pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
            norm = colors.Normalize(vmin=-1., vmax=1.)

            # Colors to flatten them into a list and normalize them so that they can be passed to the facecolors parameter of Matplotlib.scatter()
            norm.autoscale(pixel_colors)
            pixel_colors = norm(pixel_colors).tolist()
            axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")

            axis.set_xlabel("Red")
            axis.set_ylabel("Green")
            axis.set_zlabel("Blue")

            plt.show()

        # k-means algorithm
        def onClicked():

            # getting the input value of K
            k = entry.get()

            Z = img_send.reshape((-1, 3))
            Z = np.float32(Z)

            # determining the criteria of the algorithm and executing it
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = int(k)
            ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((img_send.shape))

            # output plot of clustered image
            def plotOut(Z, center):
                fig = plt.figure()
                axis = fig.add_subplot(1, 1, 1, projection="3d")
                axis.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=label[:, 0], cmap='viridis', alpha=0.05)
                axis.scatter(center[:, 0], center[:, 1], center[:, 2], s=500, c='black', alpha=1)
                axis.set_xlabel("Red")
                axis.set_ylabel("Green")
                axis.set_zlabel("Blue")
                plt.show()

            # output plot of clustered image - online using Plotly
            def plotOnline(Z, center):

                # pixel visualization
                figure1 = go.Scatter3d(x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],name = 'pixels', mode='markers', marker=dict(
                    color=label[:, 0],
                    size=1.5,
                    symbol='circle'
                )
                                       )

                # centroids visualization
                figure2 = go.Scatter3d(x=center[:, 0], y=center[:, 1], z=center[:, 2], name = 'centroids', mode='markers', marker=dict(
                    size=50,
                    color='rgb(127,127,127)',
                    symbol='circle'
                ), opacity=0.7
                                       )

                data = [figure1, figure2]
                layout = go.Layout()
                figure = go.Figure(data, layout)
                figure.show()

            buttonPlotOut = Button(self.master, text="Wykres obrazu wyjściowego", command= lambda: plotOut(Z,center))
            buttonPlotOut.place(x=300, y=170)

            buttonPlotOnline = Button(self.master, text="Wykres obrazu wyjściowego online", command= lambda: plotOnline(Z,center))
            buttonPlotOnline.place(x=300, y=200)

            # displaying the output image
            res2=Image.fromarray(res2)
            render2 = ImageTk.PhotoImage(res2)

            res2 = Label(self, image=render2)
            res2.grid(row=20, column=0, sticky=W, pady=2)
            res2.image = render2
            res2.place(x=500, y=80)

        entry = Entry(self.master)
        entry.place(x=110, y=45)
        entry.insert(0, "4")

        button = Button(self.master, text="OK", command=lambda: onClicked())
        button.place(x=250, y=43)

        buttonPlotIn = Button(self.master, text="Wykres obrazu wejsciowego", command=lambda: plotIn(img_send))
        buttonPlotIn.place(x=300, y=140)

        label = Label(root)
        label.pack()

root = Tk()

root.geometry("800x400")

app = Window(root)

root.mainloop()  