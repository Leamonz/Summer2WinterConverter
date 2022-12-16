import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox as msgBox
import os

import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils
from PIL import Image, ImageTk
from generator import Generator
import config
import time

window = tk.Tk()
window.title('Summer2WinterConverter')
window.geometry('812x512')
gen = Generator(config.IMG_CHANNELS)
checkpoint = torch.load('models/GenWinter.pth', map_location=config.DEVICE)
gen.load_state_dict(checkpoint["state_dict"])
# transform 进行大小、类型的转换
resize = transforms.Resize([256, 256])
toTensor = transforms.ToTensor()
toPILImage = transforms.ToPILImage()
# 显示图片需要用到的变量
srcImage = None
destImage = None
pImage1 = None
pImage2 = None


def selectSourceImage():
    global lCanvas, srcImage, pImage1
    entry1.delete(0, 'end')
    filepath = filedialog.askopenfilename(title='选择输入图片',
                                          filetypes=[('png files', '*.png'),
                                                     ('jpg files', '*.jpg'),
                                                     ('All Files', '*')])
    entry1.insert('end', filepath)
    srcImage = Image.open(filepath).convert("RGB")
    srcImage = resize(srcImage)
    pImage1 = ImageTk.PhotoImage(srcImage)
    lCanvas.create_image(100, 22, anchor='nw', image=pImage1)


def selectDestinationImage():
    entry2.delete(0, 'end')
    filepath = filedialog.asksaveasfilename(title='选择保存图片位置',
                                            filetypes=[('png files', '*.png')])
    entry2.insert('end', filepath)


def saveImage():
    filepath = entry2.get()
    if filepath is None:
        msgBox.showerror('错误', '请先选择保存路径!')
    else:
        if filepath[-3:] != '.png':
            filepath += '.png'
        destImage.save(filepath)
        msgBox.showinfo('提示', '保存成功')


def process():
    global srcImage, destImage, rCanvas, pImage2, gen
    if srcImage is None:
        msgBox.showerror('错误', '请先选择源图片!')
    else:
        with torch.no_grad():
            input = np.asarray(srcImage)
            augs = config.test_transforms(image=input)
            input = augs['image']
            destImage = gen(input)
            destImage = 0.5 * destImage + 0.5
            destImage = toPILImage(destImage)
            pImage2 = ImageTk.PhotoImage(destImage)
            rCanvas.create_image(0, 22, anchor='nw', image=pImage2)


def closeThisWindow():
    msgBox.showinfo('提示', '退出程序')
    window.destroy()


label1 = tk.Label(window, text="请选择输入图片:")
entry1 = tk.Entry(window, width=75, font=('Arial', 8))
selectSrcButton = tk.Button(window, text='浏览', width=8, command=selectSourceImage)
label2 = tk.Label(window, text='请选择保存路径:')
entry2 = tk.Entry(window, width=75, font=('Arial', 8))
selectDestButton = tk.Button(window, text='浏览', width=8, command=selectDestinationImage)

label1.pack()
entry1.pack()
selectSrcButton.pack()
label2.pack()
entry2.pack()
selectDestButton.pack()
label1.place(x=50, y=30)
entry1.place(x=150, y=30)
selectSrcButton.place(x=630, y=25)
label2.place(x=50, y=65)
entry2.place(x=150, y=65)
selectDestButton.place(x=630, y=60)

lCanvas = tk.Canvas(window, width=356, height=300)
lCanvas.pack()
lCanvas.place(x=0, y=140)
rCanvas = tk.Canvas(window, width=356, height=300)
rCanvas.pack()
rCanvas.place(x=456, y=140)

processButton = tk.Button(window, text='转换', width=15, height=2,
                          font=('Arial', 12), command=process)
saveButton = tk.Button(window, text='保存', width=15, height=2,
                       font=('Arial', 12), command=saveImage)
closeButton = tk.Button(window, text='关闭', width=15, height=2,
                        font=('Arial', 12), command=closeThisWindow)
processButton.pack()
saveButton.pack()
closeButton.pack()
processButton.place(x=170, y=450)
saveButton.place(x=340, y=450)
closeButton.place(x=510, y=450)

if __name__ == "__main__":
    window.mainloop()
