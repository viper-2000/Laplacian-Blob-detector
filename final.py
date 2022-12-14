import numpy as np
import cv2
import math
import cProfile
from termcolor import colored
import time

link = r"C:\Users\karth\Desktop\GRE\NCSU\SEMESTER1\ECE558_Digital_Imaging_Tianfu_Wu\Project\Project03\TestImages4Project\\"
images = ["demo1.jpeg","demo2.jpeg","demo3.jpeg","demo4.jpeg","butterfly.jpg","sunflowers.jpg","fishes.jpg","einstein.jpg"]
suffix = ""

Sigma = 1.15
N = 15
K = 1.24
threshold = 1
KernelSizes =[]
Sigmas=[]
LoGKs = []
DoGKs = []
ConvImages = []


def showImage(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def readImage(link):
    Image = cv2.imread(link)
    showImage("Color",Image)
    Image = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    showImage("Gray",Image)
    return Image
    
def getParameters(N,K,Sigma):
    #global Sigmas,KernelSizes
    for i in range(N):
        Sigmas.append(Sigma*(K**i))
    for i in range(N):
      n = round(Sigmas[i]*6)
      if(n%2==0):
        KernelSizes.append(n+1)
      else:
        KernelSizes.append(n)
    return Sigmas,KernelSizes

def G(x,y,k,sigma):
    ret = (1/(np.pi*2*(k*sigma)**2)) * math.exp(-(x**2+y**2)/(2* (sigma**2)* k**2))
    return ret

def DoGKernels(N,k):
    for n in range(N):
        sigma = Sigmas[n]
        kernelSize = KernelSizes[n]
        DoG = np.empty((kernelSize,kernelSize))
        for x in range(int(-math.floor(kernelSize/2)),int(math.floor(kernelSize/2))+1):
            for y in range (int(-math.floor(kernelSize/2)),int(math.floor(kernelSize/2))+1):
                DoGk = G(x,y,k,sigma)
                DoGs = G(x,y,1,sigma)
                DoG[x+int(math.floor(kernelSize/2))][y+int(math.floor(kernelSize/2))] = DoGk - DoGs
        DoGKs.append(DoG)
    return DoGKs

def LoGKernels(N):
    #global LoGKs
    for n in range(N):
        sigma = Sigmas[n]
        kernelSize = KernelSizes[n]
        LoG = np.empty((kernelSize,kernelSize))
        for x in range(int(-math.floor(kernelSize/2)),int(math.floor(kernelSize/2))+1):
            for y in range (int(-math.floor(kernelSize/2)),int(math.floor(kernelSize/2))+1):
                LoG[x+int(math.floor(kernelSize/2))][y+int(math.floor(kernelSize/2))] = (-1/(math.pi*(sigma**2)))*(1-(((x**2)+(y**2))/(2*(sigma**2))))*np.exp(-((x**2)+(y**2))/(2*(sigma**2)))
        LoGKs.append(LoG)
    return LoGKs

def padder(Image,Kernel):
    sx = (Kernel.shape[0]-1)//2
    sy = (Kernel.shape[1]-1)//2

    ax,ay=0,0
    if Kernel.shape[0]%2==0:
        ax=1
    if Kernel.shape[1]%2==0:
        ay=1

    padded_img = np.zeros((Image.shape[0]+sx+sx+ax,Image.shape[1]+sy+sy+ay))

    startxi = 0 + sx
    endxi = padded_img.shape[0] - sx - 1 - ax

    startyi = 0 + sy
    endyi = padded_img.shape[1] - sy - 1 - ay

    padded_img[startxi:endxi+1,startyi:endyi+1] = Image

    return padded_img

# convolution optimised from Project 2:
# the optimisation is implementation of numpy arrays which perform array manipulations much faster
def conv2(Image,Kernel):
    imgw=0

    paddedImage = padder(Image,Kernel)

    ret = np.zeros((Image.shape[0],Image.shape[1]))

    sx = (Kernel.shape[0]-1)//2
    sy = (Kernel.shape[1]-1)//2

    ax,ay=0,0
    if Kernel.shape[0]%2==0:
        ax=1
    if Kernel.shape[1]%2==0:
        ay=1

    startxi = 0 + sx
    endxi = paddedImage.shape[0] - sx - 1 - ax

    startyi = 0 + sy
    endyi = paddedImage.shape[1] - sy - 1 - ay   

    for i in range(startxi,endxi+1):
        for j in range(startyi, endyi+1):
            imgw = paddedImage[i-sx:i+sx+1+ax, j-sx:j+sx+1+ay]
            M = np.sum(np.multiply(imgw,Kernel))
            ret[i-startxi][j-startyi] = M

    return ret

def convAll(Image,allKernels):
    #global ConvImages
    for i in range(len(allKernels)):
        kernel = allKernels[i]
        outp = conv2(Image,kernel)
        ConvImages.append(outp)
    return ConvImages

def compare(A,B,i,j):
    global Image
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1),(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return all(A[i + dx, j + dy] < B[i, j] for dx, dy in neighbors if  0<= i + dx < Image.shape[0] and 0<= j + dy <Image.shape[1])

def nonMaxSupression(ImageI,ConvImages, Sigmas, N,Th):
    #global Outp
    radii = np.empty(N)
    for i in range(0,N):
        radii[i]=int(np.ceil(np.sqrt(2)*Sigmas[i]))
    blobCoordinates =[]
    for k in range(N):
        Rows = ConvImages[k].shape[0]
        Cols = ConvImages[k].shape[1]
        for i in range(int(radii[k]),Rows-int(radii[k])):
            for j in range(int(radii[k]),Cols-int(radii[k])):
                if ConvImages[k][i,j] > Th:
                    maxCenter = compare(ConvImages[k],ConvImages[k],i,j)
                    maxLeft = True
                    maxRight = True
                    if k - 1 >= 0:
                        maxLeft = compare(ConvImages[k-1],ConvImages[k],i,j) and ConvImages[k - 1][i, j] < ConvImages[k][i,j]
                    if k + 1 < N:
                        maxRight = compare(ConvImages[k+1],ConvImages[k],i,j) and ConvImages[k + 1][i, j] < ConvImages[k][i, j]
                    if maxCenter==True and maxLeft==True and maxRight==True:
                        blobCoordinates.append((i,j,k))

    Outp=cv2.imread(ImageI)
    for coordinates in blobCoordinates:
        radius = int(radii[coordinates[2]])
        Outp = cv2.circle(Outp,(coordinates[1],coordinates[0]),radius, (0,0,255))
    return Outp

def BlobDetector(Image,N,K,Sigma,threshold,i,logdog):
    start = time.time()
    print("Processing...")
    Sigmas,KernelSizes = getParameters(N,K,Sigma)
    #LoG Implementation
    if logdog==0:
        allKernels = LoGKernels(N)

    #DoG Implementation
    else:
        allKernels = DoGKernels(N,K)
    ConvImages = convAll(Image,allKernels)
    ret = nonMaxSupression(i,ConvImages,Sigmas,N,threshold)
    print("Done")
    end = time.time()
    print(f"Process completed in: {end-start:0.2f} seconds")
    return ret

def allImages():
    for i in range(len(images)):
        print(colored(f"Image: {images[i]}","green"))
        KernelSizes =[]
        Sigmas=[]
        LoGKs = []
        DoGKs = []
        ConvImages = []
        Image = readImage(i)

        final=""
        cProfile.run(r"final = BlobDetector(Image,N,K,Sigma,threshold,i)")

        showImage("Final",final)

        cv2.imwrite(link+images[i]+"_blob.jpg"+suffix,final)

#allImage()
def getUserInputs():
    imagePath = input(f"Enter full-path of image:\n")
    global KernelSizes , Sigmas, LoGKs , DoGKs , ConvImages , Image

    logdog = -1

    while(logdog<0 or logdog>1):
        logdog = int(input(f"\nEnter\n0. for LoG\n1. for DoG:\n"))

    Image=""
    Image = readImage(imagePath)
    print(colored(f"Image:\n{imagePath}","green"))
    final=""
    final = BlobDetector(Image,N,K,Sigma,threshold,imagePath,logdog)
    #cProfile.run(r"final = BlobDetector(Image,N,K,Sigma,threshold,i,logdog)")
    blob = "LoG"
    if logdog==1:
        blob = "DoG"
    print(colored(f"{blob} blob detection complete."))
    showImage("Final",final)

getUserInputs()