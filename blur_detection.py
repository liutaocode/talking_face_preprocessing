import cv2

# code is modified from https://pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
def getImageVar(imgPath):
    image = cv2.imread(imgPath)

    height, width, _ = image.shape
    
    # Center area will be calculated
    start_row, start_col = int(height * .25), int(width * .25)
    end_row, end_col = int(height * .75), int(width * .75)

    image = image[start_row:end_row , start_col:end_col]

    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F).var()
    return imageVar

image_path = '/path/to/image.png'
blur_threshold = 20

imageVar = getImageVar(image_path)
if imageVar < blur_threshold:
    print(f'{image_path} is blurry')