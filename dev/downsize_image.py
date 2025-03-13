import cv2

def downsize_image(image_path, down_proc=None, size=None):
    image = cv2.imread(image_path)

    image = cv2.resize(image, size)

    new_path = image_path.replace(".png", "_down.png")
    
    cv2.imwrite(new_path, image)

downsize_image(image_path="data/sanity/input/video_0153.png", size=(420, 280))