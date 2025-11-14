import math
import numpy as np
import cv2 as cv

def detect_rock(img: np.ndarray) -> list[dict]:
    height, width = img.shape[:2]

    # Step 2: Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Step 3: Apply preprocessing for better rock detection
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(blurred, 30, 100)

    # Step 4: Morphological operations to connect edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=1)
    edges = cv.erode(edges, kernel, iterations=1)

    # Step 5: Find contours
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Step 6: Filter and collect valid rocks
    valid_rocks = []

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        area = cv.contourArea(cnt)

        # Filter criteria
        min_area = 50
        max_area = 50000
        min_height = height * 0.3

        if (area > min_area and area < max_area and 
            y > min_height):
            
            aspect_ratio = w / h if h > 0 else 0
            
            if 0.3 < aspect_ratio < 3.0:
                if y > 400:
                    valid_rocks.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w//2, y + h//2),
                        'distance': math.pow((((x + w//2) - 250)**2 + ((y + h//2) - 480)**2),0.5)
                    })

    return valid_rocks

def direction_manuver(valid_rocks: dict)-> list:
    if not valid_rocks:
       return [0.0]
    else:
        dir = None
        distance = 0
        for rocks in valid_rocks:
            if rocks['distance'] > distance:
                distance = rocks['distance']
                if rocks['center'][0] < 250:
                    dir = -0.2
                else:
                    dir = 0.2
            
        # return success, rock info and simple maneuver guidance
        return [1.0, dir]