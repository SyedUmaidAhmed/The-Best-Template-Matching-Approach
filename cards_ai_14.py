import cv2 
import imutils
import numpy as np
import matplotlib.pyplot as plt
cap = cv2.VideoCapture(1)

color=(255,0,0)
thickness=2

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 15, 15)
    edges = cv2.Canny(blur, 50, 150, True)
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(edges, kernel, iterations=1)
    _,contours, hierarchy = cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        if cv2.contourArea(contour) > 1500:
            x,y,w,h = cv2.boundingRect(contour)
            rect = cv2.minAreaRect(contour)
            ((x,y),(w,h),a) = rect
            
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(frame,[box],0,(0,0,255),2)

            width = int(rect[1][0])
            height = int(rect[1][1])
           

            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],[0, 0],[width-1, 0],[width-1, height-1]], dtype="float32")


            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(frame, M, (width, height))

            cv2.imwrite("Region.jpg", warped)

            
            cv2.rectangle(warped, (3, 2), (24, 50), (255,0, 0), 2)

            img1 = cv2.imread("crop.jpg", cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread("Region.jpg", cv2.IMREAD_GRAYSCALE)

            sift = cv2.xfeatures2d.SIFT_create()


            kp1, des1 = sift.detectAndCompute(img1, None)

            kp2, des2 = sift.detectAndCompute(img2, None)

            FLANN_INDEX_KDTREE = 0

            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

            search_params = dict(checks=50)   # or pass empty dictionary

            flann = cv2.FlannBasedMatcher(index_params,search_params)

            matches = flann.knnMatch(des1,des2,k=2)

            matchesMask = [[0,0] for i in range(len(matches))]

            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
   

            draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),matchesMask = matchesMask, flags = 0)

            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
            print(matchesMask)

            cv2.imshow("Img3", img3)         

            cv2.imshow("Region.jpg", warped)
            
    cv2.imshow("Edges", edges)
    cv2.imshow("Dilate", dilate)
    cv2.imshow("gray", gray)
    cv2.imshow("blur", blur)
    cv2.imshow("Processed Frame",frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):        
        break

cap.release()
cv2.destroyAllWindows()
