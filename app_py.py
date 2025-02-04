import streamlit as st
import cv2
import numpy as np
from PIL import Image

def sift_match_and_homography(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Reduce threshold to avoid incorrect matches
    good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
    
    st.write(f"Number of good matches: {len(good_matches)}")

    if len(good_matches) > 4:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        
        st.write(f"Homography Matrix:\n{H}")

        return H, good_matches, kp1, kp2
    return None, [], None, None

def stitch_images(img1, img2):
    H, matches, kp1, kp2 = sift_match_and_homography(img1, img2)
    
    if H is not None:
        height, width = img2.shape[:2]
        
        # Increase canvas size to avoid clipping
        result = cv2.warpPerspective(img1, H, (width * 3, height * 3))

        # Place second image onto the result safely
        result[:img2.shape[0], :img2.shape[1]] = img2

        return result, matches, kp1, kp2
    
    return None, [], None, None

def main():
    st.title("Image Stitching with SIFT and Homography")
    uploaded_files = st.file_uploader("Upload two images", type=["jpg", "png"], accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) == 2:
        image1 = Image.open(uploaded_files[0])
        image2 = Image.open(uploaded_files[1])
        
        img1 = np.array(image1.convert('RGB'))
        img2 = np.array(image2.convert('RGB'))
        
        result, matches, kp1, kp2 = stitch_images(img1, img2)
        
        if result is not None:
            st.image(result, caption="Stitched Image", use_column_width=True)
        else:
            st.error("Could not stitch images. Try different images.")

if __name__ == "__main__":
    main()
