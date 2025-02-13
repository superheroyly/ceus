'''
Author: LiuyiYang 1183140624@qq.com
Date: 2025-02-13 10:17:41
LastEditors: LiuyiYang 1183140624@qq.com
LastEditTime: 2025-02-13 18:57:08
FilePath: \CEUS\github-code\ceus\test.py
Description: 尝试计算每一帧各个连通域的像素强度值
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_p = r"D:\Vscode_code\Python\CEUS\animal\threshold-continual\threshold_100\Ceus-outlier-scatter\seg_no_registra\no_stack_scattered"
image_files = sorted(os.listdir(image_p), key=lambda x: int(os.path.splitext(x)[0]))

class bbox:
    def __init__(self, x, y, w, h, frame=None, intensity=0, update_times=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame
        self.intensity = intensity
        self.update_times = update_times
        self.pointer = None  # Add a pointer attribute

    def set_frame(self, frame):
        self.frame = frame

    def set_intensity(self, intensity):
        self.intensity = intensity

    def set_update_times(self, update_times):
        self.update_times = update_times

    def set_pointer(self, pointer):
        self.pointer = pointer  # Method to set the pointer

    def __repr__(self):
        return f"bbox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, frame={self.frame}, intensity={self.intensity}, update_times={self.update_times})"

class bbox_top:
    def __init__(self):
        self.pointer = None
        self.rear = None
        self.update_times = 0
        pass

    def set_pointer(self, pointer):
        self.pointer = pointer

    def set_rear(self, rear):
        self.rear = rear

class BBoxChain:
    def __init__(self):
        self.chain = []

    def add_bbox(self, bbox):
        self.chain.append(bbox)

    def get_chain(self):
        return self.chain

    def calculate_total_intensity(self):
        return sum(bbox.intensity for bbox in self.chain if hasattr(bbox, 'intensity'))

    def __repr__(self):
        return f"BBoxChain with {len(self.chain)} bboxes"

    def print_chain(self):
        for b in self.chain:
            print(b)
            while b.pointer is not None:
                b = b.pointer
                print(f" -> {b}")


if __name__ == '__main__':
    bbox_chain = BBoxChain()

    for i, img in enumerate(image_files):
        img_name = os.path.splitext(img)[0]

        img_path = os.path.join(image_p, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if cv2.countNonZero(img) == 0:
            continue

        # Find contours in the image
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours
        for contour in contours:
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)

            # Check for overlap with existing bounding boxes
            overlap_found = False

            for b in bbox_chain.get_chain():
                b_top = b
                b = b_top.rear
                x1, y1, w1, h1 = b.x, b.y, b.w, b.h
                # Calculate the intersection area
                x_overlap = max(0, min(x + w, x1 + w1) - max(x, x1))
                y_overlap = max(0, min(y + h, y1 + h1) - max(y, y1))
                intersection_area = x_overlap * y_overlap
                # Calculate the union area
                union_area = w * h + w1 * h1 - intersection_area
                # Calculate the overlap ratio
                overlap_ratio = intersection_area / union_area

                if x >= x1 and y >= y1 and x + w <= x1 + w1 and y + h <= y1 + h1:
                    overlap_found = True
                    break

                if overlap_ratio > 0.5 and not (x >= x1 and y >= y1 and x + w <= x1 + w1 and y + h <= y1 + h1):
                    overlap_found = True
                    # Update the bounding box
                    new_bbox = bbox(
                        min(x, x1), min(y, y1),
                        max(x + w, x1 + w1) - min(x, x1),
                        max(y + h, y1 + h1) - min(y, y1)
                    )
                    new_bbox.set_frame(img_name)
                    new_bbox.set_intensity(np.sum(img[y:y+h, x:x+w]))
                    new_bbox.set_update_times(b.update_times + 1)
                    b.set_pointer(new_bbox)
                    b_top.set_rear(new_bbox)
                    b_top.update_times += 1
                    break

            if not overlap_found:
                # Calculate the intensity of the current region
                new_bbox = bbox(x, y, w, h)
                new_bbox.set_frame(img_name)
                new_bbox.set_intensity(np.sum(img[y:y+h, x:x+w]))
                new_bbox.set_update_times(1)

                new_bbox_top = bbox_top()
                new_bbox_top.set_pointer(new_bbox)
                new_bbox_top.set_rear(new_bbox)
                new_bbox_top.update_times = 1
                bbox_chain.add_bbox(new_bbox_top)

    # 打印链表中的所有 bbox 实例
    bbox_chain.print_chain()
    
    # Find the top 3 bbox chains with the longest update_times
    top_bboxes = sorted(bbox_chain.get_chain(), key=lambda b: b.update_times, reverse=True)[:3]
    # print(top_bboxes)

    # Print the top 3 bbox chains
    # for i, b in enumerate(top_bboxes):
    #     print(f"Top {i+1} bbox chain:")
    #     while b is not None:
    #         print(b)
    #         b = b.pointer
    #     print("--" * 10)

   

    # Function to extract frame numbers from bbox chain
    def extract_frame_numbers(bbox_chain):
        x, y = [], []
        i = 0
        while bbox_chain is not None:
            if i == 0:
                i += 1 
                bbox_chain = bbox_chain.pointer
                continue
            x.append(int(bbox_chain.frame))
            y.append(float(bbox_chain.intensity))  # 将强度值转换为浮点数
            bbox_chain = bbox_chain.pointer
        return x, y


    # Extract frame numbers for the top 3 bbox chains
    top_frames_intensities = [(extract_frame_numbers(b)) for b in top_bboxes]

    # Plot the frame numbers and intensities for the top 3 bbox chains
    plt.figure(figsize=(10, 6))
    for i, (frames, intensities) in enumerate(top_frames_intensities):
        plt.plot(frames, intensities, label=f'Top {i+1} bbox chain')

        # Calculate the slopes between consecutive points with a fixed distance of 1
        slopes = []
        for k in range(len(intensities) - 1):
            dx = 1
            dy = int(intensities[k+1] - intensities[k])
            slopes.append(dy / dx)
 
        
        # Print and mark the slopes on the plot
        # if i == 0:  # Only plot and mark slopes for the first bbox chain
        #     for j in range(len(slopes)):
        #         print(f"slope-{j}: {slopes[j]:.1f}")
        #         plt.text(frames[j + 1], intensities[j + 1], f'{slopes[j]:.1f}', fontsize=6, color='blue')
        # Find the index of the maximum positive slope
        max_slope_index = np.argmax(slopes)
        
        # Mark the point with the maximum slope in red
        plt.scatter(frames[max_slope_index + 1], intensities[max_slope_index + 1], color='red')

    plt.xlabel('Frame Number')
    plt.ylabel('Intensity')
    plt.title('Top 3 BBox Chains Frame Numbers and Intensities')
    plt.legend()
    plt.show()


    # print('--'*10)
    # # Sort the bounding boxes by their count values in descending order
    # sorted_bboxes = sorted(bboxes.items(), key=lambda item: item[1], reverse=True)
    # # Select the top 3 bounding boxes
    # top_bboxes = sorted_bboxes[:3]

    # # Create a blank image to overlay all bounding boxes
    # blank_image = np.zeros_like(img)


    # save_p = r"D:\Vscode_code\Python\CEUS\animal\threshold-continual\threshold_100\Ceus-outlier-scatter\seg_no_registra\separate_region"
    # for bi in bbox_chain.get_chain():
    #     print(f"Bounding box: {bi.x}, {bi.y}, {bi.w}, {bi.h}")
    #     while bi.pointer is not None:
    #         bi = bi.pointer        

    #     print(f"Bounding box2: {bi.x}, {bi.y}, {bi.w}, {bi.h}")

    #     continue 


    #     img = bboxes_img[bi]
        

    #     x, y, w, h = b.x, b.y, b.w, b.h
    #     img_with_bbox = cv2.rectangle(
    #         img.copy(), (x, y), (x + w, y + h), (25, 100, 50), 1
    #     )

    #     save_path = os.path.join(save_p, f'bbox_{x}_{y}_{w}_{h}.png')
    #     cv2.imwrite(save_path, img_with_bbox)
    #     # cv2.imshow('Bounding Box', img_with_bbox)
    #     # cv2.waitKey(0)
    #     # Save each image with bounding box
    #     # save_path = os.path.join(image_p, f'bbox_{x}_{y}_{w}_{h}.png')
    #     # cv2.imwrite(save_path, img_with_bbox)
        
    #     # Overlay the bounding box on the blank image only if the current pixel is zero
    #     for i in range(blank_image.shape[0]):
    #         for j in range(blank_image.shape[1]):
    #             if blank_image[i, j] == 0:
    #                 blank_image[i, j] = img_with_bbox[i, j]

    # # Display the final image with all bounding boxes
    # save_path = os.path.join(save_p, 'all_bboxes.png')
    # cv2.imwrite(save_path, blank_image)
    # # cv2.imshow('All Bounding Boxes', blank_image)
    # # cv2.waitKey(0)
    # cv2.destroyAllWindows()
