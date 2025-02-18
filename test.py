'''
Author: LiuyiYang 1183140624@qq.com
Date: 2025-02-13 10:17:41
LastEditors: yangliuyi liuyi.yang@wuerzburg-dynamics.com
LastEditTime: 2025-02-18 15:27:27
FilePath: \CEUS\github-code\ceus\test.py
Description: 尝试计算每一帧各个连通域的像素强度值
'''

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from utils import create_video_from_images_with_background
import shutil



# Define a class to represent a bounding box
class bbox:
    def __init__(self, x, y, w, h, frame=None, intensity=0, update_times=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame = frame
        self.intensity = intensity
        self.update_times = update_times

    def set_frame(self, frame):
        self.frame = frame

    def set_intensity(self, intensity):
        self.intensity = intensity

    def set_update_times(self, update_times):
        self.update_times = update_times

    def __repr__(self):
        return f"bbox(x={self.x}, y={self.y}, w={self.w}, h={self.h}, frame={self.frame}, intensity={self.intensity}, update_times={self.update_times})"


# Define a class to represent a list of bounding boxes
class BBoxList:
    def __init__(self):
        self.bboxes = []

    def add_top_bbox(self, new_bbox):
        top_bbox = bbox(x=new_bbox.x, y=new_bbox.y, w=new_bbox.w, h=new_bbox.h, frame=0, intensity=new_bbox.intensity, update_times=1)
        empty_list = [top_bbox, new_bbox]
        self.bboxes.append(empty_list)

    def add_bbox(self, bbox):
        self.bboxes.append(bbox)

    def get_bboxes(self):
        return self.bboxes
    
    def __repr__(self):
        return f"BBoxList with {len(self.bboxes)} bboxes"

    def print_bboxes(self):
        for b in self.bboxes:
            print(b)

    # Function to extract frame numbers from bbox list
    def extract_frame_numbers(self, b):
        x, y = [], []
        for bbox in b:
            x.append(int(bbox.frame))
            y.append(float(bbox.intensity))  # 将强度值转换为浮点数
        return x, y
    

def stack_images(fore, back_p, save_path=None):
    foreground = cv2.cvtColor(fore, cv2.COLOR_GRAY2BGR)
    background = cv2.imread(back_p)

    # Ensure the foreground and background images have the same dimensions
    if foreground.shape != background.shape:
        print(f"Foreground shape: {foreground.shape}, Background shape: {background.shape}")
        raise ValueError("Foreground and background images must have the same dimensions")
    
    # Create an output image by stacking the foreground and background with transparency
    stacked_image = cv2.addWeighted(foreground, 0.5, background, 1, 0)

    # Save the stacked image
    # cv2.imwrite(save_path, stacked_image)

    return stacked_image


def registra(moving_image, fixed_image):
    # 将图像转换为标量图像
    if fixed_image.GetNumberOfComponentsPerPixel() > 1:
        fixed_image = sitk.VectorIndexSelectionCast(fixed_image, 0)
    if moving_image.GetNumberOfComponentsPerPixel() > 1:
        moving_image = sitk.VectorIndexSelectionCast(moving_image, 0)
        
    # 将图像转换为 32 位浮点数，以确保配准方法支持的图像类型
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    # 创建一个配准方法对象
    registration_method = sitk.ImageRegistrationMethod()
    
    # 根据选择的方法设置初始变换
    transform = sitk.Euler2DTransform()

    # 设置图像的初始变换为基于重心的对齐(MOMENTS)
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image,  # 固定图像
        moving_image,  # 移动图像
        transform,  # 选择的变换方法
        sitk.CenteredTransformInitializerFilter.MOMENTS   
    )

    # 设置初始变换
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # 设置度量标准为均方误差
    registration_method.SetMetricAsMeanSquares() 

    # 网格搜索进行一个最优化配准寻找
    best_metric_value = float('inf')
    best_learning_rate = 0.1
    best_number_of_iterations = 100

    for learning_rate in np.arange(0.02, 0.21, 0.02):
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=learning_rate,
            numberOfIterations=best_number_of_iterations,
            convergenceMinimumValue=1e-6,
            convergenceWindowSize=10
        )
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        try:
            registration_method.Execute(fixed_image, moving_image)
            current_metric_value = registration_method.GetMetricValue()
            if current_metric_value < best_metric_value:
                best_metric_value = current_metric_value
                best_learning_rate = learning_rate
        except Exception as e:
            print(f"Error with learning rate {learning_rate}")

    # 使用最佳参数重新设置优化器
    registration_method.SetOptimizerScalesFromPhysicalShift()  
    registration_method.SetOptimizerAsGradientDescent(  # 梯度下降
        learningRate=best_learning_rate,
        numberOfIterations=best_number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )

    # 设置插值方法为最邻近插值，对于二值mask不要用线性插值
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # 进行图像配准，执行配准方法
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # 输出变换矩阵在x、y轴上的分量值
    transform_parameters = final_transform.GetParameters()
    # print("变换参数:", transform_parameters)

    return final_transform


def apply_transform(transform, image, output_p=None):
    # 将图像转换为标量图像
    if image.GetNumberOfComponentsPerPixel() > 1:
        image = sitk.VectorIndexSelectionCast(image, 0)
    # 将图像转换为 32 位浮点数，以确保支持的图像类型
    image = sitk.Cast(image, sitk.sitkFloat32)

    # 确保变换和图像的维度匹配
    if transform.GetDimension() != image.GetDimension():
        print(f"Transform dimension: {transform.GetDimension()}, Image dimension: {image.GetDimension()}")
        raise ValueError("Transform and image dimensions do not match")
    
    # 将图像进行重采样
    resampled_image = sitk.Resample(
        image,  # 输入图像
        image,  # 参考图像
        transform,  # 变换矩阵
        sitk.sitkNearestNeighbor,  # 最邻近插值
        0.0,  # 默认像素值
        image.GetPixelID()  # 像素类型
    )
    
    # 将图像转换为无符号字符类型以确保兼容性
    resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt8)
    
    # 保存重采样后的图像
    # sitk.WriteImage(resampled_image, output_p)
    # print(f"Transformed image saved to {output_p}")
    return sitk.GetArrayFromImage(resampled_image)


def generate_video(image_p, ceus_mode_p, save_p, name='ouput_video1'):
    image_files = sorted(os.listdir(image_p), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    height, width = cv2.imread(os.path.join(image_p, image_files[0])).shape[:2]
    video_path = os.path.join(save_p, f'{name}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_p, img_file)
        frame_number = int(img_file.split('_')[-1].split('.')[0])   

        ceus_path = os.path.join(ceus_mode_p, str(frame_number) + '.png')
        stacked_image = stack_images(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), ceus_path, None)
        video.write(stacked_image)  

    video.release()
    print(f"Video saved to {video_path}")


def generate_video_with_two_images(image_p1, image_p2, fore_p, save_p, name='output_video2'):
    image_files1 = sorted(os.listdir(image_p1), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    image_files2 = sorted(os.listdir(image_p2), key=lambda x: int(x.split('_')[-1].split('.')[0]))
    fore_files = sorted(os.listdir(fore_p), key=lambda x: int(x.split('_')[-1].split('.')[0]))

    height1, width1 = cv2.imread(os.path.join(image_p1, image_files1[0])).shape[:2]
    height2, width2 = cv2.imread(os.path.join(image_p2, image_files2[0])).shape[:2]
    height_fore, width_fore = cv2.imread(os.path.join(fore_p, fore_files[0])).shape[:2]

    if height1 != height2 or width1 != width2 or height1 != height_fore or width1 != width_fore:
        raise ValueError("The dimensions of images in all directories must be the same")

    combined_height = height1
    combined_width = width1 + width2

    video_path = os.path.join(save_p, f'{name}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(video_path, fourcc, 5, (combined_width, combined_height))
    
    for f in fore_files:
        frame_number = int(f.split('_')[-1].split('.')[0])
        img_path1 = os.path.join(image_p1, f"{frame_number}.png")
        img_path2 = os.path.join(image_p2, f"{frame_number}.png")
        fore_path = os.path.join(fore_p, f)

        fore_img = cv2.imread(fore_path, cv2.IMREAD_GRAYSCALE)
        left = stack_images(fore_img, img_path1)
        right = stack_images(fore_img, img_path2)
        combine = np.hstack((left, right))
        
        video.write(combine)

    video.release()
    print(f"Video saved to {video_path}")


def process(b_mode_p, image_p, save_p, end_index=None):
    bbox_list = BBoxList()
    image_files = sorted(os.listdir(image_p), key=lambda x: int(os.path.splitext(x)[0]))
    
    seg_continual_img = None
    reg_continual_img = None

    for i, img in enumerate(image_files):
        img_name = os.path.splitext(img)[0]
        img_path = os.path.join(image_p, img)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # If the image is blank, skip to the next image
        if cv2.countNonZero(img) == 0:
            continue
        
        seg_full_image = np.zeros_like(img)
        reg_full_image = np.zeros_like(img)
        if i > end_index:
            shutil.copyfile(os.path.join(save_p, 'origin', f"{end_index}.png"), os.path.join(save_p, "origin", f'{img_name}.png'))
            shutil.copyfile(os.path.join(save_p, 'continual', f"{end_index}.png"), os.path.join(save_p, "continual", f'{img_name}.png'))
            transform = registra(sitk.ReadImage(rf"{b_mode_p}\{end_index}.png"),
                                 sitk.ReadImage(rf"{b_mode_p}\{img_name}.png"))
            registra_img = apply_transform(transform, sitk.GetImageFromArray(seg_continual_img))
            cv2.imwrite(os.path.join(save_p, "registra_continual", f'{img_name}.png'), registra_img)            
            continue
        
        # Find contours in the image
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop over the contours
        for contour in contours:
            # Get the bounding box for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Check for overlap with existing bounding boxes
            overlap_found = False

            for idx, b in enumerate(bbox_list.get_bboxes()):
                b = b[0]
                x1, y1, w1, h1 = b.x, b.y, b.w, b.h
                # Calculate the intersection area
                x_overlap = max(0, min(x + w, x1 + w1) - max(x, x1))
                y_overlap = max(0, min(y + h, y1 + h1) - max(y, y1))
                intersection_area = x_overlap * y_overlap
                # Calculate the union area
                union_area = w * h + w1 * h1 - intersection_area
                # Calculate the overlap ratio
                overlap_ratio = intersection_area / union_area

                # 如果新的bbox完全被包含在已有的bbox中，则skip
                if x >= x1 and y >= y1 and x + w <= x1 + w1 and y + h <= y1 + h1:
                    overlap_found = True
                    break

                if overlap_ratio > 0.3:
                    overlap_found = True
                    # 只有当前bbox的强度值大于已有bbox的强度值时，才更新bbox
                    if np.sum(img[y:y+h, x:x+w]) / 255 > b.intensity:
                        # Update the bounding box
                        new_bbox = bbox(min(x, x1), min(y, y1), max(x + w, x1 + w1) - min(x, x1), max(y + h, y1 + h1) - min(y, y1),
                                        img_name, np.sum(img[y:y+h, x:x+w]) / 255, b.update_times + 1)
                        b.update_times += 1
                        b.x, b.y, b.w, b.h = new_bbox.x, new_bbox.y, new_bbox.w, new_bbox.h
                        b.intensity = np.sum(img[y:y+h, x:x+w]) / 255
                        bbox_list.bboxes[idx].append(new_bbox)
                    break

            if not overlap_found:
                # Calculate the intensity of the current region
                new_bbox = bbox(x, y, w, h)
                new_bbox.set_frame(img_name)
                new_bbox.set_intensity(np.sum(img[y:y+h, x:x+w]) / 255)
                new_bbox.set_update_times(1)
                bbox_list.add_top_bbox(new_bbox)
    

        # Find the top X bboxes with the longest update_times
        if len(bbox_list.get_bboxes()) > 5:
            top_bboxes = sorted(bbox_list.get_bboxes(), key=lambda b: b[0].update_times, reverse=True)[:5]
        else:
            top_bboxes = bbox_list.get_bboxes()
        # Extract frame numbers
        top_frames_intensities = [(bbox_list.extract_frame_numbers(b)) for b in top_bboxes]

        # Plot the frame numbers and intensities
        # plt.figure(figsize=(10, 6))
        transform = None
        for i, (frames, intensities) in enumerate(top_frames_intensities):
            # plt.plot(frames, intensities, label=f'Top {i+1} bbox')

            # Calculate the slopes between consecutive points with a fixed distance of 1
            slopes = np.diff(intensities[1:])
            # Find the index of the maximum positive slope
            max_slope_index = 1
            if len(slopes) < 1:
                max_slope_index = 1
                continue
            elif len(slopes) == 1:
                if slopes[0] > 0:
                    max_slope_index = 2
                else:
                    max_slope_index = 1
            else:
                max_slope_index = np.argmax(slopes)+2
            # Mark the point with the maximum slope in red
            # plt.scatter(frames[max_slope_index + 1], intensities[max_slope_index + 1], color='red')

            temp = top_bboxes[i][max_slope_index]
            if temp.intensity < 400:
                continue

            # 把Top连通区域的关键帧截出来
            img = cv2.imread(rf"{image_p}\{temp.frame}.png")
            x, y, w, h = temp.x, temp.y, temp.w, temp.h

            # Get the cropped region
            blank_image_with_cropped = np.zeros_like(img)
            blank_image_with_cropped[y:y+h, x:x+w] = img[y:y+h, x:x+w]
            seg_full_image[y:y+h, x:x+w] = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

            # registra
            b_mode_moving_img = sitk.ReadImage(rf"{b_mode_p}\{temp.frame}.png") 
            b_mode_fixed_img = sitk.ReadImage(rf"{b_mode_p}\{img_name}.png")
            b_mode_moving_img.CopyInformation(b_mode_fixed_img)
            transform = registra(b_mode_moving_img, b_mode_fixed_img)
            
            # Apply the transform to the cropped region
            blank_image_with_cropped = sitk.GetImageFromArray(blank_image_with_cropped[:, :, 0])
            registra_img = apply_transform(transform, blank_image_with_cropped, f"{save_p}/registra_{x}_{y}_{w}_{h}_{temp.frame}.png")
            reg_full_image[registra_img > 0] = registra_img[registra_img > 0]

        
        # continual img
        seg_continual_img = seg_full_image if seg_continual_img is None else seg_continual_img + seg_full_image
        # registra continual img
        reg_continual_img = reg_full_image if reg_continual_img is None else reg_continual_img + reg_full_image
        # save
        cv2.imwrite(os.path.join(save_p, "continual", f'{img_name}.png'), seg_continual_img)
        cv2.imwrite(os.path.join(save_p, "registra_continual", f'{img_name}.png'), reg_continual_img)
        cv2.imwrite(os.path.join(save_p, "origin", f'{img_name}.png'), seg_full_image)
        print(f"Processed frame {img_name}")
        print('--'*10)


    # 打印列表中的所有 bbox 实例
    # bbox_list.print_bboxes()
    print('--'*10)

    # plt.xlabel('Frame Number')
    # plt.ylabel('Intensity')
    # plt.title('BBox Frame Numbers and Intensities')
    # plt.legend()
    # plt.savefig(os.path.join(save_p, 'intensity_plot.png'))   
    # plt.show()

    # Display the final image with all bounding boxes
    # save_path = os.path.join(save_p, 'all_bboxes.png')
    # cv2.imwrite(save_path, blank_image)
    # cv2.imshow('All Bounding Boxes', blank_image)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':    

    kidney_p = r"D:\Vscode_code\Python\CEUS\animal\kidney_pred"
    image_p = r"D:\Vscode_code\Python\CEUS\animal\threshold-continual\threshold_100\Ceus-outlier-scatter\seg_no_registra\no_stack_scattered"
    save_p = r"D:\Vscode_code\Python\CEUS\animal\threshold-continual\threshold_100\Ceus-outlier-scatter\seg_no_registra\separate_region_method"
    # process(kidney_p, image_p, save_p, end_index=int(140*0.5))
    


    b_mode_p = r"D:\Vscode_code\Python\CEUS\animal\b_mode"
    fore_p = r"D:\Vscode_code\Python\CEUS\animal\threshold-continual\threshold_100\Ceus-outlier-scatter\seg_no_registra\separate_region_method"
    back_p = r"D:\Vscode_code\Python\CEUS\animal\ceus_mode"
    save_p = r"D:\Vscode_code\Python\CEUS\animal\threshold-continual\threshold_100\Ceus-outlier-scatter\seg_no_registra\output_video"
    generate_video(os.path.join(fore_p, "origin"), back_p, save_p, name='output_video1')
    generate_video_with_two_images(back_p, b_mode_p, os.path.join(fore_p, "origin"), save_p, name='output_video2')
    create_video_from_images_with_background(os.path.join(fore_p, "origin"), 
                                             os.path.join(fore_p, "continual"), 
                                             os.path.join(fore_p, "registra_continual"), 
                                             back_p, 
                                             os.path.join(save_p, "output_video3.avi"), 
                                             fps=5, stack_method='continual')
    print("Done")