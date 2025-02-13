'''
Author: LiuyiYang 1183140624@qq.com
Date: 2025-01-21 09:17:29
LastEditors: LiuyiYang 1183140624@qq.com
LastEditTime: 2025-02-11 10:00:08
FilePath: "/CEUS/main/utils.py"
'''
import cv2
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import re


def create_video_with_background(folder1, folder2, background_folder, output_path, fps):
    """
    从图片文件夹创建视频，添加背景图片。
    """
    files = sorted(os.listdir(folder1))
    if not files:
        print(f"文件夹 {folder1} 中没有文件。")
        return

    first_image = cv2.imread(os.path.join(folder1, files[0]))
    height, width, _ = first_image.shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for file in files:
        img1 = cv2.imread(os.path.join(folder1, file))
        img2 = cv2.imread(os.path.join(folder2, file))
        background = cv2.imread(os.path.join(background_folder, file))

        if img1 is None or img2 is None or background is None:
            print(f"跳过文件: {file}")
            continue

        img_combined = cv2.addWeighted(img1, 0.5, background, 0.5, 0)
        video.write(img_combined)

    video.release()
    print(f"视频保存至: {output_path}")


def plot_transform_para(start, end, win_len, transform_params_set, file_p, 
                        outlier_threshold=5, file_name='Kidney-registra'):
    # Plot the transform parameters 
    plt.figure(figsize=(12, 6))
    labels = ['Rotation', 'X', 'Y']
    for j in range(3):
        plt.plot(range(start + win_len, end, win_len), transform_params_set[j], linestyle='-', label=labels[j])
        # 标注异常值
        for idx, val in enumerate(transform_params_set[j]):
            if abs(val) > outlier_threshold:
                plt.scatter(start + win_len + idx * win_len, val, color='red', zorder=5)
                plt.text(start + win_len + idx * win_len, val, f'{val:.2f}', fontsize=9, color='red')
    plt.title(f'{file_name} Transform Parameters')
    plt.xlabel('Image Index')
    plt.ylabel('Transform Parameter Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file_p}\\transform_params_plot.png")
    # plt.show()


def plot_intensity(start, end, win_len, intensity_list, file_p, file_name='1'):
    plt.figure(figsize=(10, 6))
    plt.plot(range(start, end, win_len), intensity_list, linestyle='-', color='c')
    plt.title('CEUS Region Intensity Over Time')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.savefig(os.path.join(file_p, f'intensity_plot_{file_name}.png'))


def create_video_from_images_with_background(folder1, folder2, folder3, background_image_path, 
                                             output_video_path, fps=10, stack_method='1-frame'):
    """
    # 将三个文件夹中的图片合并为一个视频，背景为ceus_mode
    从三个文件夹中读取同名图片，创建视频，每帧并排显示三个图像，并在每个子图片的背景添加一个固定的图像。
    Args:
        folder1: 第一个文件夹的路径。
        folder2: 第二个文件夹的路径。
        folder3: 第三个文件夹的路径。
        background_image_path: 背景图像的路径。
        output_video_path: 输出视频的路径。
        fps: 视频的帧率 (每秒帧数)。
    """

    # 获取所有图片文件名（假设三个文件夹中的文件名相同）
    print("folder3: ", folder3)
    image_files = [f for f in os.listdir(folder3) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort(key=lambda x: int(re.match(r'^(\d+)', x).group(1)))

    if not image_files:
        print(f"Error: No images found in {folder3}")
        return

    # 读取第一张图片，以确定输出视频的尺寸
    img1 = cv2.imread(os.path.join(folder1, image_files[0]))
    if img1 is None:
         print(f"Error reading image: {os.path.join(folder1, image_files[0])}")
         return
    height, width, _ = img1.shape
    combined_width = width * 3  # 三张图并排，总宽度是单张图宽度的3倍

    # 创建视频写入器 (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 编码 (也可以尝试 'XVID', 'MJPG' 等)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (combined_width, height + 50))  # 增加高度以容纳标题

    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

    # 遍历并处理图片
    for filename in image_files:
        # 读取三个文件夹中的同名图片
        img1_path = os.path.join(folder1, filename)
        img2_path = os.path.join(folder2, filename)
        img3_path = os.path.join(folder3, filename)

        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        img3 = cv2.imread(img3_path)
        
        # 检查图片是否成功读取, 并resize成统一大小
        if img1 is None or img2 is None or img3 is None:
            print(f"Error reading image files: {img1_path}, {img2_path}, {img3_path}")
            continue  #跳过当前这张图片
            
        if img1.shape != (height, width, 3):
            img1 = cv2.resize(img1, (width,height))
        if img2.shape != (height, width, 3):
            img2 = cv2.resize(img2, (width,height))
        if img3.shape != (height, width, 3):
            img3 = cv2.resize(img3, (width,height))

         # 读取背景图像
        background_image = cv2.imread(os.path.join(background_image_path, filename))
        if background_image is None:
            print(f"Error reading background image: {background_image_path}")
            return
        # 添加背景图像
        img1_with_bg = cv2.addWeighted(img1, 1, cv2.resize(background_image, (width, height)), 1, 0)
        img2_with_bg = cv2.addWeighted(img2, 0.75, cv2.resize(background_image, (width, height)), 1, 0)
        img3_with_bg = cv2.addWeighted(img3, 0.5, cv2.resize(background_image, (width, height)), 1, 0)

        # 合并图像 (水平拼接)
        combined_frame = np.concatenate((img1_with_bg, img2_with_bg, img3_with_bg), axis=1)

        # 创建一个新的图像来容纳标题和合并后的图像
        frame_with_title = np.zeros((height + 50, combined_width, 3), dtype=np.uint8)
        frame_with_title[50:, :] = combined_frame

        # 添加标题
    
        cv2.putText(frame_with_title, 'No Registrata', 
                    (width // 2 - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame_with_title, f'No Registrata & {stack_method}', 
                    (width + width // 2 - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame_with_title, f'Registrata & {stack_method}', 
                    (2 * width + width // 2 - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 写入视频帧
        out.write(frame_with_title)

        # print(f"Processed: {filename}")

    # 释放资源
    out.release()
    print(f"Video created successfully: {output_video_path}")
