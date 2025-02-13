import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import SimpleITK as sitk
from utils import plot_transform_para, plot_intensity, create_video_from_images_with_background


class Processor:
    def __init__(self, params):
        self.params = params
        self.root_p = self.params["root_p"]
        self.ceus_pred_p = os.path.join(self.root_p, "kidney_pred")
        self.ceus_mode_p = os.path.join(self.root_p, "ceus_mode")
        self.threshold_p = os.path.join(root_p, "threshold-continual")
        self.file_name = self.params["file_name"]

        self.start = self.params["start"]
        self.end = self.params["end"]
        self.temp_end = self.params["temp_end"]
        self.win_len = self.params["win_len"]
        self.outlier_threshold = self.params["outlier_threshold"]
        self.threshold_range = self.params["threshold_range"]
        self.registra_method = self.params["registra_method"]  # rigid or affine
        self.num_scatter = self.params["num_scatter"]
        self.area_threshold = self.params["area_scatter"]

        self.fps = self.params["fps"]
        self.file_name = self.params["file_name"]
        self.file_p = None
        self.seg_no_registra_p = None
        self.seg_registra_p = None


    def create_directory(self, path):
        os.makedirs(path, exist_ok=True)

    
    def create_saved_directory(self, threshold):
        # 创建保存路径
        detailed_threshold_p = os.path.join(self.threshold_p, f"threshold_{threshold}")
        self.create_directory(detailed_threshold_p)
        self.file_p = os.path.join(detailed_threshold_p, self.params['file_name'])
        self.create_directory(self.file_p)

        self.seg_no_registra_p = os.path.join(self.file_p, "seg_no_registra")
        self.create_directory(os.path.join(self.seg_no_registra_p, "no_stack"))
        self.create_directory(os.path.join(self.seg_no_registra_p, "no_stack_scattered"))
        self.create_directory(os.path.join(self.seg_no_registra_p, "1_frame_stack"))
        self.create_directory(os.path.join(self.seg_no_registra_p, "continual_stack"))

        self.seg_registra_p = os.path.join(self.file_p, "seg_registra")
        self.create_directory(os.path.join(self.seg_registra_p, "1_frame_stack"))
        self.create_directory(os.path.join(self.seg_registra_p, "continual_stack"))


    def seg_artery(self, ceus_img, b_mode_seg, dilate=(50, 50), threshold=100, morphology=(5, 5)):
        """
        分割高亮区域的函数。
        """
        image = cv2.imread(str(ceus_img))
        mask = cv2.imread(str(b_mode_seg))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        mask[mask > 0] = 1
        # mask = cv2.dilate(mask, np.ones(dilate, np.uint8), iterations=1)
        image_region = image * mask
        _, result = cv2.threshold(image_region, threshold, 255, cv2.THRESH_BINARY)

        # 去散点
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, np.ones(morphology, np.uint8))
        return result


    def registration(self, fixed_image, moving_image, output_p=None, method='rigid'):
        """
        图像配准代码，可选择刚性或仿射变换。
        返回变换矩阵、配准后的图像和变换参数（Angle、X和Y）。
        """
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
        if method == 'rigid':
            transform = sitk.Euler2DTransform()
        elif method == "affine":
            transform = sitk.AffineTransform(2)
        else:
            raise ValueError("Unsupported registration method: {}".format(method))

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

        # 将配准后的图像进行重采样
        resampled_image = sitk.Resample(
            moving_image,  # 移动图像
            fixed_image,  # 固定图像
            final_transform,  # 最终变换
            sitk.sitkNearestNeighbor,  # 最邻近插值
            0.0,  # 默认像素值
            moving_image.GetPixelID()  # 像素类型
        )
        
        resampled_array = sitk.GetArrayFromImage(resampled_image)

        # 输出变换矩阵在x、y轴上的分量值
        transform_parameters = final_transform.GetParameters()
        # print("变换参数:", transform_parameters)

        # 获取度量值 (MSE)
        mse = registration_method.GetMetricValue()

        return final_transform, resampled_array, transform_parameters
    
    
    def apply_transform(self, image, transform, output_p=None):
        """
        Apply a given SimpleITK transform to an image and save the result.

        Parameters:
        - image_p: str, path to the input image.
        - transform: SimpleITK.Transform, the transform to apply.
        - output_p: str, path to save the transformed image.
        """

        # 将图像转换为 32 位浮点数，以确保支持的图像类型
        image = sitk.Cast(image, sitk.sitkFloat32)
        
        # 将图像进行重采样
        resampled_image = sitk.Resample(
            image,  # 输入图像
            image,  # 参考图像
            transform,  # 变换矩阵
            sitk.sitkNearestNeighbor,  # 最邻近插值
            0.0,  # 默认像素值
            image.GetPixelID()  # 像素类型
        )
        
        # 保存重采样后的图像
        # sitk.WriteImage(resampled_image, output_p)
        # print(f"Transformed image saved to {output_p}")
        return sitk.GetArrayFromImage(resampled_image)


    def saved_video(self):
        # 保存两个视频，第一个是单帧堆叠，第二个是连续帧堆叠
        # 1. 1-frame stacking
        folder1_path = rf"{self.seg_no_registra_p}\no_stack_scattered"  
        folder2_path = rf"{self.seg_no_registra_p}\1_frame_stack"  
        folder3_path = rf"{self.seg_registra_p}\1_frame_stack"
        background_image_path = rf"{self.ceus_mode_p}"
        output_path = rf"{self.file_p}\1-frame-stack-video.mp4"  # 输出视频的路径
        create_video_from_images_with_background(folder1_path, folder2_path, folder3_path, 
                                                 background_image_path, output_path, 
                                                 fps=self.fps, stack_method="1_frame")  # 可以调整帧率

        # 2. continual stacking
        folder2_path = rf"{self.seg_no_registra_p}\continual_stack"  # 替换为第二个文件夹的路径
        folder3_path = rf"{self.seg_registra_p}\continual_stack"  # 替换为第三个文件夹的路径
        output_path = rf"{self.file_p}\continual-stack-video.mp4"  # 输出视频的路径
        create_video_from_images_with_background(folder1_path, folder2_path, folder3_path, 
                                                 background_image_path, output_path, 
                                                 fps=self.fps, stack_method="continual")  # 可以调整帧率


    def remove_scattered(self, res):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(res, connectivity=8)
        filtered_res = np.zeros_like(res)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            top_indices = np.argsort(areas)[-self.num_scatter:]
            for idx in top_indices:
                if areas[idx] >= self.area_threshold:
                    filtered_res[labels == idx + 1] = 255
        return filtered_res


    def compute_ceus_region_intensity(self, ceus_img_path, mask_img_path):
        """
        计算CEUS区域的平均灰度值。
        """
        intensity_list = []
        for i in range(self.start, self.end, self.win_len):
            ceus_img = Path(ceus_img_path) / f"{i}.png"
            mask_img = Path(mask_img_path) / f"{i}.png"
        
            ceus_image = cv2.imread(str(ceus_img), cv2.IMREAD_GRAYSCALE)
            mask_image = cv2.imread(str(mask_img), cv2.IMREAD_GRAYSCALE)

            # Ensure mask is binary
            mask_image[mask_image > 0] = 1

            # Apply mask to CEUS image
            masked_ceus = ceus_image * mask_image

            # Calculate the total intensity of the masked region
            total_intensity = np.sum(masked_ceus[mask_image > 0])
            intensity_list.append(total_intensity)

        return intensity_list


    def interval(self, intensity_list):
        """
        计算intensity_list的斜率。
        """
        intensity_list = [x / 1e6 for x in intensity_list]
        window_size = 5
        smoothed = np.convolve(intensity_list, np.ones(window_size)/window_size, mode='valid')
        smoothed_intensity_list = np.concatenate((intensity_list[:window_size-1], smoothed))        

        # 计算smoothed_intensity_list的斜率
        slopes = np.diff(intensity_list)

        plot_intensity(self.start, self.end - 1, self.win_len, slopes, self.file_p, file_name="Slope of Smoothed Intensity")
        plot_intensity(self.start, self.end, self.win_len, smoothed_intensity_list, self.file_p, file_name="Smoothed Intensity")
        plot_intensity(self.start, self.end, self.win_len, intensity_list, self.file_p, file_name="Intensity")

        # 计算斜率的峰值，用峰值前后10帧作为关键区间
        max_slope_index = np.argmax(slopes)
        key_interval_start = max(0, max_slope_index - 10)
        key_interval_end = min(len(slopes), max_slope_index + 10)
        print(f"Max slope index: {max_slope_index}")
        print(f"Key interval: {key_interval_start} to {key_interval_end}")
        return key_interval_start, key_interval_end
        

    def process_threshold(self, threshold):
        print(f"Processing threshold: {threshold}")
        
        self.create_saved_directory(threshold)
        return
        moving_1frame, moving_continual = None, None
        stack_continual = None
        stack_registra_1frame, stack_registra_continual = None, None
        last_frame = 0

        # 计算ceus kidney region的像素强度
        intensity_list = self.compute_ceus_region_intensity(self.ceus_mode_p, self.ceus_pred_p)
        # Get the key interval (process the segmentation first, then denoise the pixel intensity sequence, find inflection points, and then find the key interval)
        key_interval_start, key_interval_end = self.interval(intensity_list)

        # 保存每次配准的变换参数（角度、X和Y）
        transform_params_set = [[] for i in range(3)]
        for i in range(self.start, self.temp_end, self.win_len):
            fixed_p = os.path.join(self.ceus_mode_p, f"{i}.png")
            fixed_image = sitk.ReadImage(fixed_p) 

            # 1. 阈值分割并保存 
            res = self.seg_artery(rf"{self.ceus_mode_p}\\{i}.png", rf"{self.ceus_pred_p}\\{i}.png", 
                                threshold=threshold, morphology=(5, 5))
            cv2.imwrite(rf"{self.seg_no_registra_p}\\no_stack\\{i}.png", res)

            # 保留阈值结果连通域前5名的且面积大于阈值并保存
            if self.num_scatter > 0:
                res_scattered = self.remove_scattered(res)
                # Find contours and draw bounding boxes
                stack_continual = res_scattered if stack_continual is None else stack_continual + res_scattered
                res_scattered_bbox = res_scattered.copy()
                contours, _ = cv2.findContours(res_scattered_bbox, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(res_scattered_bbox, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imwrite(rf"{self.seg_no_registra_p}\\no_stack_scattered\\{i}.png", res_scattered_bbox)

            if i == self.start:
                moving_image = fixed_image
                moving_image.CopyInformation(fixed_image)
                # 用分割后的阈值结果作为moving_registra_image
                moving_1frame = sitk.GetImageFromArray(res_scattered)
                moving_continual = sitk.GetImageFromArray(res_scattered)
                continue

            # 检查图像是否为空或全黑
            if sitk.GetArrayFromImage(moving_1frame).sum() == 0:
                print(f"Moving 1frame {i-1} is empty or all black!")
                moving_1frame = sitk.GetImageFromArray(res_scattered)  
                if sitk.GetArrayFromImage(moving_continual).sum() == 0:
                    print(f"Moving continual {i-1} is empty or all black!")
                    moving_continual = sitk.GetImageFromArray(res_scattered)
                transform_params_set[0].append(0)
                transform_params_set[1].append(0)
                transform_params_set[2].append(0)
                continue

            if sitk.GetArrayFromImage(moving_continual).sum() == 0:
                print(f"Moving continual {i-1} is empty or all black!")
                moving_continual = sitk.GetImageFromArray(res_scattered)
                transform_params_set[0].append(0)
                transform_params_set[1].append(0)
                transform_params_set[2].append(0)
                continue

            if res_scattered.sum() == 0:   
                print(f"Fixed image {i} is empty or all black!")
                transform_params_set[0].append(0)
                transform_params_set[1].append(0)
                transform_params_set[2].append(0)
                continue

            # 2. 阈值分割后结果与上一帧进行堆叠，根据配准结果决定是否进行这一步，在下方代码
            
            if 'Kidney' in self.file_name:
                # 3.1 肾部轮廓的配准
                transform, registed_arr, transform_params = self.registration(fixed_image, moving_image)
            elif 'Ceus' in self.file_name:
                # 3.3 Ceus_mode的配准
                transform, registed_arr, transform_params = self.registration(fixed_image, moving_image)
            elif 'Blood' in self.file_name:
                # 3.2 直接使用血管分割结果的配准
                fixed_registra_image = sitk.GetImageFromArray(res)
                transform, res_registra, transform_params = self.registration(fixed_registra_image, moving_registra_image)

            # 将每次的配准变换矩阵参数保存，便于后续绘画图像
            [transform_params_set[j].append(transform_params[j]) for j in range(3)]
            
            # 判断变换矩阵参数是否为异常值，决定是否保存图像
            if abs(transform_params[0]) < self.outlier_threshold and abs(transform_params[1]) < self.outlier_threshold and abs(transform_params[2]) < self.outlier_threshold:
                if 'Kidney' in self.file_name or 'Ceus' in self.file_name:
                    # 3.1 肾部配准后的处理，使用配准得到的变换矩阵对血管进行配准
                    # 3.3 Ceus_model配准后的处理，使用配准得到的变换矩阵对血管进行配准
                    res_registra_1frame = self.apply_transform(moving_1frame, transform)
                    res_registra_continual = self.apply_transform(moving_continual, transform)

                # 2. 保存血管阈值分割的堆叠结果，1frame and continual【不变】
                # 3. 保存血管配准后的结果，1frame and continual【不变】
                res_registra_1frame[res_registra_1frame > 0] = 255
                stack_1frame = sitk.GetArrayFromImage(moving_1frame) + res_scattered
                stack_registra_1frame = res_registra_1frame + res_scattered
                cv2.imwrite(rf"{self.seg_no_registra_p}\1_frame_stack\{i}.png", stack_1frame)
                cv2.imwrite(rf"{self.seg_registra_p}\1_frame_stack\{i}.png", stack_registra_1frame)

                res_registra_continual[res_registra_continual > 0] = 255
                stack_registra_continual = res_registra_continual + res_scattered
                cv2.imwrite(rf"{self.seg_no_registra_p}\continual_stack\{i}.png", stack_continual)
                cv2.imwrite(rf"{self.seg_registra_p}\continual_stack\{i}.png", stack_registra_continual)

                # 配准结束后再改变moving_image
                moving_image = fixed_image
                moving_image.CopyInformation(fixed_image)
                moving_1frame = sitk.GetImageFromArray(res_scattered)
                moving_continual = sitk.GetImageFromArray(stack_registra_continual)

                # save last frame
                last_frame = i
        
        
        # 在temp_end到end之间，使用最后一帧的结果进行堆叠
        for i in range(self.temp_end, self.end, self.win_len):
            # 肾部配准
            fixed_p = os.path.join(self.ceus_mode_p, f"{i}.png")
            fixed_image = sitk.ReadImage(fixed_p) 
            # 计算ceus kidney region的像素值

            if 'Kidney' in self.file_name or 'Ceus' in self.file_name:
                transform, registed_arr, transform_params = self.registration(fixed_image, moving_image)

            # 将每次的配准变换矩阵参数保存，便于后续绘画图像
            [transform_params_set[j].append(transform_params[j]) for j in range(3)]
            if abs(transform_params[0]) < self.outlier_threshold and abs(transform_params[1]) < self.outlier_threshold and abs(transform_params[2]) < self.outlier_threshold:
                if 'Kidney' in self.file_name or 'Ceus' in self.file_name:
                    res_registra_1frame = self.apply_transform(sitk.GetImageFromArray(stack_registra_1frame), transform)
                    res_registra_continual = self.apply_transform(sitk.GetImageFromArray(stack_registra_continual), transform)
                
                cv2.imwrite(rf"{self.seg_no_registra_p}\1_frame_stack\{i}.png", stack_1frame)
                cv2.imwrite(rf"{self.seg_registra_p}\1_frame_stack\{i}.png", res_registra_1frame)
                cv2.imwrite(rf"{self.seg_no_registra_p}\continual_stack\{i}.png", stack_continual)
                cv2.imwrite(rf"{self.seg_registra_p}\continual_stack\{i}.png", res_registra_continual)
                shutil.copyfile(rf"{self.seg_no_registra_p}\no_stack\{last_frame}.png", rf"{self.seg_no_registra_p}\no_stack\{i}.png")
                shutil.copyfile(rf"{self.seg_no_registra_p}\no_stack_scattered\{last_frame}.png", rf"{self.seg_no_registra_p}\no_stack_scattered\{i}.png")

                stack_registra_1frame = res_registra_1frame
                stack_registra_continual = res_registra_continual

            # 配准结束后再改变moving_image
            moving_image = fixed_image
            moving_image.CopyInformation(fixed_image)
        
        
        # 保存变换参数图
        plot_transform_para(self.start, self.end, self.win_len, 
                            transform_params_set, self.file_p, 
                            self.outlier_threshold, self.file_name
                            )
                
        # self.saved_video()


    def run(self):
        for threshold in self.threshold_range:
            self.process_threshold(threshold)
            self.saved_video()
            print('thresholod {} done!'.format(threshold))
        print("All done!")

if __name__ == "__main__":
    # 配置参数
    root_p = r"D:\Vscode_code\Python\CEUS\animal"

    params = {
        "root_p": root_p,
        "file_name": "Ceus-outlier-scatter",
        "start": 0,
        "end": 140, 
        "temp_end": int(140*0.5),
        "win_len": 1,
        "outlier_threshold": 5,
        "threshold_range": range(100, 101, 20),
        "registra_method": "rigid",
        "fps": 5,
        "num_scatter": 5,
        "area_scatter": 400,
    }

    processor = Processor(params)
    processor.run()
    

    
