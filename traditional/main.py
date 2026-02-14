from matplotlib import pyplot as plt
from torchvision import transforms, datasets

from traditional.models.dbs import HVSModel, DBSAlgorithm
from traditional.models.error_diffusion import error_diffusion
from traditional.models.ordered_dithering import ordered_dithering
import os

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 配置信息
DOWNLOAD_VOC2012 = False

GRAY = True
ORDERED_DITHERING = True
ERROR_DIFFUSION = True
DBS = True

GRAY_PATH = '..\\out\\gray'
ORDERED_DITHERING_PATH = '..\\out\\ordered_dithering'
ERROR_DIFFUSION_PATH = '..\\out\\error_diffusion'
DBS_PATH = '..\\out\\dbs'
PLOT_PATH = '..\\out\\plot'

if __name__ == "__main__":
    # 定义变换：转为灰度图 -> 转为Tensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 加载VOC数据集样本
    voc_train = datasets.VOCSegmentation(
        root='../data',
        year='2012',
        image_set='train',
        download=DOWNLOAD_VOC2012,
        transform=transform
    )
    gray_tensor, gray_target = voc_train[0]  # 获取第一个样本的灰度张量


    # 获取原图像文件名
    image_path = voc_train.images[0]
    image_filename = os.path.basename(image_path)

    plt.figure(figsize=(12, 8))
    index = 0

    if GRAY:
        os.makedirs(GRAY_PATH, exist_ok=True)
        gray_img = transforms.ToPILImage()(gray_tensor)
        gray_output_path = os.path.join(GRAY_PATH, image_filename)
        gray_img.save(gray_output_path)
        print(f"灰度图已保存为 '{gray_output_path}'")

        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(gray_img, cmap='gray')
        plt.title("原始灰度图")
        plt.axis('off')

    if ORDERED_DITHERING:
        # 应用有序抖动
        od_tensor, od_img = ordered_dithering(gray_tensor)
        print("有序抖动法结果已生成")
        os.makedirs(ORDERED_DITHERING_PATH, exist_ok=True)
        od_output_path = os.path.join(ORDERED_DITHERING_PATH, image_filename)
        od_img.save(od_output_path)
        print(f"有序抖动结果已保存为 '{od_output_path}'")

        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(od_img, cmap='gray')
        plt.title("有序抖动结果")
        plt.axis('off')

    if ERROR_DIFFUSION:
        # 应用误差扩散
        ed_tensor, ed_img = error_diffusion(gray_tensor, kernel_type='floyd-steinberg')
        print("误差扩散法结果已生成")
        os.makedirs(ERROR_DIFFUSION_PATH, exist_ok=True)
        ed_output_path = os.path.join(ERROR_DIFFUSION_PATH, image_filename)
        ed_img.save(ed_output_path)
        print(f"有序抖动结果已保存为 '{ed_output_path}'")

        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(ed_img, cmap='gray')
        plt.title("误差抖动结果")
        plt.axis('off')

    if DBS:
        # DBS
        hvs = HVSModel()
        dbs = DBSAlgorithm(hvs_model=hvs, max_iter=20)
        dbs_tensor, dbs_img = dbs.optimize(gray_tensor)
        print("DBS结果已生成")
        os.makedirs(DBS_PATH, exist_ok=True)
        dbs_output_path = os.path.join(DBS_PATH, image_filename)
        dbs_img.save(dbs_output_path)
        print(f"DBS优化后半色调图已保存为 '{dbs_output_path}'")

        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(dbs.initialize_halftone(gray_tensor).squeeze(0).numpy(), cmap='gray')
        plt.title("初始阈值半色调图")
        plt.axis('off')

        index += 1
        plt.subplot(3, 2, index)
        plt.imshow(dbs_img, cmap='gray')
        plt.title("DBS优化后半色调图")
        plt.axis('off')

    os.makedirs(PLOT_PATH, exist_ok=True)
    plot_filename = f"{os.path.splitext(image_filename)[0]}_combined.png"
    plot_output_path = os.path.join(PLOT_PATH, plot_filename)
    plt.savefig(plot_output_path, dpi=300, bbox_inches='tight')
    print(f"合并可视化图已保存为 `{plot_output_path}`")

    plt.show()
