from random import randint
import cv2
import numpy as np
import Defects as defects
from matplotlib import pyplot as plt


class Product:
    scratch_sum = 0
    blot_sum = 0
    count = 0

    def __init__(self, p, xi, yi, x, y, w, h):
        # 工件ID
        self.pid = p
        # 中心坐标
        self.centre_x = xi
        self.centre_y = yi
        # 边框
        self.bound_x = x
        self.bound_y = y
        self.bound_w = w
        self.bound_h = h
        # 颜色
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        # 运动轨迹
        self.tracks = []
        # 0代表正常，1代表工件存在划痕，2代表工件存在污渍
        self.state = 0
        self.sample = 0
        # 储存缺陷对象
        self.defects = []

    # 获取ID
    def getId(self):
        return self.pid

    def getBoundX(self):
        return self.bound_x

    def getBoundY(self):
        return self.bound_y

    def getBoundW(self):
        return self.bound_w

    def getBoundH(self):
        return self.bound_h

    def getX(self):  # 获取中心x
        return self.centre_x

    def getY(self):  # 获取中心y
        return self.centre_y

    # 更新位置
    def updateCoords(self, xn, yn, x, y, w, h):
        self.centre_x = xn
        self.centre_y = yn
        self.bound_x = x
        self.bound_y = y
        self.bound_w = w
        self.bound_h = h
        # 增加轨迹位置
        self.tracks.append([self.centre_x, self.centre_y])

    # 获取轨迹
    def getTracks(self):
        return self.tracks

    # 获取标记颜色
    def getRGB(self):
        return self.R, self.G, self.B

    # 保存
    def save_pic(self, frame):
        pic = frame[self.bound_y:self.bound_y + self.bound_h, self.bound_x:self.bound_x + self.bound_w]
        self.sample = pic
        # 改变图像大小
        cv2.imwrite("./output/" + str(self.pid) + ".jpg", pic)

    # 缺陷识别和分类
    def defect_detect(self):
        gray = cv2.cvtColor(self.sample, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 135, 250, cv2.THRESH_BINARY)[1]
        cv2.imwrite("./output/" + str(self.pid) + "thresh.jpg", thresh)
        contour, hierar = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img = self.sample.copy()
        for cnt in contour:
            area = cv2.contourArea(cnt)
            if 100 < area <= 15000:
                x, y, w, h = cv2.boundingRect(cnt)
                ret = img[y:y + h, x: x + w]
                # 提取缺陷
                gray = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 135, 250, cv2.THRESH_BINARY_INV)[1]
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                mask = np.zeros(gray.shape).astype(gray.dtype)
                cv2.fillPoly(mask, contours, (255, 255, 255))# 填充任意形状的多边形
                cv2.imwrite("./output/" + str(self.pid) + "fillPoly.jpg", mask)
                result = cv2.bitwise_and(gray, mask)
                cv2.imwrite("./output/" + str(self.pid) + "bitwise_and.jpg", result)#得到缺陷部分
                # 直方图计算
                """
                cv2.calcHist函数：
                imaes:输入的图像（）
                channels:选择图像的通道
                mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
                histSize:使用多少个bin(柱子)，一般为256
                ranges:像素值的范围，一般为[0,255]表示0~255
                
                
                """
                hist = cv2.calcHist([result], [0], None, [256], [0, 255])
                s = sum(hist)
                # 判断缺陷，计算各像素点值的个数百分比
                for i in range(len(hist)):
                    if hist[i] > 0:
                        hist[i] = hist[i] / s

                # 根据规则来匹配各种缺陷的情况
                hist_sum_scratch = 0
                hist_sum_blot = 0
                for i in range(90, 135):
                    hist_sum_scratch = hist_sum_scratch + hist[i]

                for i in range(15, 90):
                    hist_sum_blot = hist_sum_blot + hist[i]

                if hist_sum_scratch > 0.1:
                    d = defects.Defect(1, x, y, w, h)
                    self.defects.append(d)
                    self.state = 1
                    # print("此处缺陷划痕")

                if hist_sum_blot > 0.1:
                    d = defects.Defect(2, x, y, w, h)
                    self.defects.append(d)
                    self.state = 2
                    # print("此处缺陷污渍")

        if self.state == 1:
            Product.scratch_sum = Product.scratch_sum + 1
        elif self.state == 2:
            Product.blot_sum = Product.blot_sum + 1

        return self.defects
