# ocr识别
import os
from PaddleOCR.paddleocr import PaddleOCR, draw_ocr
from PaddleNLP.paddlenlp.taskflow.taskflow import Taskflow
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib


# 命名实体识别与词频统计

def deletaNum(doc):
    return [i for i in doc if len(i) > 1]


def LAC(lac, res_list):  # 根据结果筛选
    doc = [text[1][0] for res in res_list for text in res]

    doc = deletaNum(doc)
    # print('\n\ndoc is ',doc)
    lac_dic = lac(doc)
    enti = []
    for la in lac_dic:
        if 'ORG' or 'nt' in la['tags']:
            idx_, idx_1 = [], []
            if 'ORG' in la['tags']:
                idx_ = [la['tags'].index('ORG')]
            elif 'nt' in la['tags']:
                idx_1 = [la['tags'].index('nt')]
            # print(idx_, idx_1, la)
            t = [la['segs'][i] for i in idx_ + idx_1]
            enti += t
    print('\n\nentity is ', enti)

    return enti


def PlotHist(counter):
    matplotlib.rc("font", family='FZHuaLi-M14S')

    cnt = {}
    for key in counter.keys():
        if counter[key] >= 10:
            cnt[key] = counter[key]
    print(cnt)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(cnt)), cnt.values(), tick_label=list(cnt.keys()))  # , orientation="horizontal"
    # for a, b in zip(x_list, y_list):
    #     plt.text(a, b + 0.05, '%.0f' % b, ha='center', va='bottom', fontsize=10)
    plt.xticks(rotation=45)
    plt.savefig('./img.png')
    plt.show()


if __name__ == '__main__':
    # 模型路径下必须含有model和params文件
    ocr = PaddleOCR(det_model_dir='./PaddleOCR/output/ch_db_mv3_inference/inference',
                    use_angle_cls=True)
    lac = Taskflow("lexical_analysis")

    enti_list = []
    pdfFolder = './ResearchReport'
    for p in os.listdir(pdfFolder):
        if os.path.isdir(os.path.join(pdfFolder, p)):
            print(p)
            imgPath = pdfFolder + '/' + p
            res_list = []
            for i in os.listdir(imgPath):
                img_path = os.path.join(imgPath, i)
                result = ocr.ocr(img_path, cls=True)
                res_list.append(result)

            enti = LAC(lac, res_list)
            enti_list += enti

        print(Counter(enti_list))

    PlotHist(Counter)