##深度学习过程中，需要制作训练集和验证集、测试集。

import os, random, shutil
def moveFile(fileDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
       
     
        pick=random.shuffle(pathDir)
        
        seg = split_list_n_list(pick, 5)  #随机选取picknumber数量的样本图片
        
        
        for name in seg:
               print(name)
        return
def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
 
    for i in range(0, n):
        yield origin_list[i*cnt:(i+1)*cnt]

if __name__ == '__main__':
    fileDir = "D:\FYP\data\\BUSI_mask\\"    #源图片文件夹路径
    tarDir = 'D:\FYP\data\\testmask1\\'    #移动到新的文件夹路径
    random.seed(1)
    moveFile(fileDir)