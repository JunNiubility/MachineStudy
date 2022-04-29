import numpy as np
import xlrd
import pandas as pd
import itertools
import random as rd


def get_date(url='excel.xls'):
    """
    获取数据
    :param url:文件地址,字符串
    :return:allclass所有同学的课程表,字典; name所有同学的姓名,列表
    """
    # Excel表格数据
    data = xlrd.open_workbook(url, formatting_info=False)
    # 获取表格中的子表格名称，即姓名
    name = data.sheet_names()
    # 统计人数
    m = len(name)
    # 把Excel中所有数据添加到一个列表中去
    classdata = [data.sheet_by_index(i).row_values(j) for i in range(m) for j in range(4)]
    # 将数据转为整数
    classdata = np.array(classdata, dtype=np.int_)
    # 形成每一人的课程表信息
    allclass = {name[i]: classdata[4 * i:4 * (i + 1), ] for i in range(m)}
    print(allclass)
    return allclass, name


def ispossible(allclass, name):
    """
    判断是否存在时间上没有一个人有空的情况
    :param allclass:所有同学的课程表,字典
    :param name:所有同学的姓名,列表
    :return:True or False
    """
    tmpt = np.ones([4, 5])
    for i in range(len(name)):
        tmpt *= np.array(allclass[name[i]])

    if np.max(np.max(tmpt)) == 0:
        return True
    return False


def total(allclass, name):
    """
    统计空闲时间
    :param allclass:所有同学的课程表,字典
    :param name:所有同学的姓名,列表
    :return:spare_name有空时间段名单,列表;spare_count有空时间段名单人数,列表
    """
    # 用来添加每一段时间，有空的人的姓名
    spare_name = [[[], [], [], [], []],
                  [[], [], [], [], []],
                  [[], [], [], [], []],
                  [[], [], [], [], []]]
    # 用来添加每一段时间，有空人的数量
    spare_count = np.zeros([4, 5])
    for i in range(len(name)):
        for row in range(4):
            for col in range(5):
                # 如果这个人的某一段时间为0，那么将他的姓名添加到空闲统计表中，并进行计数
                if allclass[name[i]][row, col] == 0:
                    spare_name[row][col].append(name[i])
                    spare_count[row, col] += 1
    return spare_name, spare_count


def solution(spare_name, spare_count, name):
    """
    最终结果输出
    :param spare_name:有空时间段名单,列表
    :param spare_count:有空时间段名单人数,列表
    :return:result_name,排班表结果
    """
    # 用来添加最终的值班表
    result_name = [[[], [], [], [], []],
                   [[], [], [], [], []],
                   [[], [], [], [], []],
                   [[], [], [], [], []]]
    # 用来统计最终的值班表，每一段时间的人数
    cnt = np.zeros([4, 5])
    # 用来统计每个人被安排值班的次数
    count = {name[i]: 0 for i in range(len(name))}
    while True:
        for row in range(spare_count.shape[0]):
            for col in range(spare_count.shape[1]):
                if spare_count[row][col] >= 2:
                    tempt = itertools.combinations(spare_name[row][col], 2)
                    tempt = list(tempt)
                    result_name[row][col].append(list(tempt[rd.randint(0, len(tempt) - 1)]))
                else:
                    result_name[row][col].append(spare_name[row][col])
                for item in result_name[row][col]:
                    for i in item:
                        count[i]=1
        for v in count.values():
            if v != 2:
                break
        else:
            break


    return result_name

if __name__ == '__main__':
    allclass, name = get_date()
    if ispossible(allclass, name) is False:
        print("当前有些课时没有一个人有空闲时间")
        exit()
    result, result_count = total(allclass, name)
    time = ['第一二节', '第三四节', '第五六节', '第七八节']
    date = ['星期一', '星期二', '星期三', '星期四', '星期五']
    result_kongxian = pd.DataFrame(result, columns=date, index=time)
    print(result_kongxian)

    class_count = pd.DataFrame(result_count, columns=date, index=time)
    print(class_count)

    result = solution(result, result_count, name)
    result = pd.DataFrame(result, columns=date, index=time)
    print(result.astype(int))
    pd.DataFrame(result).to_excel('result.xls',sheet_name='值班表',index=True,header=True)

