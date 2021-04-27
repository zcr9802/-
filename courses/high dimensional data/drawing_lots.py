#coding = utf-8
import pandas as pd
import os
import sys
import numpy as np
import time

def csv_get():
    if os.path.exists('drawing records.csv'):
        csv_records = pd.read_csv('drawing records.csv', encoding='GBK')
    else:
        print('preparing for the first drawing')
        df = pd.DataFrame(data=None, index=None, columns=['序列','名字','编号','时间','已汇报次数','本次抽签概率'])
        df.to_csv('drawing records.csv', index=False, encoding='GBK')
        csv_records = pd.read_csv('drawing records.csv', encoding='GBK')
    return csv_records

def nb_moni(rec):
    try:
        records_fm = list(map(int,rec['编号'].values))
    except:
        print('记录中编号有误')
        sys.exit(1)
    for i in records_fm:
        if not i in list(d.keys()):
            print('记录中编号有误')
            sys.exit(1)
    return records_fm

def redo():
    csv_records = csv_get()
    csv_records = csv_records.iloc[0:-1,:]
    csv_records.to_csv('drawing records.csv', index=False, encoding='GBK')

def main():
    global d
    d = {1: '张朝瑞', 2: '储企', 3:'金娜', 4:'陶伟', 5:'周文变', 6:'陈睿靖', 7:'郭胜争', 8:'宋霄汉', 9:'吴姝颖', 10:'张治国' }
    p = dict.fromkeys(list(d.keys()),1)
    csv_records = csv_get()
    drawing_list=[]
    records_fm=nb_moni(csv_records)

    for i in list(d.keys()):
        count = records_fm.count(i)
        p[i] = p[i]*0.5**count
    cont = 1/min(list(p.values()))
    for i in list(p.keys()):
        drawing_list.extend([i]*(int(cont*p[i])))
    chosen = np.random.choice(drawing_list)
    print(chosen)
    seq =len(csv_records)+1
    name = d[chosen]
    tm = time.strftime("%Y/%m/%d", time.localtime())
    cs_ar = records_fm.count(chosen)+1
    p_p = p[chosen]/sum(p.values())
    csv_records.loc[len(csv_records)]=[seq,name,chosen,tm,cs_ar,p_p]
    csv_records.to_csv('drawing records.csv', index=False, encoding='GBK')
    return name

if __name__ == '__main__':
    main()
