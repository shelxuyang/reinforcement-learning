import tkinter as tk
from tkinter import filedialog
import glob
import csv
import time
import slim.nets.inception_v3 as inception_v3
from create_tf_record import *
import tensorflow.contrib.slim as slim
import pandas as pd
import os  # 保存するディレクトリを指定するため
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def data_process(file_list):
    time_data = time.strftime('%Y_%m_%d_%H_%M_%S')

    df = pd.read_csv(file_list, encoding="SHIFT-JIS")
    df_melt = df[df["積算電力量[MWh]"] <= 100]
    df_refine = df[df["積算電力量[MWh]"] >= 110]

    df_melt_file = "save/" + time_data + "melt_result.csv"
    df_refine_file = "save/" + time_data + "refine_result.csv"

    df_melt.to_csv(df_melt_file, encoding="SHIFT-JIS")
    df_refine.to_csv(df_refine_file, encoding="SHIFT-JIS")

    df2 = df["data_time"].tolist()

    return df_melt_file, df_refine_file, df2[0], df2[-1]


def data_rolling(d_set_Vdev):
    i = 0
    d_set_Vdevl = []
    while i < 150:
        d_set_Vdevl.append(sum(d_set_Vdev[i:i + 16]) / 16)
        i = i + 1

    return d_set_Vdevl


def Data_to_pic(Set_I_list, Set_V_list, Real_I_list, Real_V_list, Epos_list):
    if min(Real_I_list) >= -5:  # 電流が切れたdata setは処理しない <=== (3/3)やめるため -5に設定
        d_set_Vdev = np.array(Real_V_list) - np.array(Set_V_list)  # （実績電圧ー設定電圧）に変換するため符号を逆転
        d_set_Curr = (np.array(Real_I_list) - np.array(Set_I_list))  # 電流偏差の算出???????????????????????????????????
        Epos_center = (max(Epos_list) + min(Epos_list)) * 0.5
        Epos_value = (Epos_center - np.array(Epos_list)) * 5

        # 移動平均の方法　https://note.nkmk.me/python-pandas-rolling/　# 移動平均　windowで個数指定
        # d_set_Vdevl = -d_set_Vdev.rolling(16, center=True).mean()  # 電圧変動の周波数の低い成分
        d_set_Vdevl = data_rolling(d_set_Vdev)

        #       特徴量の算出　---------------------------------------------------------------------------------------------------------
        # 電圧変動（高周波、低周波）の標準偏差、ゼロ交差数
        mean_Curr = np.mean(d_set_Curr)  # 電流偏差の平均
        std_Vdevl = np.std(d_set_Vdevl)  # 電圧変動低周波分の標準偏差

        # 変動パターンの分類　categoryを0〜8 に分けた
        if mean_Curr < -3:  # 電流偏差の平均が-3kA以下は　Pulse-shift発生と見做す
            chart_label = 'Pulse_shift'

        elif std_Vdevl < 20:  # 電圧偏差の標準偏差がMiddleの 1/2以下をStableに
            chart_label = 'Stable'

        elif std_Vdevl >= 20 and std_Vdevl < 40:  # 電圧偏差の標準偏差が20〜40VをMiddle（＝精錬期のレベル）
            chart_label = 'Middle'

        else:
            chart_label = 'Others'

        Epos_list = list(Epos_value)[8:-7]
        Curr_list = list(d_set_Curr * 10)[8:-7]
        Vdevl_list = list(d_set_Vdevl)
        y_size = 1000

        img = np.zeros((y_size, 300, 3))
        img = img + np.array([255, 255, 255])  # 白い背景に変換する

        point_red = (0, 0, 255)
        point_green = (0, 255, 0)
        point_blue = (255, 0, 0)
        thickness = 2

        i = 0
        j = 0
        while i < (len(Epos_list) - 2):

            y1_epos = Epos_list[i]
            y2_epos = Epos_list[i + 1]
            y1_curr = Curr_list[i]
            y2_curr = Curr_list[i + 1]
            y1_vdevl = Vdevl_list[i]
            y2_vdevl = Vdevl_list[i + 1]

            item_list = [y1_epos, y2_epos, y1_vdevl, y2_vdevl, y1_curr, y2_curr]  # 座標系を修正する

            for item in item_list:
                if item < 0:
                    item_list[item_list.index(item)] = abs(item) + y_size * 0.5
                else:
                    item_list[item_list.index(item)] = y_size * 0.5 - abs(item)

            point_s_Epos = (j, int(item_list[0]))
            point_e_Epos = (j + 1, int(item_list[1]))
            img = cv.line(img, point_s_Epos, point_e_Epos, point_blue, thickness)  # 電極位置の変化

            point_s_Vdevl = (j, int(item_list[2]))
            point_e_Vdevl = (j + 1, int(item_list[3]))
            img = cv.line(img, point_s_Vdevl, point_e_Vdevl, point_red, thickness)  # 電圧偏差の低周波の変化

            # point_s_Icurr = (j, int(item_list[4]))
            # point_e_Icurr = (j + 1, int(item_list[5]))
            # img = cv.line(img, point_s_Icurr, point_e_Icurr, point_green, thickness)  # 電圧偏差の低周波の変化

            i += 1
            j += 2

        width, height = (750, 750)
        img = cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

        return img, chart_label
    else:
        pass


def read_image(filename, resize_height, resize_width, normalization=False):
    '''
    读取图片数据,默认返回的是uint8,[0,255]
    :param filename:
    :param resize_height:
    :param resize_width:
    :param normalization:是否归一化到[0.,1.0]
    :return: 返回的图片数据
    '''

    bgr_image = filename
    if len(bgr_image.shape) == 2:  # 若是灰度图则转为三通道
        print("Warning:gray image", filename)
        bgr_image = cv.cvtColor(bgr_image, cv.COLOR_GRAY2BGR)

    rgb_image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)  # 将BGR转为RGB
    # show_image(filename,rgb_image)
    # rgb_image=Image.open(filename)
    if resize_height > 0 and resize_width > 0:
        rgb_image = cv.resize(rgb_image, (resize_width, resize_height))
    rgb_image = np.asanyarray(rgb_image)
    if normalization:
        # 不能写成:rgb_image=rgb_image/255
        rgb_image = rgb_image / 255.0
    # show_image("src resize image",image)
    return rgb_image


def Pulse_shift_analyze(file_name):

    labels_nums = 7
    labels_filename = 'label.txt'
    models_path = "models_s/best_models_27800_0.9099.ckpt"
    # models_path = "models/best_models_130200_0.9041.ckpt"
    batch_size = 1  #
    resize_height = 150  # 保存する図のheight
    resize_width = 150  # 保存する図のwidth
    depths = 3

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=1.0,
                                                    is_training=False)

    # outputをsoftmaxで分けて,最大確率の所属を求める
    score = tf.nn.softmax(out, name='pre')
    class_id = tf.argmax(score, 1)

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, models_path)

    # *********************************************************************************

    time_data = time.strftime('%H_%M_%S')
    # file_name = "save/" + time_data + "_result.csv"
    result_file_name = "save/" + time_data + "_Flicker_count_result_.csv"

    # *********************************************************************************
    # サンプリングのリストを定義する
    Set_I_list = []  # 設定電流
    Set_V_list = []  # 設定電圧
    Real_I_list = []  # 実績電流
    Real_V_list = []  # 実績電圧
    Epos_list = []  # 電極位置

    last_pattern = ""
    single_x = 0
    last_pic = None

    # while True:
    pic_path = "save/pic_save"
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    ai_stable = []
    ai_middle = []
    ai_hunting = []
    ai_slow_slow = []
    ai_slow_big = []
    ai_pulse_shift = []
    ai_step_up = []
    ai_step_down = []

    auto_stable = []
    auto_middle = []
    auto_hunting = []
    auto_slow_slow = []
    auto_slow_big = []
    auto_pulse_shift = []
    auto_step_up = []
    auto_step_down = []

    flicker_list = []
    melt_patter_sum = []

    with open(file_name, "r", encoding="SHIFT-JIS")as f:
        lines = csv.reader(f)
        next(lines)

        for line in lines:

            ai_mode = line[15]
            chart_num = line[1]
            flicker_value = line[26]
            real_v = float(line[2])
            real_i = float(line[3])
            set_v = float(line[4])
            set_i = float(line[5])
            epos = float(line[6])

            if len(Set_I_list) < 166:

                Real_V_list.append(real_v)
                Real_I_list.append(real_i)
                Set_V_list.append(set_v)
                Set_I_list.append(set_i)
                Epos_list.append(epos)
                flicker_list.append(flicker_value)

            else:
                # 画像生成とロジック判定
                image, chart_label = Data_to_pic(Set_I_list, Set_V_list, Real_I_list, Real_V_list, Epos_list)
                print("ロジック判定された結果は：", chart_label)

                if chart_label == "Others":

                    pic = image
                    image = np.array(image, dtype=np.float32)
                    im = read_image(image, resize_height, resize_width, normalization=True)
                    im = im[np.newaxis, :]
                    pre_score, pre_label = sess.run([score, class_id], feed_dict={input_images: im})
                    max_score = pre_score[0, pre_label]

                    if max_score > 0.8:
                        chart_label = labels[pre_label][0]
                        print("AIで識別されたパターンは：", chart_label)

                    else:
                        if last_pattern != "":
                            chart_label = last_pattern
                            print("AIで識別できない、前回パターンは：", chart_label)
                        else:
                            chart_label = "Stable"  # default is "Stable"
                            print("AIで識別できない、defaultパターンは：", chart_label)

                else:
                    pic = image

                melt_patter_sum.append(chart_label)

                if ai_mode == "0":
                    if chart_label == "Stable":
                        ai_stable += flicker_list

                    elif chart_label == "Middle":
                        ai_middle += flicker_list

                    elif chart_label == "Hunting":
                        ai_hunting += flicker_list

                    elif chart_label == "Step_up":
                        ai_step_up += flicker_list

                    elif chart_label == "Step_down":
                        ai_step_down += flicker_list

                    elif chart_label == "Slow_big":
                        ai_slow_big += flicker_list

                    elif chart_label == "Slow_slow":
                        ai_slow_slow += flicker_list

                    else:
                        ai_pulse_shift += flicker_list

                else:
                    if chart_label == "Stable":
                        auto_stable += flicker_list

                    elif chart_label == "Middle":
                        auto_middle += flicker_list

                    elif chart_label == "Hunting":
                        auto_hunting += flicker_list

                    elif chart_label == "Step_up":
                        auto_step_up += flicker_list

                    elif chart_label == "Step_down":
                        auto_step_down += flicker_list

                    elif chart_label == "Slow_big":
                        auto_slow_big += flicker_list

                    elif chart_label == "Slow_slow":
                        auto_slow_slow += flicker_list

                    else:
                        auto_pulse_shift += flicker_list

                # Set_I_list = Set_I_list[33:]
                # Set_V_list = Set_V_list[33:]
                # Real_I_list = Real_I_list[33:]
                # Real_V_list = Real_V_list[33:]
                # Epos_list = Epos_list[33:]

                Set_I_list = []
                Set_V_list = []
                Real_I_list = []
                Real_V_list = []
                Epos_list = []
                flicker_list = []

                last_pattern = chart_label

        # pattern_item = ["Stable", "Middle", "Slow_slow", "Slow_big", "Step_up", "Step_down", "Hunting",
        #                 "Pulse_shift"]
        pattern_item = [ai_stable, ai_middle, ai_slow_slow, ai_slow_big, ai_step_up, ai_step_down,
                        ai_hunting,
                        ai_pulse_shift, auto_stable, auto_middle, auto_slow_slow, auto_slow_big,
                        auto_step_up, auto_step_down, auto_hunting,
                        auto_pulse_shift]

        pattern_num_list = []

        for item in pattern_item:
            pattern_num_list.append(len(item))

        num_max = max(pattern_num_list)

        num = max([len(ai_middle), len(ai_stable), len(auto_stable), len(auto_middle)])
        with open(result_file_name, "w", newline="", encoding="utf-8")as ptnum:
            pattern_writer = csv.writer(ptnum)
            pattern_writer.writerow(["pattern"] + np.arange(0, num_max, 1).tolist())
            pattern_writer.writerow(["ai_stable"] + ai_stable + (num_max - len(ai_stable)) * [0])
            pattern_writer.writerow(["ai_middle"] + ai_middle + (num_max - len(ai_middle)) * [0])
            pattern_writer.writerow(["ai_slow_slow"] + ai_slow_slow + (num_max - len(ai_slow_slow)) * [0])
            pattern_writer.writerow(["ai_slow_big"] + ai_slow_big + (num_max - len(ai_slow_big)) * [0])
            pattern_writer.writerow(["ai_step_up"] + ai_step_up + (num_max - len(ai_step_up)) * [0])
            pattern_writer.writerow(["ai_step_down"] + ai_step_down + (num_max - len(ai_step_down)) * [0])
            pattern_writer.writerow(["ai_hunting"] + ai_hunting + (num_max - len(ai_hunting)) * [0])
            pattern_writer.writerow(["ai_pulse_shift"] + ai_pulse_shift + (num_max - len(ai_pulse_shift)) * [0])

    sess.close()

    return result_file_name, melt_patter_sum

    # break


def create_pic(result_file_name, melt_patter_sum, begin_time, end_time):
    global df_value_counts

    i = 1

    Stable_list = []
    Middle_list  = []
    Slow_slow_list  = []
    Slow_big_list  = []
    Step_up_list  = []
    Step_down_list  = []
    Hunting_list  = []
    Pulse_shift_list  = []

    while i < len(melt_patter_sum):

        if melt_patter_sum[-i] == "Stable":
            Stable_list.append(melt_patter_sum[-i-1])

        elif melt_patter_sum[-i] == "Middle":
            Middle_list.append(melt_patter_sum[-i - 1])

        elif melt_patter_sum[-i] == "Slow_slow":
            Slow_slow_list.append(melt_patter_sum[-i - 1])

        elif melt_patter_sum[-i] == "Slow_big":
            Slow_big_list.append(melt_patter_sum[-i - 1])

        elif melt_patter_sum[-i] == "Step_up":
            Step_up_list.append(melt_patter_sum[-i - 1])

        elif melt_patter_sum[-i] == "Step_down":
            Step_down_list.append(melt_patter_sum[-i - 1])

        elif melt_patter_sum[-i] == "Hunting":
            Hunting_list.append(melt_patter_sum[-i - 1])

        else:
            Pulse_shift_list.append(melt_patter_sum[-i-1])

        i = i + 1

    Stable_result = pd.Series(Stable_list).value_counts()
    Middle_result = pd.Series(Middle_list).value_counts()
    Slow_slow_result = pd.Series(Slow_slow_list).value_counts()
    Slow_big_result = pd.Series(Slow_big_list).value_counts()
    Step_up_result = pd.Series(Step_up_list).value_counts()
    Step_down_result = pd.Series(Step_down_list).value_counts()
    Hunting_result = pd.Series(Hunting_list).value_counts()
    Pulse_shift_result = pd.Series(Pulse_shift_list).value_counts()

    result_sum = [Stable_result, Middle_result, Slow_slow_result, Slow_big_result, Step_down_result, Step_up_result, Hunting_result, Pulse_shift_result]
    pattern_item = ["Stable", "Middle", "Slow_slow", "Slow_big", "Step_up", "Step_down", "Hunting", "Pulse_shift"]
    # pattern_item2 = [Stable, Middle, Slow_slow, Slow_big, Step_up, Step_down, Hunting, Pulse_shift]

    #stableの各直前パターンの分布
    stable_add1 = [x for x in Stable_result.index.tolist() if x in pattern_item]
    stable_add2 = [x for x in pattern_item if x not in Stable_result.index]

    stable_orignal = pd.Series(Stable_result.tolist(), index=stable_add1)
    stable_add_list = pd.Series([0]*len(stable_add2), index=stable_add2, name="y")

    stable_contact = pd.concat([stable_orignal, stable_add_list, pd.Series(sum(Stable_result.tolist()), index=["SUM"])])

    #siddleの各直前パターンの分布
    middle_add1 = [x for x in Middle_result.index.tolist() if x in pattern_item ]
    middle_add2 = [x for x in pattern_item if x not in Middle_result.index]

    middle_orignal = pd.Series(Middle_result.tolist(), index=middle_add1)
    middle_add_list = pd.Series([0]*len(middle_add2), index=middle_add2, name="y")

    middle_contact = pd.concat([middle_orignal, middle_add_list, pd.Series(sum(Middle_result.tolist()), index=["SUM"])])

    #slow_slowの各直前パターンの分布
    slow_slow_add1 = [x for x in Slow_slow_result.index.tolist() if x in pattern_item]
    slow_slow_add2 = [x for x in pattern_item if x not in Slow_slow_result.index]

    slow_slow_orignal = pd.Series(Slow_slow_result.tolist(), index=slow_slow_add1)
    slow_slow_add_list = pd.Series([0]*len(slow_slow_add2), index=slow_slow_add2, name="y")

    slow_slow_contact = pd.concat([slow_slow_orignal, slow_slow_add_list, pd.Series(sum(Slow_slow_result.tolist()), index=["SUM"])])

    #Slow_bigの各直前パターンの分布
    slow_big_add1 = [x for x in Slow_big_result.index.tolist() if x in pattern_item]
    slow_big_add2 = [x for x in pattern_item if x not in Slow_big_result.index]

    slow_big_orignal = pd.Series(Slow_big_result.tolist(), index=slow_big_add1)
    slow_big_add_list = pd.Series([0]*len(slow_big_add2), index=slow_big_add2, name="y")

    slow_big_contact = pd.concat([slow_big_orignal, slow_big_add_list, pd.Series(sum(Slow_big_result.tolist()), index=["SUM"])])

    #step_upの各直前パターンの分布
    step_up_add1 = [x for x in Step_up_result.index.tolist() if x in pattern_item]
    step_up_add2 = [x for x in pattern_item if x not in Step_up_result.index]

    step_up_orignal = pd.Series(Step_up_result.tolist(), index=step_up_add1)
    step_up_add_list = pd.Series([0]*len(step_up_add2), index=step_up_add2, name="y")

    step_up_contact = pd.concat([step_up_orignal, step_up_add_list, pd.Series(sum(Step_up_result.tolist()), index=["SUM"])])

    # step_downの各直前パターンの分布
    step_down_add1 = [x for x in Step_down_result.index.tolist() if x in pattern_item]
    step_down_add2 = [x for x in pattern_item if x not in Step_down_result.index]

    step_down_orignal = pd.Series(Step_down_result.tolist(), index=step_down_add1)
    step_down_add_list = pd.Series([0] * len(step_down_add2), index=step_down_add2, name="y")

    step_down_contact = pd.concat([step_down_orignal, step_down_add_list, pd.Series(sum(Step_down_result.tolist()), index=["SUM"])])

    # huntingの各直前パターンの分布
    hunting_add1 = [x for x in Hunting_result.index.tolist() if x in pattern_item]
    hunting_add2 = [x for x in pattern_item if x not in Hunting_result.index]

    hunting_orignal = pd.Series(Hunting_result.tolist(), index=hunting_add1)
    hunting_add_list = pd.Series([0]*len(hunting_add2), index=hunting_add2, name="y")

    hunting_contact = pd.concat([hunting_orignal, hunting_add_list, pd.Series(sum(Hunting_result.tolist()), index=["SUM"])])

    # pulse_shiftの各直前パターンの分布
    pulse_shift_add1 = [x for x in Pulse_shift_result.index.tolist() if x in pattern_item]
    pulse_shift_add2 = [x for x in pattern_item if x not in Pulse_shift_result.index]

    pulse_shift_orignal = pd.Series(Pulse_shift_result.tolist(), index=pulse_shift_add1)
    pulse_shift_add_list = pd.Series([0]*len(pulse_shift_add2), index=pulse_shift_add2, name="y")

    pulse_shift_contact = pd.concat([pulse_shift_orignal, pulse_shift_add_list, pd.Series(sum(Pulse_shift_result.tolist()), index=["SUM"])])

    sum_contact = [stable_contact, middle_contact, slow_slow_contact, slow_big_contact, step_up_contact, step_down_contact, hunting_contact, pulse_shift_contact]

    pattern_count_result = pd.concat(sum_contact, axis=1)
    pattern_count_result.columns = pattern_item
    pattern_count_result.to_csv("pattern_count_result.csv")

    df = pd.read_csv("pattern_count_result.csv")
    df2 = df.T
    i = 0
    sum_list = []
    item_list = []

    while i < 8:
        var = (df2[i].tolist())[1:]
        sum_list.append(sum(var))
        item_list.append((df2[i].tolist())[0])
        # print(var)
        i = i + 1

    item_add = pd.Series(sum_list + [""], index=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    # item_add = pd.Series(sum_list + [""], index=pattern_item + [""])
    df_contact = df.assign(SUM=item_add)

    real_item = df_contact.iloc[:,0].tolist()
    df_contact.index = real_item

    df_final = df_contact.drop(df_contact.columns[[0]], axis=1)
    df_final = df_final.reindex(pattern_item + ["SUM"])

    df_final.to_csv("pattern_count_result.csv")


if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()

    print("ファイル選定してください：")

    Filepath = filedialog.askopenfilename()
    file_list = Filepath

    if file_list == "":
        print("ファイルを選択してないため、プログラムが終了")
        exit()

    print(file_list)

    df_melt, df_refine, begin_time, end_time = data_process(file_list)

    print("溶解期に関連する解析結果図を生成します！")
    result_file_name, melt_patter_sum = Pulse_shift_analyze(df_melt)
    create_pic(result_file_name, melt_patter_sum, begin_time, end_time)