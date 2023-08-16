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


def data_sum(file_list):
    time_data = time.strftime('%Y_%m_%d_%H_%M_%S')
    file_name = "save/" + time_data + "_result.csv"

    with open(file_name, "w", newline="", encoding="SHIFT-JIS") as file:
        writer = csv.writer(file)

        with open(file_list[0], "r", newline="", encoding="SHIFT-JIS") as reader:
            lines = csv.reader(reader)
            for line in lines:
                writer.writerow(line)
                break

    with open(file_name, "a+", newline="", encoding="SHIFT-JIS") as file:
        writer = csv.writer(file)

        for item in file_list:
            print(item)
            with open(item, "r", newline="", encoding="SHIFT-JIS") as read:
                lines = csv.reader(read)
                next(lines)
                for line in lines:

                    if int(line[2]) <= 2 or line[13] == "0":
                        pass
                    else:
                        writer.writerow(line)

    return file_name


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
    models_path = "models/best_models_130200_0.9041.ckpt"
    batch_size = 1  #
    resize_height = 150  # 保存する図のheight
    resize_width = 150  # 保存する図のwidth
    depths = 3

    labels = np.loadtxt(labels_filename, str, delimiter='\t')
    input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')

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
    result_file_name = "Pulse_shift_count_result_.csv"

    with open(result_file_name, "w", newline="")as f:
        writer = csv.writer(f)
        item = ["Number", "Before_Pattern", "After_Pattern", "Count", "Continued_Time", "AI_mode"]
        writer.writerow(item)

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
    rows_num = len([None for l in open(file_name, "rb")])

    # while True:
    pic_path = "save/pic_save"
    if not os.path.exists(pic_path):
        os.mkdir(pic_path)

    ai_stable = 0
    ai_middle = 0
    ai_hunting = 0
    ai_slow_slow = 0
    ai_slow_big = 0
    ai_pulse_shift = 0
    ai_step_up = 0
    ai_step_down = 0

    auto_stable = 0
    auto_middle = 0
    auto_hunting = 0
    auto_slow_slow = 0
    auto_slow_big = 0
    auto_pulse_shift = 0
    auto_step_up = 0
    auto_step_down = 0

    with open(file_name, "r", encoding="SHIFT-JIS")as f:
        lines = csv.reader(f)
        next(lines)

        Number = 0
        Before_Pattern = ""
        PS_Count = 0

        with open(result_file_name, "a+", newline="") as file:
            writer = csv.writer(file)

            for line in lines:

                ai_mode = line[14]
                real_v = float(line[1])
                real_i = float(line[2])
                set_v = float(line[3])
                set_i = float(line[4])
                epos = float(line[5])

                if len(Set_I_list) < 166:

                    Real_V_list.append(real_v)
                    Real_I_list.append(real_i)
                    Set_V_list.append(set_v)
                    Set_I_list.append(set_i)
                    Epos_list.append(epos)

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
                                print("AIで識別でない、前回パターンは：", chart_label)
                            else:
                                chart_label = "Stable"  # default is "Stable"
                                print("AIで識別でない、defaultパターンは：", chart_label)

                    else:
                        pic = image

                    width, height = (750, 750)
                    pic = cv.resize(pic, (width, height), interpolation=cv.INTER_AREA)

                    if ai_mode == "0":
                        if chart_label == "Stable":
                            ai_stable += 1

                        elif chart_label == "Middle":
                            ai_middle += 1

                        elif chart_label == "Hunting":
                            ai_hunting += 1

                        elif chart_label == "Step_up":
                            ai_step_up += 1

                        elif chart_label == "Step_down":
                            ai_step_down += 1

                        elif chart_label == "Slow_big":
                            ai_slow_big += 1

                        elif chart_label == "Slow_slow":
                            ai_slow_slow += 1

                        else:
                            ai_pulse_shift += 1

                    else:
                        if chart_label == "Stable":
                            auto_stable += 1

                        elif chart_label == "Middle":
                            auto_middle += 1

                        elif chart_label == "Hunting":
                            auto_hunting += 1

                        elif chart_label == "Step_up":
                            auto_step_up += 1

                        elif chart_label == "Step_down":
                            auto_step_down += 1

                        elif chart_label == "Slow_big":
                            auto_slow_big += 1

                        elif chart_label == "Slow_slow":
                            auto_slow_slow += 1

                        else:
                            auto_pulse_shift += 1

                    if chart_label != 'Pulse_shift' and single_x == 0:
                        last_pic = pic
                        last_label = chart_label
                        # last_data_time = data_time

                    elif chart_label == 'Pulse_shift' and single_x == 0:
                        single_x = 1
                        PS_Count = 1
                        Number = Number + 1

                        if last_pic is not None:
                            Before_Pattern = last_label
                            # image_file = pic_path + "/" + last_data_time + "_" + last_label + ".png"
                            # cv.imwrite(image_file, last_pic)
                            #
                            # image_file = pic_path + "/" + data_time + "_" + chart_label + ".png"
                            # cv.imwrite(image_file, pic)

                            last_pic = None

                        else:
                            pass
                            # image_file = pic_path + "/" + data_time + "_" + chart_label + ".png"
                            # cv.imwrite(image_file, pic)

                    elif chart_label == 'Pulse_shift' and single_x == 1:
                        PS_Count = PS_Count + 1

                        # image_file = pic_path + "/" + data_time + "_" + chart_label + ".png"
                        # cv.imwrite(image_file, pic)

                    elif chart_label != 'Pulse_shift' and single_x == 1:
                        After_Pattern = chart_label
                        writer.writerow([Number, Before_Pattern, After_Pattern, PS_Count, PS_Count * 5, ai_mode])

                        # image_file = pic_path + "/" + data_time + "_" + chart_label + ".png"
                        # cv.imwrite(image_file, pic)

                        single_x = 0

                    else:
                        last_pic = pic
                        last_label = chart_label

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

                    last_pattern = chart_label

            ai_num_list = [ai_stable, ai_middle, ai_slow_slow, ai_slow_big, ai_step_up, ai_step_down, ai_hunting,
                           ai_pulse_shift]
            auto_num_list = [auto_stable, auto_middle, auto_slow_slow, auto_slow_big, auto_step_up, auto_step_down,
                             auto_hunting,
                             auto_pulse_shift]

            ai_pattern_sum = sum(ai_num_list)
            auto_pattern_sum = sum(auto_num_list)

            ai_ave_list = (np.array(ai_num_list) / np.array(ai_pattern_sum)).tolist()
            auto_ave_list = (np.array(auto_num_list) / np.array(auto_pattern_sum)).tolist()

            pattern_item = ["Stable", "Middle", "Slow_slow", "Slow_big", "Step_up", "Step_down", "Hunting",
                            "Pulse_shift"]

            with open("pattern_sum.csv", "w", newline="")as ptnum:
                pattern_writer = csv.writer(ptnum)
                pattern_writer.writerow(pattern_item)
                pattern_writer.writerow(ai_ave_list)
                pattern_writer.writerow(auto_ave_list)

    sess.close()

    return result_file_name, ai_pattern_sum, auto_pattern_sum

    # break


def create_PIC(data_file, begin_time, end_time, ai_pattern_sum, auto_pattern_sum):
    df = pd.read_csv("pattern_sum.csv", encoding="SHIFT-JIS")
    df.index = ["AI制御", "従来制御"]
    c = df.T
    c.plot(kind="line", figsize=(10, 8))
    ax = plt.gca()
    plt.title("各パターンの分布(溶解期)", fontname="MS Gothic", fontsize=18)
    plt.xlabel("パターン種類", fontname="MS Gothic", fontsize=18)
    plt.ylabel("各パターンの発生比率", fontname="MS Gothic", fontsize=18)
    # pattern_item = ["Stable", "Middle", "Slow_slow", "Slow_big", "Step_up", "Step_down", "Hunting", "Pulse_shift"]
    plt.legend(["AI制御", "従来制御"], prop={"family": "MS Gothic"}, fontsize=18)
    text_pic1 = "データ収集期間は：" + begin_time + "~" + end_time + ' ' \
                                                            'データファイルは:' + "pattern_sum.csv" + '                     ' \
                                                                                      "AI制御のパターン総数は：　　" + str(ai_pattern_sum) + '                                 ' \
                                                                                                                           "従来制御のパターン総数は：　　" + str(auto_pattern_sum)

    plt.text(0.4, 0.7, text_pic1, transform=ax.transAxes, wrap=True, fontname="MS Gothic", bbox=dict(boxstyle='round,pad=0.5', ec='black',lw=2 ,alpha=0.7))
    plt.savefig("pattern_count.png")
    plt.show()

    df = pd.read_csv(data_file, encoding="SHIFT-JIS")

    # ps_time_count_list = ps_time_count.index.tolist()
    # count_max = max(ps_time_count_list)

    ai_df = df[df["AI_mode"] == 1]
    ai_bp = ai_df["Before_Pattern"]
    ai_ap = ai_df["After_Pattern"]
    ai_ps_time = ai_df["Count"]

    auto_df = df[df["AI_mode"] == 0]
    auto_bp = auto_df["Before_Pattern"]
    auto_ap = auto_df["After_Pattern"]
    auto_ps_time = auto_df["Count"]

    ai_bp_sum = len(ai_bp.tolist())
    ai_ap_sum = len(ai_ap.tolist())
    auto_bp_sum = len(auto_bp.tolist())
    auto_ap_sum = len(auto_ap.tolist())
    ai_ps_time_sum = sum(ai_ps_time.tolist())
    auto_ps_time_sum = sum(auto_ps_time.tolist())

    ai_ps_time_count = ai_ps_time.value_counts(normalize=True, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1000000]).sort_index()
    auto_ps_time_count = auto_ps_time.value_counts(normalize=True, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1000000]).sort_index()

    ai_result1 = ai_bp.value_counts(normalize=True).sort_index()
    auto_result1 = auto_bp.value_counts(normalize=True).sort_index()

    ai_result2 = ai_ap.value_counts(normalize=True).sort_index()
    auto_result2 = auto_ap.value_counts(normalize=True).sort_index()

    # ai_result1.to_csv("ai_before_pattern.csv")
    # auto_result1.to_csv("auto_before_pattern.csv")

    # AIの統計列とAUTOの統計列を結合する
    a = pd.concat([ai_result1, auto_result1], axis=1)
    b = pd.concat([ai_result2, auto_result2], axis=1)
    c = pd.concat([ai_ps_time_count, auto_ps_time_count], axis=1)

    a.columns = ["AI制御", "従来制御"]
    b.columns = ["AI制御", "従来制御"]
    c.columns = ["AI制御", "従来制御"]
    c.index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "over 12"]
    item_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "over 12"]

    c.plot(kind="line", figsize=(10, 8))
    plt.title("「パルスシフト」パターンの継続回数と発生比率", fontname="MS Gothic", fontsize=18)
    plt.xlabel("継続回数（1回の期間は５sec）", fontname="MS Gothic", fontsize=18)
    plt.ylabel("発生比率", fontname="MS Gothic", fontsize=18)
    my_x_ticks = np.arange(0, 13, 1)
    my_y_ticks = np.arange(0, 1.1, 0.1)
    plt.yticks(my_y_ticks)
    plt.xticks(my_x_ticks, item_list)
    plt.legend(["AI制御", "従来制御"], prop={"family": "MS Gothic"}, fontsize=18)
    plt.xticks(rotation=0)
    text_time = "データ収集期間は：" + str(begin_time) + "~" + end_time + ' ' \
                                                                 "データファイルは:" + data_file + '                     ' \
                                                                                           "AI制御で「パルスシフト」の総数：" + str(ai_ps_time_sum) + '                                 '\
                                                                                                                                       "AUTO制御で「パルスシフト」の総数：" + str(auto_ps_time_sum)

    plt.text(4, 0.8, text_time, fontname="MS Gothic", wrap=True, bbox=dict(boxstyle='round,pad=0.5', ec='black',lw=2 ,alpha=0.7))
    plt.savefig("PS_time_count.png")

    a.plot(kind="bar", figsize=(10, 8))
    plt.title("「パルスシフト」パターンに入る直前のパターンの分布", fontname="MS Gothic", fontsize=18)
    plt.xlabel("パターン種類", fontname="MS Gothic", fontsize=18)
    plt.ylabel("各パターンの発生比率", fontname="MS Gothic", fontsize=18)
    plt.xticks(rotation=0)
    plt.legend(["AI制御", "従来制御"], prop={"family": "MS Gothic"}, fontsize=18)
    text_pic2 = "データ収集期間は：" + begin_time + "~" + end_time + ' ' \
                                                            'データファイルは:' + data_file + '                     ' \
                                                                                      "AI制御の直前パターン総数は：" + str(ai_bp_sum) + '                                 ' \
                                                                                                                           "従来制御の直前パターン総数は：" + str(auto_bp_sum)

    plt.text(0.3, 0.7, text_pic2, transform=ax.transAxes, wrap=True, fontname="MS Gothic", bbox=dict(boxstyle='round,pad=0.5', ec='black',lw=2 ,alpha=0.7))

    plt.savefig("PS_Before_Pattern.png")

    b.plot(kind="bar", figsize=(10, 8))
    plt.title("「パルスシフト」パターンから脱出後のパターンの分布", fontname="MS Gothic", fontsize=18)
    plt.xlabel("パターン種類", fontname="MS Gothic", fontsize=18)
    plt.ylabel("各パターンの発生比率", fontname="MS Gothic", fontsize=18)
    plt.xticks(rotation=0)
    plt.legend(["AI制御", "従来制御"], prop={"family": "MS Gothic"}, fontsize=18)
    text_pic3 = "データ収集期間は：" + begin_time + "~" + end_time + ' ' \
                                                            'データファイルは:' + data_file + '                     ' \
                                                                                      "AI制御の脱出パターン総数は：" + str(ai_ap_sum) + '                                 ' \
                                                                                                                           "従来制御の脱出パターン総数は：" + str(auto_ap_sum)

    plt.text(0.3, 0.7, text_pic3, transform=ax.transAxes, wrap=True, fontname="MS Gothic",bbox=dict(boxstyle='round,pad=0.5', ec='black',lw=2 ,alpha=0.7))
    plt.savefig("PS_After_Pattern.png")
    plt.show()


def time_read(file_name):
    df = pd.read_csv(file_name, encoding="SHIFT-JIS")
    df2 = df["data_time"].tolist()

    return df2[0], df2[-1]


if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()

    # 生成したもののフォルダ
    if not os.path.exists("save"):
        os.mkdir("save")

    # 処理されたデータのフォルダ
    if not os.path.exists("data"):
        os.mkdir("data")

    print("フォルダ選定　or  ファイル選定：")
    select_word = input("入力：")
    file_list = []

    if select_word == "ファイル":
        Filepath = filedialog.askopenfilenames()
        file_list = Filepath

    elif select_word == "フォルダ":
        File_Path = filedialog.askdirectory()
        file_list = glob.glob(File_Path + "/*.csv")

    else:
        print("識別できない、ソフト終了します")
        time.sleep(1)
        exit()

    print(file_list)

    file_name = data_sum(file_list)
    begin_time, end_time = time_read(file_name)
    data_file_name, ai_pattern_sum, auto_pattern_sum = Pulse_shift_analyze(file_name)
    create_PIC(data_file_name, begin_time, end_time, ai_pattern_sum, auto_pattern_sum)

