import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

print("ファイル選定してください：")

Filepath = filedialog.askopenfilename()
file_list = Filepath

if file_list == "":
    print("ファイルを選択してないため、プログラムが終了")
    exit()

print(file_list)

df1 = pd.read_csv(file_list)

new_list = []
values_sum = 0
item_list = []

df3 = pd.read_csv("reinforce_para_table.csv")
# select_pattern = df3.columns.tolist()[-1]
actions_list = df3.iloc[0:9, [1, 2, 3, 4, 5]].values.tolist()
print(actions_list)

df = df1.iloc[:,[1,2,3,4]]
print(df)


if not os.path.exists("save"):
    os.mkdir("save")

time_data = time.strftime('%H_%M_%S')
pic_file_path = "save/" + time_data
if not os.path.exists(pic_file_path):
    os.mkdir(pic_file_path)


item_name = ["case" + str(N) for N in range(len(actions_list))]
print(df1.loc[:, item_name])
ave_list = df1.loc[:, item_name].mean()
ave_values = ave_list.values.tolist()
ave_items = ave_values
ave_name = ave_list.index.values
ave_items.sort(reverse=True)
result_list = [ave_values.index(ave_items[0]) + 5, ave_values.index(ave_items[1]) + 5, ave_values.index(ave_items[2]) + 5]


Prob_list = df1.iloc[:, result_list]

Prob_list.plot(figsize=(14, 10), linewidth=2)
plt.title("注目した「パラメータの組み合わせ」の選択確率の推移\n （一番高い確率の「組み合せ」が選択される）", fontname="MS Gothic", fontsize=18)
plt.xlabel("試験回数", fontname="MS Gothic", fontsize=18)
plt.ylabel("「パラメータの組み合わせ」の選択される確率", fontname="MS Gothic", fontsize=18)
plt.savefig(pic_file_path + "/pic0.png")
plt.show()

for i in range(9):
    print(i)
    df_item = df[df["select_number"] == i]
    print(df_item)
    df_value = df_item["flicker_level"].value_counts(bins=(np.arange(0, 100, 10).tolist() + [200])).sort_index()
    item = ["(0, 10.0]", "(10.0, 20.0]", "(20.0, 30.0]", "(30.0, 40.0]", "(40.0, 50.0]", "(50.0, 60.0]", "(60.0, 70.0]", "(70.0, 80.0]", "(80.0, 90.0]", "(90~"]
    df_value.index = item
    item_list.append("pattern_" + str(actions_list[i]))
    values_sum += sum(df_value.tolist())
    new_list.append(df_value)

df2 = pd.concat(new_list, axis=1)
df2.columns = item_list
a_list = np.arange(0, 10, 1)
df3 = pd.DataFrame(df2/values_sum)
df3.to_csv("flicker_level_result.csv")
print(df2)
df3.plot(figsize=(14, 10))
plt.title("5秒間フリッカレベルで特定パターンの発生分布", fontname="MS Gothic", fontsize=18)
plt.xlabel("フリッカ（５秒間）のレベル区分", fontname="MS Gothic", fontsize=18)
plt.xticks(a_list, item)
plt.ylabel("各パラメータの組み合わせの発生比率", fontname="MS Gothic", fontsize=18)
plt.savefig(pic_file_path + "/pic1.png")
plt.show()

########################################################


df_select = df["select_number"]
print(df_select)
df_item = df_select.value_counts().sort_index()
var_sum = sum(df_item.tolist())
df_result = df_item/ var_sum
df_result.index = item_list
# print(df_result)

df_result.plot(figsize=(14, 10))
plt.title("選択されたパラメータの組合の比率", fontname="MS Gothic", fontsize=18)
plt.xlabel("「制御パラメータの組合せ」の番号", fontname="MS Gothic", fontsize=18)
a = np.arange(0, 9, 1)
plt.xticks(a, item_list)
plt.ylabel("選択された比率", fontname="MS Gothic", fontsize=18)
plt.savefig(pic_file_path + "/pic2.png")
plt.show()

########################################################################

df_select.plot(figsize=(14, 10))
plt.title("選択された「制御パラメータの組合せ」の推移", fontname="MS Gothic", fontsize=18)
plt.xlabel("試験回数", fontname="MS Gothic", fontsize=18)
plt.ylabel("選択された「制御パラメータの組合せ」の番号", fontname="MS Gothic", fontsize=18)
plt.yticks(a, item_list)
plt.savefig(pic_file_path + "/pic3.png")
plt.show()



########################################################################
df_reward = df["reward"].tolist()
ave_reward = []
l = 1
for item in df_reward:
    ave_reward.append(item/l)

    l = l + 1

# print(ave_reward)
df_ave_reward = pd.DataFrame(ave_reward, columns=["ave_reward"])
df_ave_reward.plot(figsize=(14, 10))
plt.title("得られた平均報酬レベルの推移", fontname="MS Gothic", fontsize=18)
plt.xlabel("試験回数", fontname="MS Gothic", fontsize=18)
plt.ylabel("平均報酬レベルの推移", fontname="MS Gothic", fontsize=18)
b_list = np.arange(0, 1.1, 0.1)
plt.yticks(b_list)
plt.savefig(pic_file_path + "/pic4.png")
plt.show()





