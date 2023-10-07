import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# input
NS1 = np.arange(0, 1, 0.001, np.float32)
TR1 = np.arange(0, 1, 0.001, np.float32)
CC1 = np.arange(0, 1, 0.001, np.float32)
# print(NS1)
# output
score1 = np.arange(0, 1, 0.01, np.float32)

# # build fuzzy variables
NS = ctrl.Antecedent(NS1, 'Normalized NS')
TR = ctrl.Antecedent(TR1, 'Normalized TR')
CC = ctrl.Antecedent(CC1, 'Normalized CC')
score = ctrl.Consequent(score1, 'Score')

# Defining fuzzy sets and their membership functions
NS['shortage'] = fuzz.gaussmf(NS1, 0.0000, 0.15)
NS['average'] = fuzz.gaussmf(NS1, 0.7, 0.15)
NS['sufficient'] = fuzz.gaussmf(NS1, 1.0, 0.15)
NS_plt1 = fuzz.gaussmf(NS1, 0.0000, 0.15)
NS_plt2 = fuzz.gaussmf(NS1, 0.7, 0.15)
NS_plt3 = fuzz.gaussmf(NS1, 1.0, 0.15)

TR['fast'] = fuzz.gaussmf(TR1, 0.0000, 0.14)
TR['middle'] = fuzz.gaussmf(TR1, 0.327, 0.14)
TR['slow'] = fuzz.gaussmf(TR1, 1.0, 0.14)
TR_plt1 = fuzz.gaussmf(TR1, 0.0000, 0.14)
TR_plt2 = fuzz.gaussmf(TR1, 0.327, 0.14)
TR_plt3 = fuzz.gaussmf(TR1, 1.0, 0.14)

CC['weak'] = fuzz.gaussmf(CC1, 0.2, 0.16)
CC['middle'] = fuzz.gaussmf(CC1, 0.650, 0.16)
CC['strong'] = fuzz.gaussmf(CC1, 1.0, 0.16)
CC_plt1 = fuzz.gaussmf(CC1, 0.2, 0.16)
CC_plt2 = fuzz.gaussmf(CC1, 0.650, 0.16)
CC_plt3 = fuzz.gaussmf(CC1, 1.0, 0.16)

score['L$_0$'] = fuzz.gaussmf(score1, 0.0, 0.13)
score['L$_1$'] = fuzz.gaussmf(score1, 0.2, 0.13)
score['L$_2$'] = fuzz.gaussmf(score1, 0.4, 0.13)
score['L$_3$'] = fuzz.gaussmf(score1, 0.6, 0.13)
score['L$_4$'] = fuzz.gaussmf(score1, 0.8, 0.13)
score['L$_5$'] = fuzz.gaussmf(score1, 1.0, 0.13)
score_plt1 = fuzz.gaussmf(score1, 0.0, 0.13)
score_plt2 = fuzz.gaussmf(score1, 0.2, 0.13)
score_plt3 = fuzz.gaussmf(score1, 0.4, 0.13)
score_plt4 = fuzz.gaussmf(score1, 0.6, 0.13)
score_plt5 = fuzz.gaussmf(score1, 0.8, 0.13)
score_plt6 = fuzz.gaussmf(score1, 1.0, 0.13)
# Visualization
# NS.view()
# # plt.savefig('./NS.pdf')
# TR.view()
# CC.view()
# score.view()

# defuzzy
score.defuzzify_method = 'mom'

# fuzzy rules
rule1 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['fast'] & CC['strong'])), consequent=score['L$_5$'],
                  label='score=L$_5$')
rule2 = ctrl.Rule(antecedent=(
        (NS['average'] & TR['fast'] & CC['strong']) | (NS['sufficient'] & TR['middle'] & CC['strong']) | (
        NS['sufficient'] & TR['slow'] & CC['strong'])), consequent=score['L$_4$'],
    label='score=L$_4')
rule3 = ctrl.Rule(antecedent=(
        (NS['shortage'] & TR['fast'] & CC['strong']) | (NS['average'] & TR['middle'] & CC['strong']) | (
        NS['shortage'] & TR['slow'] & CC['strong'])), consequent=score['L$_3$'],
    label='score=L$_3$')
rule4 = ctrl.Rule(antecedent=(
        (NS['shortage'] & TR['middle'] & CC['strong']) | (NS['sufficient'] & TR['fast'] & CC['middle']) | (
        NS['shortage'] & TR['fast'] & CC['middle'])), consequent=score['L$_2$'],
    label='score=L$_2$')
rule5 = ctrl.Rule(antecedent=(
        (NS['average'] & TR['slow'] & CC['strong']) | (NS['average'] & TR['fast'] & CC['middle']) | (
        NS['sufficient'] & TR['middle'] & CC['middle']) | (NS['average'] & TR['middle'] & CC['middle']) | (
                NS['shortage'] & TR['middle'] & CC['middle']) | (
                NS['sufficient'] & TR['slow'] & CC['middle']) | (NS['average'] & TR['slow'] & CC['middle']) | (
                NS['shortage'] & TR['slow'] & CC['middle']) | (NS['sufficient'] & TR['fast'] & CC['weak']) | (
                NS['average'] & TR['fast'] & CC['weak']) | (NS['shortage'] & TR['fast'] & CC['weak'])),
    consequent=score['L$_1$'],
    label='score=L$_1$')
rule6 = ctrl.Rule(antecedent=(
        (NS['sufficient'] & TR['middle'] & CC['weak']) | (NS['average'] & TR['middle'] & CC['weak']) | (
        NS['shortage'] & TR['middle'] & CC['weak']) | (NS['sufficient'] & TR['slow'] & CC['weak']) | (
                NS['average'] & TR['slow'] & CC['weak']) | (NS['shortage'] & TR['slow'] & CC['weak'])),
    consequent=score['L$_0$'],
    label='score=L$_0$')
# rule1 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['fast'] & CC['strong'])), consequent=score['L$_5$'],
#                   label='score=L$_5$')
# rule2 = ctrl.Rule(antecedent=((NS['average'] & TR['fast'] & CC['strong'])), consequent=score['L$_4$'],
#                   label='score=L$_4')
# rule3 = ctrl.Rule(antecedent=((NS['shortage'] & TR['fast'] & CC['strong'])), consequent=score['L$_3$'],
#                   label='score=L$_3$')
# rule4 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['middle'] & CC['strong'])), consequent=score['L$_4$'],
#                   label='score=L$_4$')
# rule5 = ctrl.Rule(antecedent=((NS['average'] & TR['middle'] & CC['strong'])), consequent=score['L$_3$'],
#                   label='score=L$_3$')
# rule6 = ctrl.Rule(antecedent=((NS['shortage'] & TR['middle'] & CC['strong'])), consequent=score['L$_2$'],
#                   label='score=L$_2$')
# rule7 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['slow'] & CC['strong'])), consequent=score['L$_4$'],
#                   label='score=L$_4$')
# rule8 = ctrl.Rule(antecedent=((NS['average'] & TR['slow'] & CC['strong'])), consequent=score['L$_1$'],
#                   label='score=L$_1$')
# rule9 = ctrl.Rule(antecedent=((NS['shortage'] & TR['slow'] & CC['strong'])), consequent=score['L$_3$'],
#                   label='score=L$_3$')
# rule10 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['fast'] & CC['middle'])), consequent=score['L$_2$'],
#                    label='score=L$_2$')
# rule11 = ctrl.Rule(antecedent=((NS['average'] & TR['fast'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule12 = ctrl.Rule(antecedent=((NS['shortage'] & TR['fast'] & CC['middle'])), consequent=score['L$_2$'],
#                    label='score=L$_2$')
# rule13 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['middle'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule14 = ctrl.Rule(antecedent=((NS['average'] & TR['middle'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule15 = ctrl.Rule(antecedent=((NS['shortage'] & TR['middle'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule16 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['slow'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule17 = ctrl.Rule(antecedent=((NS['average'] & TR['slow'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule18 = ctrl.Rule(antecedent=((NS['shortage'] & TR['slow'] & CC['middle'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule19 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['fast'] & CC['weak'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule20 = ctrl.Rule(antecedent=((NS['average'] & TR['fast'] & CC['weak'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule21 = ctrl.Rule(antecedent=((NS['shortage'] & TR['fast'] & CC['weak'])), consequent=score['L$_1$'],
#                    label='score=L$_1$')
# rule22 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['middle'] & CC['weak'])), consequent=score['L$_0$'],
#                    label='score=L$_0$')
# rule23 = ctrl.Rule(antecedent=((NS['average'] & TR['middle'] & CC['weak'])), consequent=score['L$_0$'],
#                    label='score=L$_0$')
# rule24 = ctrl.Rule(antecedent=((NS['shortage'] & TR['middle'] & CC['weak'])), consequent=score['L$_0$'],
#                    label='score=L$_0$')
# rule25 = ctrl.Rule(antecedent=((NS['sufficient'] & TR['slow'] & CC['weak'])), consequent=score['L$_0$'],
#                    label='score=L$_0$')
# rule26 = ctrl.Rule(antecedent=((NS['average'] & TR['slow'] & CC['weak'])), consequent=score['L$_0$'],
#                    label='score=L$_0$')
# rule27 = ctrl.Rule(antecedent=((NS['shortage'] & TR['slow'] & CC['weak'])), consequent=score['L$_0$'],
#                    label='score=L$_0$')

# System and operating environment inTRialization
# rule = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15,
#         rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27]
rule = [rule1, rule2, rule3, rule4, rule5, rule6]
score_ctrl = ctrl.ControlSystem(rule)
level = ctrl.ControlSystemSimulation(score_ctrl)

# input Output
print("\nplease input NS, TR, CC!!!")
# input_NS = input('NS:')
# input_TR = input('TR:')
# input_CC = input('CC:')
input_NS = 0.6
input_TR = 0.1
input_CC = 0.9
level.input['Normalized NS'] = float(input_NS)
level.input['Normalized TR'] = float(input_TR)
level.input['Normalized CC'] = float(input_CC)
level.compute()
print('level 为：', level.output['Score'])
font2 = {'family': 'Arial',
         'weight': 'normal',
         'size': 28,
         }
font1 = {'family': 'Arial',
         'weight': 'normal',
         'size': 16,
         }
# x_major_locator=MultipleLocator(0.5)
# #把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(0.5)
# #把y轴的刻度间隔设置为10，并存在变量里
fig, ax0 = plt.subplots(figsize=(4, 3))
# font2 = {'family': 'Arial',
#          'weight': 'normal',
#          'size': 28
#          }
# plt1
# NS.view()
# TR.view(), CC.view(),
# score.view(sim=level)
# plt.show()
# plt2
# 蓝色：'#1f77b4'
# 橘色：'#ff7f0e'
# 绿色：'#2ca02c'
color = ['#1f77b4', '#ff7f0e', '#2ca02c']
# # print('----------NS-----------')
# # ax0.plot(NS1, NS_plt1, label='shortage', marker='s', markersize=12, markevery=200,color='black')
# # ax0.plot(NS1, NS_plt2, label='average', marker='s', markersize=12, markevery=200, color='black')
# ax0.plot(NS1, NS_plt3, label='sufficient', marker='s', markersize=12, markevery=200, color='black')
# # ax0.plot(NS1, NS_plt1, label='shortage', color='black')
# # ax0.plot(NS1, NS_plt2, label='average', color='black')
# # ax0.plot(NS1, NS_plt3, label='sufficient', color='black')
# # ax0.plot([0.6, 0.6], [0.03, 1.0], color='black', linestyle='--', linewidth=0.5)
# # ax0.plot([0.6, 0.6], [0, 0.03], color='black', linewidth=0.5)
# # plt.xlabel('Normalized NS', font2)
# # plt.ylabel('Membership', font2)
# # plt.ylim(0, 1.5)
# plt.yticks([0, 0.5, 1.0], size=16)
# plt.xticks([0, 0.5, 1.0], size=16)
# ax0.legend(prop=font1, ncol=2, loc=2)
# plt.tight_layout()
# plt.savefig('C:///Users/DELL/Desktop/新建文件夹/NS_sufficient.png')
# plt.show()
# print('----------TR-----------')
# # ax0.plot(TR1, TR_plt1, label='slow', marker='^', markersize=12, markevery=200, color='blue')
# # ax0.plot(TR1, TR_plt2, label='medium', marker='^', markersize=12, markevery=200, color='blue')
# ax0.plot(TR1, TR_plt3, label='fast', marker='^', markersize=12, markevery=200, color='blue')
# # ax0.plot([0.327,0.327],[0.01,1.0], color='black', linestyle='--',linewidth=0.5)
# # ax0.plot(TR1, TR_plt1, label = 'fast', color = 'black')
# # ax0.plot(TR1, TR_plt2, label = 'medium', color = 'black')
# # ax0.plot(TR1, TR_plt3, label= 'slow', color = 'black')
# # plt.xlabel('Normalized TR', font2)
# # plt.ylabel('Membership', font2)
# # plt.ylim(0, 1.5)
# plt.yticks([0, 0.5, 1.0], size=16)
# plt.xticks([0, 0.5, 1.0], size=16)
# ax0.legend(prop=font1, ncol=2, loc=2)
# plt.tight_layout()
# plt.savefig('C:///Users/DELL/Desktop/新建文件夹/TR_fast.png')
# plt.show()
print('----------CC-----------')
# ax0.plot(CC1, CC_plt1, label = 'weak', marker='o', markersize=12, markevery=200,color='red')
# ax0.plot(CC1, CC_plt2, label = 'medium', marker='o', markersize=12, markevery=200,color='red')
ax0.plot(CC1, CC_plt3, label= 'strong', marker='o', markersize=12, markevery=200,color='red')
# ax0.plot([0.65,0.65],[0.01,1.0], color='black', linestyle='--',linewidth=0.5)
# ax0.plot(CC1, CC_plt1, label = 'weak',color = 'black')
# ax0.plot(CC1, CC_plt2, label = 'medium', color = 'black')
# ax0.plot(CC1, CC_plt3, label= 'strong',color = 'black')
# plt.xlabel('Normalized NC', font2)
# plt.ylabel('Membership', font2)
# plt.ylim(0, 1.5)
plt.yticks([0, 0.5, 1.0], size=16)
plt.xticks([0, 0.5, 1.0], size=16)
ax0.legend(prop=font1, ncol=2, loc=2)
plt.tight_layout()
plt.savefig('C:///Users/DELL/Desktop/新建文件夹/CC_strong.png')
plt.show()
# print('----------score-----------')
# # ax0.plot(score1, score_plt1, label='L$_0$', marker='o', markersize=12, markevery=16)
# ax0.plot(score1, score_plt2, label='L$_1$', marker='s', markersize=12, markevery=16,color=color[1])
# # ax0.plot(score1, score_plt3, label='L$_2$', marker='^', markersize=7, markevery=16)
# # ax0.plot(score1, score_plt4, label='L$_3$', marker='h', markersize=8, markevery=16)
# # ax0.plot(score1, score_plt5, label='L$_4$', marker='d', markersize=7, markevery=16)
# # ax0.plot(score1, score_plt6, label='L$_5$', marker='*', markersize=12, markevery=16,color='#8c564b')
# # ax0.plot(score1, score_plt2, label='L$_1$', color='black')
# # ax0.plot([0.0,1.0],[0,0], color='black')
# # plt.xlabel('Score', font2)
# # plt.ylabel('Membership', font2)
# # plt.ylim(0, 1.5)
# plt.yticks([0, 0.5, 1.0], size=16)
# plt.xticks([0, 0.5, 1.0], size=16)
# ax0.legend(prop=font1, ncol=3, loc=9)
# # ax0.legend(prop=font1, ncol=3, bbox_to_anchor=(0.2,0.1))
# plt.tight_layout()
# plt.savefig('./l1.png')
# plt.show()
