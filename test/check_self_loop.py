with open('../../data/Car_Hacking_Challenge_Dataset_rev20Mar2021/0_Preliminary/0_Training/Pre_train_D_1_2_dataset/Pre_train_D_1_2_200_A.txt') as f:
    line_num = 0
    equl_num = 0
    for line in f:
        line_num += 1
        nodes = line.strip('\n').split(',')
        a = int(nodes[0])
        b = int(nodes[1])
        if a == b:
            equl_num+=1
            print(f'{line_num} {a} == {b}')
        # print(f'{line_num} {a} == {b}')
        # break
print('end')
print(equl_num)