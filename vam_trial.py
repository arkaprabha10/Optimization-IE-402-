from collections import defaultdict
import numpy
import sys
import scipy
##########################

def max_value_in_matrix(X):
    final_ans=0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if(X[i][j]!=0):
                final_ans+=X[i][j]*another_costs[i][j]
    return final_ans

def look_for_trend(row_present_size, column_present_size, another_matrix):
    another_matrix = another_matrix.copy()
    present_row = [0]*row_present_size
    present_colum =[0]*column_present_size
    for ind_i, ind_j in another_matrix:
        present_row[ind_i] += 1
        present_colum[ind_j] += 1
    while True:
        bool_value = True
        for k in range(row_present_size):
            if present_row[k] == 1:
                bool_value = False
                for ind_i, ind_j in another_matrix:
                    if ind_i == k:
                        present_colum[ind_j] -= 1
                        present_row[ind_i] = 0
                        another_matrix.remove((ind_i, ind_j))
                        break
        for k in range(column_present_size):
            if present_colum[k] == 1:
                bool_value = False
                for ind_i, ind_j in another_matrix:
                    if ind_j == k:
                        present_row[ind_i] -= 1
                        present_colum[ind_j] = 0
                        another_matrix.remove((ind_i, ind_j))
                        break
        if bool_value:
            return another_matrix
        if len(another_matrix) < 4:
            return None


def search_possible(TP_matrix, another_matrix):
    infinite_value = float('inf')
    u_matrix_tp = [infinite_value]*len(TP_matrix)
    v_matrix_tp = [infinite_value]*len(TP_matrix[0])
    u_matrix_tp[0] = 0
    for _ in range(len(another_matrix)):
        for ind_i, ind_j in another_matrix:
            if v_matrix_tp[ind_j] == infinite_value and u_matrix_tp[ind_i] != infinite_value:
                #print(v_matrix_tp)
                #print(u_matrix_tp)
                v_matrix_tp[ind_j] = TP_matrix[ind_i][ind_j] - u_matrix_tp[ind_i]
                break
            elif u_matrix_tp[ind_i] == infinite_value and v_matrix_tp[ind_j] != infinite_value:
                u_matrix_tp[ind_i] = TP_matrix[ind_i][ind_j] - v_matrix_tp[ind_j]
                #print(v_matrix_tp)
                #print(u_matrix_tp)
                break
    #print(v_matrix_tp)
    #print(u_matrix_tp)
    return u_matrix_tp, v_matrix_tp


def break_trend(index_0, index_J_0, matrix_possible_cycle):
    negative_set = set()
    positive_set = set()
    positive_set.add((index_0, index_J_0))
    for _ in range(len(matrix_possible_cycle) >> 1):
        for ind_i, ind_j in matrix_possible_cycle:
            if ind_i == index_0 and ind_j != index_J_0:
                neg_i= ind_i
                neg_j = ind_j
                break
        negative_set.add((neg_i, neg_j))
        for ind_i, ind_j in matrix_possible_cycle:
            if ind_j == neg_j and ind_i != neg_i:
                index_0= ind_i
                index_J_0 = ind_j
                break
        positive_set.add((index_0, index_J_0))
    return negative_set, positive_set


def answer_matrix(matrix_supply, matrix_demand, cost_matrix,Final_cost_matrix,present_in_basis,not_present_in_basis):
    length_1, length_2 = len(matrix_supply), len(matrix_demand)
    length_difference = sum(matrix_supply) - sum(matrix_demand)
    if length_difference < 0:
        matrix_supply.append(abs(length_difference))
        length_1 += 1
        cost_matrix.append([0 for _ in range(length_2)])
    elif length_difference > 0:
        matrix_demand.append(length_difference)
        length_2 += 1
        for row_present in cost_matrix:
            row_present.append(0)
    
    if len(not_present_in_basis) == 0:
        return Final_cost_matrix

    while 0!=1:
        matrix_u_tp, matrix_v_tp = search_possible(cost_matrix, present_in_basis)
        index_0, index_J_0 = min(not_present_in_basis, key=lambda x: cost_matrix[x[0]][x[1]]-matrix_u_tp[x[0]]-matrix_v_tp[x[1]])
        d_least_value = cost_matrix[index_0][index_J_0]-matrix_u_tp[index_0]-matrix_v_tp[index_J_0]
        if d_least_value >= 0:
            return Final_cost_matrix

        present_in_basis.add((index_0, index_J_0))
        not_present_in_basis.remove((index_0, index_J_0))
        matrix_possible_cycle = look_for_trend(length_1, length_2, present_in_basis)
        negative_set, positive_set = break_trend(index_0, index_J_0, matrix_possible_cycle)
        special_i_value, special_j_value = min(negative_set, key=lambda el: Final_cost_matrix[el[0]][el[1]])
        theta = Final_cost_matrix[special_i_value][special_j_value]
        for value in positive_set:
            Final_cost_matrix[value[0]][value[1]] += theta
        for value in negative_set:
            Final_cost_matrix[value[0]][value[1]] -= theta
        present_in_basis.remove((special_i_value, special_j_value))
        not_present_in_basis.add((special_i_value, special_j_value))

#############################

r,c=input().split(' ')
col_ind=[]
row_ind=[]
for i in range(int(c)):
    #col_ind.append(str(chr(65+int(i))))
    col_ind.append('D'+str(int(i)+1))
for i in range(int(r)):
    #row_ind.append(str(chr(77+int(i))))
    row_ind.append('O'+str(int(i)+1))

trial_dict = [dict() for x in range(int(r))]
current_net_cost_matrix={}
temp_list=[]
trial_list=[]
another_costs=[]
for i in range(int(r)):
    temp_list=input().split(' ')
    k=0
    for j in range(int(c)):
        trial_dict[i][col_ind[j]]=int(temp_list[j])
        trial_list.append(int(temp_list[j]))
    current_net_cost_matrix[row_ind[i]]=trial_dict[i]
    another_costs.append(trial_list)
    trial_list=[]
    temp_list.clear()
#print(another_costs)   
#current_net_cost_matrix  = {'W': {'A': 16, 'B': 16, 'C': 13, 'D': 22, 'E': 17},
#          'X': {'A': 14, 'B': 14, 'C': 13, 'D': 19, 'E': 15},
#          'Y': {'A': 19, 'B': 19, 'C': 20, 'D': 23, 'E': 50},
#          'Z': {'A': 50, 'B': 12, 'C': 50, 'D': 15, 'E': 11}}

# current_net_cost_matrix  = {'W': {'A': 3, 'B': 1, 'C': 7, 'D': 4},
#           'X': {'A': 2, 'B': 6, 'C': 5, 'D': 9},
#           'Y': {'A': 8, 'B': 3, 'C': 3, 'D': 2},
#           }


#current_supply_tp = {'W': 50, 'X': 60, 'Y': 50, 'Z': 50}
#current_supply_tp = {'O1': 300, 'O2': 400, 'O3': 500}
current_supply_tp={}
another_supply=[]
temp_list=input().split(' ')
for i in range(int(r)):
    current_supply_tp[row_ind[i]]=int(temp_list[i])
    another_supply.append(int(temp_list[i]))
temp_list.clear()


#current_demand_tp = {'A': 30, 'B': 20, 'C': 70, 'D': 30, 'E': 60}
#current_demand_tp = {'A': 250, 'B': 350, 'C': 400, 'D': 200}

current_demand_tp={}
another_demand=[]
temp_list=input().split(' ')
i=0
for ind1 in range(int(c)):
    current_demand_tp[col_ind[ind1]]=int(temp_list[ind1])
    another_demand.append(int(temp_list[ind1]))
temp_list.clear()
arranged_column = sorted(current_demand_tp.keys())

new_dict_modify = dict((val, defaultdict(int)) for val in current_net_cost_matrix)
matrix_use_to_modify = {}
for ind1 in current_supply_tp:
    matrix_use_to_modify[ind1] = sorted(current_net_cost_matrix[ind1].keys(), key=lambda matrix_use_to_modify: current_net_cost_matrix[ind1][matrix_use_to_modify])
for ind1 in current_demand_tp:
    matrix_use_to_modify[ind1] = sorted(current_net_cost_matrix.keys(), key=lambda matrix_use_to_modify: current_net_cost_matrix[matrix_use_to_modify][ind1])
 
while matrix_use_to_modify:
    dict_use_1 = {}
    for x in current_demand_tp:
        dict_use_1[x] = (current_net_cost_matrix[matrix_use_to_modify[x][1]][x] - current_net_cost_matrix[matrix_use_to_modify[x][0]][x]) if len(matrix_use_to_modify[x]) > 1 else current_net_cost_matrix[matrix_use_to_modify[x][0]][x]
    dict_use_2 = {}
    for x in current_supply_tp:
        dict_use_2[x] = (current_net_cost_matrix[x][matrix_use_to_modify[x][1]] - current_net_cost_matrix[x][matrix_use_to_modify[x][0]]) if len(matrix_use_to_modify[x]) > 1 else current_net_cost_matrix[x][matrix_use_to_modify[x][0]]
    max_value = max(dict_use_1, key=lambda val1: dict_use_1[val1])
    max_value_2 = max(dict_use_2, key=lambda val1: dict_use_2[val1])
    max_value_2, max_value = (max_value, matrix_use_to_modify[max_value][0]) if dict_use_1[max_value] > dict_use_2[max_value_2] else (matrix_use_to_modify[max_value_2][0], max_value_2)
    min_val_1 = min(current_supply_tp[max_value], current_demand_tp[max_value_2])
    new_dict_modify[max_value][max_value_2] += min_val_1
    current_demand_tp[max_value_2] -= min_val_1
    if current_demand_tp[max_value_2] == 0:
        for ind1, ind2 in current_supply_tp.items():
            if ind2 != 0:
                matrix_use_to_modify[ind1].remove(max_value_2)
        del matrix_use_to_modify[max_value_2]
        del current_demand_tp[max_value_2]
    current_supply_tp[max_value] -= min_val_1
    if current_supply_tp[max_value] == 0:
        for ind1, ind2 in current_demand_tp.items():
            if ind2 != 0:
                matrix_use_to_modify[ind1].remove(max_value)
        del matrix_use_to_modify[max_value]
        del current_supply_tp[max_value]
basis_check=[]
cost_final_matrix=[]
present_in_basis = set()
not_present_in_basis=set()

print("\n")
print("Initial bfs:")
cost = 0
for index_1 in sorted(current_net_cost_matrix):
    trial=[]
    temp_matrix=[]
    for n in arranged_column:
        trial=[]
        val_check = new_dict_modify[index_1][n]
        temp_matrix.append(int(val_check))
        if val_check != 0:
            trial.append(int(index_1[-1])-1)
            trial.append(int(n[-1])-1)
            present_in_basis.add((int(index_1[-1])-1, int(n[-1])-1))
            basis_check.append(trial)
            print ("x"+str(int(index_1[-1]))+str(int(n[-1]))+"="+str(val_check)+",",end=" ")
        else:
            not_present_in_basis.add((int(index_1[-1])-1, int(n[-1])-1))
        cost += val_check * current_net_cost_matrix[index_1][n]
    cost_final_matrix.append(temp_matrix)

print ("\nCost = ", cost)

if len(present_in_basis)!=len(another_supply)+len(another_demand)-1:
    temp_val=abs(len(present_in_basis)-(len(another_supply)+len(another_demand)-1))
    while temp_val>0:
       to_be_added=min(not_present_in_basis)
       present_in_basis.add(to_be_added)
       not_present_in_basis.remove(to_be_added)
       temp_val=temp_val-1
       
#print(len(present_in_basis))       
#print(basis_check)
# print("Final Matrix is :")
# print(cost_final_matrix)
# print(basis)
# print(not_present_in_basis)


print("Optimal Solution:",end=" ") 
final_matrix = answer_matrix(another_supply, another_demand, another_costs,cost_final_matrix,present_in_basis,not_present_in_basis)
final_ans=0;
for i in range(len(final_matrix)):
    for j in range(len(final_matrix[i])):
        if(final_matrix[i][j]!=0):
            print("x"+str(i+1)+str(j+1)+"="+str(final_matrix[i][j])+",",end=" ")
        final_ans+=final_matrix[i][j]*another_costs[i][j]
print("\n")
print("Optimal Cost: "+ str(final_ans))




   