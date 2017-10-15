import os
import re
import numpy as np
import matplotlib.pyplot as plt



## Consts
log_color_reg = {}
log_color_reg[0] = 'g'
log_color_reg[1] = 'b'
log_color_reg[2] = 'r'
log_color_reg[3] = 'y'
log_color_reg[4] = 'm'
log_color_reg[5] = 'c'
log_color_reg[6] = 'k'

log_linestyle_reg = {}
log_linestyle_reg[0] = '-'
log_linestyle_reg[1] = '-'
log_linestyle_reg[2] = '--'
log_linestyle_reg[3] = ':'
log_linestyle_reg[4] = ':'
log_linestyle_reg[5] = '-'
log_linestyle_reg[6] = '-'

log_offset = {}
log_offset[0] = 0
log_offset[1] = 0
log_offset[2] = 0
log_offset[3] = 0
log_offset[4] = 0
log_offset[5] = 0
log_offset[6] = 0

# log_color_reg = {}
# log_color_reg[0] = 'g'
log_color_reg[1] = 'r'
log_color_reg[2] = 'r'
log_color_reg[3] = 'r'
log_color_reg[4] = 'r'
# log_color_reg[5] = 'r'
# log_color_reg[6] = 'k'
#
#
# log_linestyle_reg = {}
# log_linestyle_reg[0] = '-'
# log_linestyle_reg[1] = '-'
# log_linestyle_reg[2] = '-'
# log_linestyle_reg[3] = '--'
# log_linestyle_reg[4] = '--'
# log_linestyle_reg[5] = '--'
# log_linestyle_reg[6] = '-'
# log_linestyle_reg[7] = '--'
# log_linestyle_reg[8] = '--'
# log_linestyle_reg[9] = '--'
# log_linestyle_reg[10] = '-'
# log_linestyle_reg[11] = '-'
# log_linestyle_reg[12] = '-'
# log_linestyle_reg[13] = '-'



log_color_smooth = {}
log_color_smooth[0] = 'g'
log_color_smooth[1] = 'b'
log_color_smooth[2] = 'r'
log_color_smooth[3] = 'y'
log_color_smooth[4] = 'm'
log_color_smooth[5] = 'c'
log_color_smooth[6] = 'k'

DEBUG = True
## Help functions


def get_smooth_res(res,smooth_factor=5):
    # smooth reward
    reward_smooth = []
    # for ii in range(0,smooth_factor-1):
    #     reward_smooth.append(float(res[ii]))
    # for ii in range(smooth_factor-1,res.__len__()-smooth_factor):
    #     cur_sm = 0
    #     for jj in range(ii-smooth_factor,ii+smooth_factor+1):
    #         cur_sm = cur_sm + float(res[jj])
    #     cur_sm = float(cur_sm) / float((2*smooth_factor)+1)
    #     reward_smooth.append(cur_sm)
    # for ii in range(res.__len__()-smooth_factor,res.__len__()):
    #     reward_smooth.append(float(res[ii]))
    for ii in range(0,res.__len__()):
        cur_sm = 0
        num_of_el = 0
        for jj in range(max(0,ii-smooth_factor),min(res.__len__(),ii+smooth_factor+1)):
            cur_sm = cur_sm + float(res[jj])
            num_of_el += 1
        cur_sm = float(cur_sm) / float(num_of_el)
        reward_smooth.append(cur_sm)

    assert reward_smooth.__len__() == res.__len__()
    return reward_smooth

def add_single_line_values_to_array(arr,arr_idx,line_values):
    if line_values:
      arr[arr_idx].append(float(line_values[0]))
    return True

def find_and_add_iterative_new(arr,stat_to_find,num_of_iter,line_to_search):
    for itr in range(0,num_of_iter):
        line_values = re.findall(r'' + re.escape(str(stat_to_find.replace('***',str(itr+1))))  + '(\d+\.*\d*)',line_to_search)
        add_single_line_values_to_array(arr,itr,line_values)

def find_and_add_iterative(arr,stat_to_find,num_of_iter,line_to_search):
    for itr in range(0,num_of_iter):
        line_values = re.findall(r'' + re.escape(str(stat_to_find)) + '\[' + re.escape(str((itr+1))) + '\]=(\d+)',line_to_search)
        add_single_line_values_to_array(arr,itr,line_values)

def list_div_element_by_element(arr1,arr2,res_lst):
    if arr1.__len__() == 0 or arr2.__len__() == 0:
        return True
    assert(arr1.__len__() == arr2.__len__())
    #assert(arr1.__len__() == res_lst.__len__())
    # if arr1.__len__() == 0:
    #     return True

    if type(arr1[0]) == list:
        ## list of lists, do recursive call
        for idx in range(0,arr1.__len__()):
            list_div_element_by_element(arr1[idx],arr2[idx],res_lst[idx])
        return True

    ## Divide
    for idx in range(0,arr1.__len__()):
        if int(arr2[idx]) != int(0):
            res_lst.append(float(arr1[idx])/float(arr2[idx]))
        else:
            res_lst.append(float(0))


def create_multidim_lists_arr(dim1,dim2=0,dim3=0):
    dim1 = dim1
    dim2 = dim2
    dim3 = dim3
    arr = []
    for d1 in range(0,dim1):
        arr.append([])
        for d2 in range(0,dim2):
            arr[d1].append([])
            for d3 in range(0,dim3):
                arr[d1][d2].append([])
    return arr

fig = {}
ax = {}
fig_id = 0



def show_lvls_compare_graph(show_graph,values,x_values,title,ylbl,xlbl,num_of_levels_to_present,maxy=None,show_smooth=True,show_reg=False,line_width=2,general_fontsize=20):
    x_title_fontsize     = general_fontsize
    y_title_fontsize     = general_fontsize
    vals_fontsize        = general_fontsize
    title_fontsize       = general_fontsize
    if not show_graph or not values[0]:
        return True
    global fig_id
    fig[fig_id], ax[fig_id] = plt.subplots(num_of_levels_to_present,sharex=True)

    # Find maximum value of all logs
    if not maxy:
        maxy = float(0)
        for lvl_i in range(0,num_of_levels_to_present):
            for lg in range(0,num_of_logs):
                if (values[lg][lvl_i].__len__() > 0):
                    maxy = int(max(maxy,int(max(map(float,values[lg][lvl_i])))))
    if show_smooth:
        ylbl = ylbl + ' (Smooth)'
    for lvl_i in range(0,num_of_levels_to_present):
        lvl_idx = lvl_i + 1
        ax_idx = (num_of_levels_to_present-1)-lvl_i
        for lg in range(0,num_of_logs):
            if not values[lg][lvl_i]:
                continue
            if values[lg][lvl_i].__len__() > 0:
                if not x_values:
                    current_x_values = range(1+log_offset[lg],(values[lg][lvl_i].__len__()+1+log_offset[lg]))
                else:
                    current_x_values = x_values[lg][lvl_i]
                if show_smooth:
                    current_y_values = get_smooth_res(values[lg][lvl_i])
                    current_color    = log_color_smooth[lg]
                else:
                    current_y_values = values[lg][lvl_i]
                    current_color    = log_color_reg[lg]
                current_linestyle= log_linestyle_reg[lg]

                if num_of_levels_to_present == 1:
                    ax[fig_id].plot(current_x_values, current_y_values,current_color,label=lognames[lg],linestyle=current_linestyle,linewidth=line_width)
                else:
                    ax[fig_id][ax_idx].plot(current_x_values, current_y_values,current_color,label=lognames[lg],linestyle=current_linestyle,linewidth=line_width)
        if num_of_levels_to_present == 1:
            ax[fig_id].set_ylim([0,maxy])
            ax[fig_id].grid()
            #plt.xlabel('Index of evaluation')
            ax[fig_id].set_ylabel(ylbl, fontsize=y_title_fontsize)
            ax[fig_id].set_xlabel(xlbl, fontsize=x_title_fontsize)
            title_str = title
            ax[fig_id].set_title('%s'%(title_str), fontsize=title_fontsize)
            ax[fig_id].tick_params(axis='both', labelsize=vals_fontsize)
            #ax.subplots_adjust(hspace=.5)
            leg = ax[fig_id].legend(loc=1)
            leg.get_frame().set_alpha(0.4)
        else:
            ax[fig_id][ax_idx].set_ylim([0,maxy])
            ax[fig_id][ax_idx].grid()
            #plt.xlabel('Index of evaluation')
            ax[fig_id][ax_idx].set_ylabel(ylbl, fontsize=y_title_fontsize)
            if lvl_idx == 1:
                ax[fig_id][ax_idx].set_xlabel(xlbl, fontsize=x_title_fontsize)
            title_str = title + str(lvl_idx)
            ax[fig_id][ax_idx].set_title('%s'%(title_str), fontsize=title_fontsize)
            ax[fig_id][ax_idx].tick_params(axis='both', labelsize=vals_fontsize)
            #ax.subplots_adjust(hspace=.5)
            leg = ax[fig_id][ax_idx].legend(loc=1)
            leg.get_frame().set_alpha(0.4)
    fig_id = fig_id + 1







## Load logs
logs = []
logslist = open("logslist.txt", "r+")
lognames = []
for line in logslist:
    line=line.rstrip()
    if line != "":
        line_arr = line.split(",")
        fname = line_arr[0]
        if line_arr.__len__() > 1:
            logname = line_arr[1]
        else:
            logname = line_arr[0]
        print("trying to open: ",fname)
        lognames.append(logname)
        logobj = open(fname, "r+")
        logs.append(logobj)

print("Logs opened")

num_of_lvls_for_stats               = 7
num_of_logs                         = logs.__len__()

reward_arr                          = create_multidim_lists_arr(num_of_logs,1)
variance_arr                        = create_multidim_lists_arr(num_of_logs,1)
numep_arr                           = create_multidim_lists_arr(num_of_logs,1)
frames_in_evaluation_arr            = create_multidim_lists_arr(num_of_logs,1)
average_frames_per_game_arr         = create_multidim_lists_arr(num_of_logs,1)
ttl_lives_in_lvl5_start_arr         = create_multidim_lists_arr(num_of_logs,1)
ttl_visits_in_lvl5_arr              = create_multidim_lists_arr(num_of_logs,1)
ttl_visits_in_lvl6_arr              = create_multidim_lists_arr(num_of_logs,1)
percent_reached_lvl5_arr            = create_multidim_lists_arr(num_of_logs,1)
percent_reached_lvl6_arr            = create_multidim_lists_arr(num_of_logs,1)
avg_lives_in_lvl5_start_arr         = create_multidim_lists_arr(num_of_logs,1)
avg_reward_per_terminal_arr         = create_multidim_lists_arr(num_of_logs,4)
terminals_per_lvl_arr               = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
acc_per_agent_arr                   = create_multidim_lists_arr(num_of_logs,6)
evl_steps_per_agent_arr             = create_multidim_lists_arr(num_of_logs,6)
visits_per_lvl_arr                  = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
total_games_per_lvl_arr             = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
pass_percent_per_lvl_arr            = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
death_rartio_per_lvl_arr            = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
gameovers_per_level                 = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
reward_per_level                    = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
avg_reward_per_level                = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
ttl_learning_steps_in_level_arr     = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)

ttl_steps_in_level_old_arr          = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
eval_steps_in_level_old_arr         = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
eval_die_percent_in_level_old_arr   = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)
eval_pass_percent_in_level_old_arr  = create_multidim_lists_arr(num_of_logs,num_of_lvls_for_stats)

log_idx = 0
for log in logs:
    if DEBUG: print("HERE1")

    for line in log:
        ## Find reward value
        find_and_add_iterative_new(reward_arr[log_idx],", reward: ",1,line)
        find_and_add_iterative_new(numep_arr[log_idx]," num. ep.: ",1,line)

        ## FIXME remove start
        if numep_arr[log_idx][0].__len__() > 0:
            numep_arr[log_idx][0][-1] = numep_arr[log_idx][0][-1] + 1
        if re.findall(r' num. ep.: (\d+),',line):
            frames_in_evaluation_arr[log_idx][0].append(125000)

        ## FIXME remove end

        ## Find std. error
        distance_from_mean_sum = 0
        num_of_items = 0
        variance = 0
        for gg in range(1,300):
            current_game_res = re.findall(r' stats_evl_reward_per_game\[' + str(gg) + '\]=(\d+),',line)
            if (not current_game_res) or int(current_game_res) == 0: break
            distance_from_mean_sum += (current_game_res[0] - reward_arr[log_idx][-1])**2
            num_of_items += 1
        if num_of_items > 0:
            variance = distance_from_mean_sum/num_of_items
        variance_arr[log_idx].append(variance)

        # line_ttl_5_lives_values = re.findall(r' stats_evl_total_lives_in_level_start\[5\]=(\d+) ',line)
        # add_single_line_values_to_array(ttl_lives_in_lvl5_start_arr[log_idx],0,line_ttl_5_lives_values)
        find_and_add_iterative_new(ttl_lives_in_lvl5_start_arr[log_idx]," stats_evl_total_lives_in_level_start[5]=",1,line)

        # line_ttl_5_lives_values = re.findall(r' stats_evl_visits_in_level\[5\]=(\d+) ',line)
        # add_single_line_values_to_array(ttl_visits_in_lvl5_arr[log_idx],0,line_ttl_5_lives_values)
        find_and_add_iterative_new(ttl_visits_in_lvl5_arr[log_idx]," stats_evl_visits_in_level[5]=",1,line)
        find_and_add_iterative_new(ttl_visits_in_lvl6_arr[log_idx]," stats_evl_visits_in_level[6]=",1,line)


        find_and_add_iterative_new(acc_per_agent_arr[log_idx],"stats_evl_clustering_acc_per_agent[***]=",6,line)
        find_and_add_iterative_new(evl_steps_per_agent_arr[log_idx],"stats_evl_steps_per_agent[***]=",6,line)


        ## Find terminal per level value
        find_and_add_iterative_new(terminals_per_lvl_arr[log_idx],"stats_evl_terminals_in_level[***]=",num_of_lvls_for_stats,line)
        find_and_add_iterative_new(visits_per_lvl_arr[log_idx],"stats_evl_visits_in_level[***]=",num_of_lvls_for_stats,line)
        find_and_add_iterative_new(gameovers_per_level[log_idx],"stats_evl_gameovers_in_level[***]=",num_of_lvls_for_stats,line)
        find_and_add_iterative_new(reward_per_level[log_idx],"stats_evl_reward_in_level[***]=",num_of_lvls_for_stats,line)

        find_and_add_iterative_new(ttl_learning_steps_in_level_arr[log_idx],"stats_agent_learning_steps_level[5][***]=",num_of_lvls_for_stats,line)

        find_and_add_iterative_new(avg_reward_per_terminal_arr[log_idx],"stats_evl_avg_reward_per_terminal[***]=",4,line)


        # find_and_add_iterative(terminals_per_lvl_arr[log_idx],"stats_evl_terminals_in_level",num_of_lvls_for_stats,line)
        # find_and_add_iterative(visits_per_lvl_arr[log_idx],"stats_evl_visits_in_level",num_of_lvls_for_stats,line)
        # find_and_add_iterative(gameovers_per_level[log_idx],"stats_evl_gameovers_in_level",num_of_lvls_for_stats,line)
        # find_and_add_iterative(reward_per_level[log_idx],"stats_evl_reward_in_level",num_of_lvls_for_stats,line)

        find_and_add_iterative_new(ttl_steps_in_level_old_arr[log_idx],"steps_in_level_total_counter[***]: ",num_of_lvls_for_stats,line)
        find_and_add_iterative_new(eval_steps_in_level_old_arr[log_idx],"steps_in_level_counter[***]: ",num_of_lvls_for_stats,line)
        find_and_add_iterative_new(eval_die_percent_in_level_old_arr[log_idx],"died_at_level_***_percent_relative: ",num_of_lvls_for_stats,line)



        ## Create number of episodes array
        for itr in range(0,num_of_lvls_for_stats):
            line_values = re.findall(r'num. ep.: (\d+)',line)
            add_single_line_values_to_array(total_games_per_lvl_arr[log_idx],itr,line_values)

    log_idx += 1
if DEBUG: print("HERE2")
list_div_element_by_element(visits_per_lvl_arr,total_games_per_lvl_arr,pass_percent_per_lvl_arr)
for lg in range(0,num_of_logs):
    pass_percent_per_lvl_arr[lg]=pass_percent_per_lvl_arr[lg][1:]

list_div_element_by_element(terminals_per_lvl_arr,visits_per_lvl_arr,death_rartio_per_lvl_arr)
list_div_element_by_element(reward_per_level,visits_per_lvl_arr,avg_reward_per_level)
list_div_element_by_element(frames_in_evaluation_arr,numep_arr,average_frames_per_game_arr)
list_div_element_by_element(ttl_lives_in_lvl5_start_arr,numep_arr,avg_lives_in_lvl5_start_arr)
list_div_element_by_element(ttl_visits_in_lvl5_arr,numep_arr,percent_reached_lvl5_arr)
list_div_element_by_element(ttl_visits_in_lvl6_arr,numep_arr,percent_reached_lvl6_arr)

if eval_die_percent_in_level_old_arr:
    for lg in range(0,eval_die_percent_in_level_old_arr.__len__()):
        if eval_die_percent_in_level_old_arr[lg]:
            for lvl in range(0,eval_die_percent_in_level_old_arr[lg].__len__()):
                for eval_idx in range(0,eval_die_percent_in_level_old_arr[lg][lvl].__len__()):
                    if eval_steps_in_level_old_arr[lg][lvl].__len__() > 1:
                        if eval_steps_in_level_old_arr[lg][lvl].__len__() <= eval_idx:
                            steps_of_eval = 0
                        else:
                            steps_of_eval = eval_steps_in_level_old_arr[lg][lvl][eval_idx]
                        eval_pass_percent_in_level_old_arr[lg][lvl].append(0)
                        eval_pass_percent_in_level_old_arr[lg][lvl][-1] = 0 if int(steps_of_eval) < 1 else (1-eval_die_percent_in_level_old_arr[lg][lvl][eval_idx])
                    else:
                        eval_pass_percent_in_level_old_arr[lg][lvl].append(0)
                        eval_pass_percent_in_level_old_arr[lg][lvl][-1] = 0 if int(eval_die_percent_in_level_old_arr[lg][lvl][eval_idx]) <= 0 else (1-eval_die_percent_in_level_old_arr[lg][lvl][eval_idx])




# if DEBUG: print(average_frames_per_game_arr[0])
# if DEBUG: print(average_frames_per_game_arr[1])
# if DEBUG: print(frames_in_evaluation_arr[0])
# if DEBUG: print(frames_in_evaluation_arr[1])
# if DEBUG: print(numep_arr[0])
# if DEBUG: print(numep_arr[1])
#
# if DEBUG: print(reward_arr[0])
# if DEBUG: print(reward_arr[1])
#show_single_graph(True,reward_arr,'Total reward','Reward',7000,False,True)
#show_lvls_compare_graph(True,pass_percent_per_lvl_arr,'Pass percent as function of evaluation','Percent',5,1,False,True)
#show_lvls_compare_graph(True,reward_arr,'Death ratio as function of evaluation - level ','Ratio',,4.5,False,True)
show_lvls_compare_graph(True,reward_arr,None,'Average reward as function of evaluation ','Avg. Reward','Evaluation num.',1,22000,False,True)
show_lvls_compare_graph(True,acc_per_agent_arr,None,'Acc per agent ','Acc','Evaluation num.',6,1.5,False,True)
show_lvls_compare_graph(True,evl_steps_per_agent_arr,None,'Evaluation steps per agent ','Steps','Evaluation num.',6,85000,False,True)


show_lvls_compare_graph(False,reward_arr,None,'Average reward as function of evaluation ','Avg. Reward','Evaluation num.',1,22000,True,False)
show_lvls_compare_graph(False,average_frames_per_game_arr,None,'Average frames per game ','Avg. num. of frames','Evaluation num.',1,10000,False,True)
show_lvls_compare_graph(False,avg_lives_in_lvl5_start_arr,None,'Average lives in when reaching level 5 ','Avg. num. of lives','Evaluation num.',1,4.5,False,True)
show_lvls_compare_graph(False,percent_reached_lvl5_arr,None,'Percent of games reached level 5 ','Percent','Evaluation num.',1,1.5,False,True)
show_lvls_compare_graph(False,percent_reached_lvl6_arr,None,'Percent of games reached level 6 ','Percent','Evaluation num.',1,1.5,False,True)


show_lvls_compare_graph(False,variance_arr,None,'Variance between games as function of evaluation ','Var','Evaluation num.',1,1000,False,True)



show_lvls_compare_graph(False,eval_pass_percent_in_level_old_arr,ttl_steps_in_level_old_arr,'Pass percent as function of total steps in level','Percent','Steps in level',5,1,False,True,2)
show_lvls_compare_graph(False,eval_pass_percent_in_level_old_arr,ttl_steps_in_level_old_arr,'Pass percent as function of total steps in level','Percent','Steps in level',5,1,True,False,2)

show_lvls_compare_graph(False,pass_percent_per_lvl_arr,ttl_learning_steps_in_level_arr,'Pass percent as function of total steps in level','Percent','Steps in level',5,1,False,True,2)
show_lvls_compare_graph(False,pass_percent_per_lvl_arr,ttl_learning_steps_in_level_arr,'Pass percent as function of total steps in level','Percent','Steps in level',5,1,True,False,2)


show_lvls_compare_graph(False,eval_pass_percent_in_level_old_arr,None,'Pass percent as function of evaluation ','Percent','Evaluation index',3,1,False,True,2)
show_lvls_compare_graph(False,eval_pass_percent_in_level_old_arr,None,'Pass percent as function of evaluation ','Percent','Evaluation index',3,1,True,False,2)

show_lvls_compare_graph(False,numep_arr,None,'Number of episodes as function of evaluation ','num ep','Evaluation num.',1,150,False,True)
show_lvls_compare_graph(False,death_rartio_per_lvl_arr,None,'Death ratio as function of evaluation - level ','Ratio','Evaluation num.',5,4.5,False,True)
show_lvls_compare_graph(False,avg_reward_per_level,None,'Average reward in level as function of evaluation - level ','Average reward','Evaluation num.',5,None,False,True)
plt.show()


#
#
# stats_evl_terminals_in_level
# stats_evl_terminals_in_level[4]=3
# Steps: 250000 (frames: 1000000), reward: 1471.26, epsilon: 0.05, lr: 0.00025, training time: 5219s, training rate: 191fps, testing time: 3292s, testing rate: 151fps,  num. ep.: 234,  num. rewards: 6542 , cur_agent: cur_ag:  , cur_ag[1]=1 ,  , cur_ag[2]=2 ,  , cur_ag[3]=3 ,  , cur_ag[4]=4 ,  , cur_ag[5]=5 ,  , cur_ag[6]=6 ,  , cur_ag[7]=7 ,  , cur_ag[8]=1 ,  , cur_ag[9]=1 ,  , cur_ag[10]=1 ,  , cur_ep: cur_ep:  , cur_ep[1]=0.1 ,  , cur_ep[2]=0.1 ,  , cur_ep[3]=0.1 ,  , cur_ep[4]=0.1 ,  , cur_ep[5]=1 ,  , cur_ep[6]=1 ,  , cur_ep[7]=1 ,  , cur_ep[8]=0.1 ,  , cur_ep[9]=0.1 ,  , cur_ep[10]=0.1 ,  , cur_steps: cur_steps:  , cur_steps[1]=328436 ,  , cur_steps[2]=41788 ,  , cur_steps[3]=3708 ,  , cur_steps[4]=1068 ,  , cur_steps[5]=0 ,  , cur_steps[6]=0 ,  , cur_steps[7]=0 ,  , cur_steps[8]=328436 ,  , cur_steps[9]=328436 ,  , cur_steps[10]=328436 ,  , stats_evl_visits_in_level: stats_evl_visits_in_level:  , stats_evl_visits_in_level[1]=235 ,  , stats_evl_visits_in_level[2]=57 ,  , stats_evl_visits_in_level[3]=8 ,  , stats_evl_visits_in_level[4]=2 ,  , stats_evl_visits_in_level[5]=0 ,  , stats_evl_visits_in_level[6]=0 ,  , stats_evl_visits_in_level[7]=0 ,  , stats_evl_visits_in_level[8]=0 ,  , stats_evl_visits_in_level[9]=0 ,  , stats_evl_visits_in_level[10]=0 ,  , stats_evl_steps_in_level: stats_evl_steps_in_level:  , stats_evl_steps_in_level[1]=110877 ,  , stats_evl_steps_in_level[2]=12762 ,  , stats_evl_steps_in_level[3]=1328 ,  , stats_evl_steps_in_level[4]=268 ,  , stats_evl_steps_in_level[5]=0 ,  , stats_evl_steps_in_level[6]=0 ,  , stats_evl_steps_in_level[7]=0 ,  , stats_evl_steps_in_level[8]=0 ,  , stats_evl_steps_in_level[9]=0 ,  , stats_evl_steps_in_level[10]=0 ,  , stats_evl_gameovers_in_level: stats_evl_gameovers_in_level:  , stats_evl_gameovers_in_level[1]=178 ,  , stats_evl_gameovers_in_level[2]=49 ,  , stats_evl_gameovers_in_level[3]=6 ,  , stats_evl_gameovers_in_level[4]=2 ,  , stats_evl_gameovers_in_level[5]=0 ,  , stats_evl_gameovers_in_level[6]=0 ,  , stats_evl_gameovers_in_level[7]=0 ,  , stats_evl_gameovers_in_level[8]=0 ,  , stats_evl_gameovers_in_level[9]=0 ,  , stats_evl_gameovers_in_level[10]=0 ,  , stats_evl_terminals_in_level: stats_evl_terminals_in_level:  , stats_evl_terminals_in_level[1]=800 ,  , stats_evl_terminals_in_level[2]=100 ,  , stats_evl_terminals_in_level[3]=9 ,  , stats_evl_terminals_in_level[4]=3 ,  , stats_evl_terminals_in_level[5]=0 ,  , stats_evl_terminals_in_level[6]=0 ,  , stats_evl_terminals_in_level[7]=0 ,  , stats_evl_terminals_in_level[8]=0 ,  , stats_evl_terminals_in_level[9]=0 ,  , stats_evl_terminals_in_level[10]=0 ,  , stats_ttl_total_steps_in_level: stats_ttl_total_steps_in_level:  , stats_ttl_total_steps_in_level[1]=328615 ,  , stats_ttl_total_steps_in_level[2]=41837 ,  , stats_ttl_total_steps_in_level[3]=3714 ,  , stats_ttl_total_steps_in_level[4]=1070 ,  , stats_ttl_total_steps_in_level[5]=0 ,  , stats_ttl_total_steps_in_level[6]=0 ,  , stats_ttl_total_steps_in_level[7]=0 ,  , stats_ttl_total_steps_in_level[8]=0 ,  , stats_ttl_total_steps_in_level[9]=0 ,  , stats_ttl_total_steps_in_level[10]=0 ,  , stats_ttl_learning_steps_in_level: stats_ttl_learning_steps_in_level:  , stats_agent_learning_steps_level: stats_agent_learning_steps_level: stats_agent_learning_steps_level[1]:  , stats_agent_learning_steps_level[1][1]=328436 ,  , stats_agent_learning_steps_level[1][2]=0 ,  , stats_agent_learning_steps_level[1][3]=0 ,  , stats_agent_learning_steps_level[1][4]=0 ,  , stats_agent_learning_steps_level[1][5]=0 ,  , stats_agent_learning_steps_level[1][6]=0 ,  , stats_agent_learning_steps_level[1][7]=0 ,  , stats_agent_learning_steps_level[1][8]=0 ,  , stats_agent_learning_steps_level[1][9]=0 ,  , stats_agent_learning_steps_level[1][10]=0 , stats_agent_learning_steps_level[12]:  , stats_agent_learning_steps_level[12][1]=0 ,  , stats_agent_learning_steps_level[12][2]=41788 ,  , stats_agent_learning_steps_level[12][3]=0 ,  , stats_agent_learning_steps_level[12][4]=0 ,  , stats_agent_learning_steps_level[12][5]=0 ,  , stats_agent_learning_steps_level[12][6]=0 ,  , stats_agent_learning_steps_level[12][7]=0 ,  , stats_agent_learning_steps_level[12][8]=0 ,  , stats_agent_learning_steps_level[12][9]=0 ,  , stats_agent_learning_steps_level[12][10]=0 , stats_agent_learning_steps_level[12]:  , stats_agent_learning_steps_level[12][1]=0 ,  , stats_agent_learning_steps_level[12][2]=0 ,  , stats_agent_learning_steps_level[12][3]=3708 ,  , stats_agent_learning_steps_level[12][4]=0 ,  , stats_agent_learning_steps_level[12][5]=0 ,  , stats_agent_learning_steps_level[12][6]=0 ,  , stats_agent_learning_steps_level[12][7]=0 ,  , stats_agent_learning_steps_level[12][8]=0 ,  , stats_agent_learning_steps_level[12][9]=0 ,  , stats_agent_learning_steps_level[12][10]=0 , stats_agent_learning_steps_level[12]:  , stats_agent_learning_steps_level[12][1]=0 ,  , stats_agent_learning_steps_level[12][2]=0 ,  , stats_agent_learning_steps_level[12][3]=0 ,  , stats_agent_learning_steps_level[12][4]=1068 ,  , stats_agent_learning_steps_level[12][5]=0 ,  , stats_agent_learning_steps_level[12][6]=0 ,  , stats_agent_learning_steps_level[12][7]=0 ,  , stats_agent_learning_steps_level[12][8]=0 ,  , stats_agent_learning_steps_level[12][9]=0 ,  , stats_agent_learning_steps_level[12][10]=0 , stats_agent_learning_steps_level[12]:  , stats_agent_learning_steps_level[12][1]=0 ,  , stats_agent_learning_steps_level[12][2]=0 ,  , stats_agent_learning_steps_level[12][3]=0 ,  , stats_agent_learning_steps_level[12][4]=0 ,  , stats_agent_learning_steps_level[12][5]=0 ,  , stats_agent_learning_steps_level[12][6]=0 ,  , stats_agent_learning_steps_level[12][7]=0 ,  , stats_agent_learning_steps_level[12][8]=0 ,  , stats_agent_learning_steps_level[12][9]=0 ,  , stats_agent_learning_steps_level[12][10]=0 , stats_agent_learning_steps_level[12]:  , stats_agent_learning_steps_level[12][1]=0 ,  , stats_agent_learning_steps_level[12][2]=0 ,  , stats_agent_learning_steps_level[12][3]=0 ,  , stats_agent_learning_steps_level[12][4]=0 ,  , stats_agent_learning_steps_level[12][5]=0 ,  , stats_agent_learning_steps_level[12][6]=0 ,  , stats_agent_learning_steps_level[12][7]=0 ,  , stats_agent_learning_steps_level[12][8]=0 ,  , stats_agent_learning_steps_level[12][9]=0 ,  , stats_agent_learning_steps_level[12][10]=0 , stats_agent_learning_steps_level[12]:  , stats_agent_learning_steps_level[12][1]=0 ,  , stats_agent_learning_steps_level[12][2]=0 ,  , stats_agent_learning_steps_level[12][3]=0 ,  , stats_agent_learning_steps_level[12][4]=0 ,  , stats_agent_learning_steps_level[12][5]=0 ,  , stats_agent_learning_steps_level[12][6]=0 ,  , stats_agent_learning_steps_level[12][7]=0 ,  , stats_agent_learning_steps_level[12][8]=0 ,  , stats_agent_learning_steps_level[12][9]=0 ,  , stats_agent_learning_steps_level[12][10]=0 ,  ,
#
#   i=0
#   visit_idx=0
#   for line in log:
#     line_reward_values = re.findall(r', reward: (\d+).\d+,',line)
#
#
#     line_died_at_values = re.findall(r'died_at_level_1_percent_relative: (\d+.\d+), died_at_level_2_percent_relative: (\d+.\d+), died_at_level_3_percent_relative: (\d+.\d+), died_at_level_4_percent_relative: (\d+.\d+), died_at_level_5_percent_relative: (\d+.\d+),',line)
#     if line_died_at_values:
#       for lvl_i in range(0,5):
#         if (float(line_died_at_values[0][lvl_i]) == float(0.00)):
#           died_at_rel_value[log_idx][lvl_i].append(float(0.0))
#         else:
#           died_at_rel_value[log_idx][lvl_i].append(1-float(line_died_at_values[0][lvl_i]))
#
#     line_numep_values = re.findall(r'num. ep.: (\d+),',line)
#
#
#     line_diedpercenttotal_values = re.findall(r'died_at_level_1_percent_total: (\d+.\d+), died_at_level_2_percent_total: (\d+.\d+), died_at_level_3_percent_total: (\d+.\d+), died_at_level_4_percent_total: (\d+.\d+), died_at_level_5_percent_total: (\d+.\d+)',line)
#     add_lvl_line_values_to_array(died_at_total_value,log_idx,line_diedpercenttotal_values)
#
#     line_visited_values = re.findall(r'totalVisit1: (\d+), totalVisit2: (\d+), totalVisit3: (\d+), totalVisit4: (\d+), totalVisit5: (\d+), totalVisit6: (\d+), totalVisit7: (\d+),',line)
#     add_lvl_line_values_to_array(visited_at_value,log_idx,line_visited_values)
#
#     line_curep_values = re.findall(r'curEp1: (\d+.\d+), curEp2: (\d+.\d+), curEp3: (\d+.\d+), curEp4: (\d+.\d+), curEp5: (\d+.\d+), curEp6: (\d+.\d+), curEp7: (\d+.\d+),',line)
#     add_lvl_line_values_to_array(curep_value,log_idx,line_curep_values)
#
#     line_cursteps_values = re.findall(r'curSteps1: (\d+), curSteps2: (\d+), curSteps3: (\d+), curSteps4: (\d+), curSteps5: (\d+), curSteps6: (\d+), curSteps7: (\d+),',line)
#     add_lvl_line_values_to_array(cursteps_value,log_idx,line_cursteps_values)
#
#     line_steps_in_level_values = re.findall(r'steps_in_level_counter\[1\]: (\d+), steps_in_level_counter\[2\]: (\d+), steps_in_level_counter\[3\]: (\d+), steps_in_level_counter\[4\]: (\d+), steps_in_level_counter\[5\]: (\d+), steps_in_level_counter\[6\]: (\d+), steps_in_level_counter\[7\]: (\d+)',line)
#     add_lvl_line_values_to_array(stepsinlevel_value,log_idx,line_steps_in_level_values)
#
#     line_new_steps_in_level_values = re.findall(r'steps_in_level_total_counter\[1\]: (\d+), steps_in_level_total_counter\[2\]: (\d+), steps_in_level_total_counter\[3\]: (\d+), steps_in_level_total_counter\[4\]: (\d+), steps_in_level_total_counter\[5\]: (\d+), steps_in_level_total_counter\[6\]: (\d+), steps_in_level_total_counter\[7]\: (\d+),',line)
#     add_lvl_line_values_to_array(new_stepsinlevel_value,log_idx,line_new_steps_in_level_values)
#
#     line_new_terminals_in_level_values = re.findall(r'terminals_in_level_counter\[1\]: (\d+), terminals_in_level_counter\[2\]: (\d+), terminals_in_level_counter\[3\]: (\d+), terminals_in_level_counter\[4\]: (\d+), terminals_in_level_counter\[5\]: (\d+), terminals_in_level_counter\[6\]: (\d+), terminals_in_level_counter\[7]\: (\d+),',line)
#     add_lvl_line_values_to_array(new_terminalsinlevel_value,log_idx,line_new_terminals_in_level_values)
#
#     line_died_abs_values = re.findall(r'diedAt1Total: (\d+), diedAt2Total: (\d+), diedAt3Total: (\d+), diedAt4Total: (\d+), diedAt5Total: (\d+), diedAt6Total: (\d+), diedAt7Total: (\d+)',line)
#
#     #### Calculations
#     if line_died_at_values and line_numep_values and line_diedpercenttotal_values:
#       reward_decreasing = 0
#       #print "no worry man"
#       for lvl_i in range(1,4):
#         tmp_passrate = 0
#         if (float(line_died_at_values[0][lvl_i]) == float(0.00)):
#           reward_decreasing = reward_decreasing
#           # total_visit_calculated[log_idx][lvl_i].append(float(0.0))
#         else:
#           tmp_passrate = float(line_diedpercenttotal_values[0][lvl_i])/float(line_died_at_values[0][lvl_i])
#           reward_decreasing = reward_decreasing + (tmp_passrate*3100)
#       reward_value_calculated[log_idx].append(float(line_reward_values[0])-reward_decreasing)
#       #total_visit_calculated[log_idx][lvl_i].append((float(line_diedpercenttotal_values[0][lvl_i])*float(float(line_numep_values[0])))/float(float(line_died_at_values[0][lvl_i])))
#
#     ## CONT
#     if line_visited_values and line_new_terminals_in_level_values:
#       for lvl_i in range(0,5):
#           if float(line_visited_values[0][lvl_i]) == float(0):
#             terminal_visited_ratio[log_idx][lvl_i].append(float(4))
#           else:
#             terminal_visited_ratio[log_idx][lvl_i].append(float(line_new_terminals_in_level_values[0][lvl_i])/float(line_visited_values[0][lvl_i]))
#
#   log_idx = log_idx + 1
#
#
# ################################################################################################################################################################################################
# ################################################################################################################################################################################################
# ################################################################################################################################################################################################
# ################################################################################################################################################################################################
# ################################################################################################################################################################################################
#
#
#
# def add_lvl_line_values_to_array(arr,log_idx,line_values):
#     if line_values:
#       for lvl_i in range(0,5):
#         if line_values[0][lvl_i]:
#           arr[log_idx][lvl_i].append(line_values[0][lvl_i])
#     return True
#
#
#
#
#
# def get_smooth_res(res,smooth_factor=5):
#     # smooth reward
#     reward_smooth = []
#     for ii in range(0,smooth_factor-1):
#         reward_smooth.append(float(res[ii]))
#     for ii in range(smooth_factor-1,res.__len__()-smooth_factor):
#         cur_sm = 0
#         for jj in range(ii-smooth_factor,ii+smooth_factor+1):
#             cur_sm = cur_sm + float(res[jj])
#         cur_sm = float(cur_sm) / float((2*smooth_factor)+1)
#         reward_smooth.append(cur_sm)
#     for ii in range(res.__len__()-smooth_factor,res.__len__()):
#         reward_smooth.append(float(res[ii]))
#     assert reward_smooth.__len__() == res.__len__()
#     return reward_smooth
#
# def get_var_res(res,smooth_factor=5,high=False):
#     # smooth reward
#     reward_smooth = []
#     for ii in range(0,smooth_factor-1):
#         reward_smooth.append(float(res[ii]))
#     for ii in range(smooth_factor-1,res.__len__()-smooth_factor):
#         cur_sm = 0
#         for jj in range(ii-smooth_factor,ii+smooth_factor+1):
#             cur_sm = cur_sm + float(res[jj])
#         cur_sm = float(cur_sm) / float((2*smooth_factor)+1)
#         avg = cur_sm
#         n=0
#         var = 0
#         for jj in range(ii-smooth_factor,ii+smooth_factor+1):
#             var = var + ((float(res[jj]) - avg)**2)
#             n = n + 1
#         from math import sqrt
#         var = sqrt((1/float(n))*var)
#         if high:
#             reward_smooth.append(cur_sm+var)
#         else:
#             max(0,reward_smooth.append(cur_sm-var))
#     for ii in range(res.__len__()-smooth_factor,res.__len__()):
#         reward_smooth.append(float(res[ii]))
#     assert reward_smooth.__len__() == res.__len__()
#     return reward_smooth
#
# def success_by_chance(success_chance=0.5):
#     from random import randint
#     # chance is the probability of getting true, for example, success_by_chance(0.9) returns true in 90%
#     precision_max = 1000000
#     precision = 10
#     while ((success_chance*precision)!=int(success_chance*precision)) and precision < precision_max:
#         precision *= 10
#     if precision > precision_max:
#         raise NotImplementedError
#     return (randint(1,precision) <= int(success_chance*precision))
#
# def get_model(pass_percent,smooth=False):
#     # smooth reward
#     model_avg = []
#     model_var_high = []
#     model_var_low = []
#     print "hi! starting model"
#     pass_percent_new = []
#     pass_percent_new.append(0)
#     pass_percent_new.append(0)
#     pass_percent_new.append(0)
#     pass_percent_new.append(0)
#     pass_percent_new.append(0)
#
#     for j in range(pass_percent.__len__()):
#         if smooth:
#             pass_percent_new[j] = get_smooth_res(pass_percent[j])
#         else:
#             pass_percent_new[j] = pass_percent[j]
#     pass_percent = pass_percent_new
#     from math import sqrt
#     for ii in range(0,pass_percent[0].__len__()):
#         bonus_reward = 3100
#         bricks_reward = 525
#         pass_percent[0][ii] = float(pass_percent[0][ii])
#         if ii > 20 and pass_percent[0][ii] == 0:
#             pass_percent[0][ii] = 1
#         pass_percent[1][ii] = float(pass_percent[1][ii])
#         pass_percent[2][ii] = float(pass_percent[2][ii])
#         pass_percent[3][ii] = float(pass_percent[3][ii])
#         pass_percent[4][ii] = float(pass_percent[4][ii])
#         assert pass_percent[0][ii] <= 1
#         assert pass_percent[1][ii] <= 1
#         assert pass_percent[2][ii] <= 1
#         assert pass_percent[3][ii] <= 1
#         assert pass_percent[4][ii] <= 1
#
#         p = []
#         p.append(0)
#         p.append(0)
#         p.append(0)
#         p.append(0)
#         p.append(0)
#         p[0] = pass_percent[0][ii]*(1-pass_percent[1][ii])
#         p[1] = pass_percent[0][ii]*pass_percent[1][ii]*(1-pass_percent[2][ii])
#         p[2] = pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*(1-pass_percent[3][ii])
#         p[3] = pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*pass_percent[3][ii]*(1-pass_percent[4][ii])
#         p[4] = pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*pass_percent[3][ii]*pass_percent[4][ii]
#         reward_for_level = bonus_reward + bricks_reward
#
#         avg = (p[0]*(reward_for_level*1)) + \
#               (p[1]*(reward_for_level*2)) + \
#               (p[2]*(reward_for_level*3)) + \
#               (p[3]*(reward_for_level*4)) + \
#               (p[4]*(reward_for_level*5))
#         if avg == 0:
#             print p
#         ex2 = (p[0]*((reward_for_level*1)**2)) + \
#               (p[1]*((reward_for_level*2)**2)) + \
#               (p[2]*((reward_for_level*3)**2)) + \
#               (p[3]*((reward_for_level*4)**2)) + \
#               (p[4]*((reward_for_level*5)**2))
#
#         avg_old = ((pass_percent[0][ii]*(1-1))*(bonus_reward+bricks_reward)) + \
#               (pass_percent[0][ii]*pass_percent[1][ii]*((bonus_reward+bricks_reward)*2)) + \
#               (pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*((bonus_reward+bricks_reward)*3)) + \
#               (pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*pass_percent[3][ii]*((bonus_reward+bricks_reward)*4)) + \
#               (pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*pass_percent[3][ii]*pass_percent[4][ii]*((bonus_reward+bricks_reward)*5))
#         ex21 = (float(pass_percent[0][ii])*float(((bonus_reward+bricks_reward)**2)))
#         ex22 = (pass_percent[0][ii]*pass_percent[1][ii]*(((bonus_reward+bricks_reward)*2)**2))
#         ex23 = (pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*(((bonus_reward+bricks_reward)*3)**2))
#         ex24 = (pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*pass_percent[3][ii]*(((bonus_reward+bricks_reward)*4)**2))
#         ex25 = (pass_percent[0][ii]*pass_percent[1][ii]*pass_percent[2][ii]*pass_percent[3][ii]*pass_percent[4][ii]*(((bonus_reward+bricks_reward)*5)**2))
#         ex2_old = ex21 + ex22 + ex23 + ex24 + ex25
#         var = ex2 - (avg**2)
#         stderr = sqrt(var)
#         model_avg.append(avg)
#         model_var_high.append(max(0,avg+stderr))
#         model_var_low.append(max(0,avg-stderr))
#
#     print "bi! starting model"
#     return (model_avg,model_var_high,model_var_low)
#
#
#
# fig = {}
# ax = {}
# fig_id = 0
# def show_lvls_compare_graph(show_graph,values,title,ylbl,maxy=None,with_smooth=False,only_smooth=False):
#     if (not show_graph) or ((values[0]) and (not values[1])):
#       return True
#     fig[fig_id], ax[fig_id] = plt.subplots(5,sharex=True)
#     if not maxy:
#       maxy = float(0)
#       for lvl_i in range(0,5):
#         if (values[0][lvl_i].__len__() > 0):
#           maxy = int(max(maxy,int(max(map(float,values[0][lvl_i])))))
#         if (values[1][lvl_i].__len__() > 0):
#           maxy = int(max(maxy,int(max(map(float,values[1][lvl_i])))))
#
#     for lvl_i in range(0,5):
#       lvl_idx = lvl_i + 1
#       ax_idx = 4-lvl_i
#       if values[0][lvl_i].__len__() > 0:
#         if not only_smooth:
#             line1, = ax[fig_id][ax_idx].plot(range(1,(values[0][lvl_i].__len__()+1)), values[0][lvl_i],'g',label=lognames[0])
#         else:
#             line1 = 1
#         if with_smooth:
#             line1_smooth, = ax[fig_id][ax_idx].plot(range(1,(values[0][lvl_i].__len__()+1)), get_smooth_res(values[0][lvl_i]),'r',label=lognames[0]+' smooth')
#       else:
#         line1 = 1
#       if values[1][lvl_i].__len__() > 0:
#         if not only_smooth:
#             line2, = ax[fig_id][ax_idx].plot(range(1,(values[1][lvl_i].__len__()+1)), values[1][lvl_i],'b',label=lognames[1])
#         else:
#             line2 = 1
#         if with_smooth:
#             line2_smooth, = ax[fig_id][ax_idx].plot(range(1,(values[1][lvl_i].__len__()+1)), get_smooth_res(values[1][lvl_i]),'m',label=lognames[1]+' smooth')
#       else:
#         line2 = 1
#       ax[fig_id][ax_idx].set_ylim([0,maxy])
#       ax[fig_id][ax_idx].grid()
#       #plt.xlabel('Index of evaluation')
#       ax[fig_id][ax_idx].set_ylabel(ylbl)
#       title_str = title + str(lvl_idx)
#       ax[fig_id][ax_idx].set_title('%s'%(title_str))
#       #ax.subplots_adjust(hspace=.5)
#       leg = ax[fig_id][ax_idx].legend(loc=1)
#       leg.get_frame().set_alpha(0.4)
#
#       lines = [line1 or 1, line2 or 1]
#       for legline, origline in zip(leg.get_lines(), lines):
#           legline.set_picker(5)  # 5 pts tolerance
#           lined[legline] = origline
#
# reward_value = ([],[],[],[])
# reward_value_calculated = ([],[],[],[])
# died_at_rel_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# visited_at_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# curep_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# terminal_visited_ratio = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# died_at_total_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# total_visit_calculated = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# cursteps_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# stepsinlevel_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# new_stepsinlevel_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# new_terminalsinlevel_value = (([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]),([],[],[],[],[]))
# visited_at_idx = ([],[],[],[])
# died_at1_rel_value = ([],[],[],[])
# died_at2_rel_value = ([],[],[],[])
# died_at3_rel_value = ([],[],[],[])
# died_at4_rel_value = ([],[],[],[])
# died_at5_rel_value = ([],[],[],[])
# num_episode_value = ([],[],[],[])
# #Steps: 250000 (frames: 1000000), reward: 153.85, epsilon: 0.05, lr: 0.00025, training time: 4613s, training rate: 216fps, testing time: 1135s, testing rate: 440fps,  num. ep.: 221,  num. rewards: 849, max_level_reached: 1, average_level_per_terminal: 1.00, average_level_per_game: 1.00, si_do_color_transformation: 1
#
#
# print stepsinlevel_value
# print "visited_at_idx0"
# print visited_at_idx[0]
# print "visited_at_value0"
# print visited_at_value[0]
# print "visited_at_idx1"
# print visited_at_idx[1]
# print visited_at_idx[1]
# print "visited_at_value1"
# print visited_at_value[1]
# #print visited_at_value[1].size()
# #exit()
#
# def onpick(event):
#     # on the pick event, find the orig line corresponding to the
#     # legend proxy line, and toggle the visibility
#     legline = event.artist
#     origline = lined[legline]
#     vis = not origline.get_visible()
#     origline.set_visible(vis)
#     # Change the alpha on the line in the legend so we can see what lines
#     # have been toggled
#     if vis:
#         legline.set_alpha(1.0)
#     else:
#         legline.set_alpha(0.2)
#     fig.canvas.draw()
#
# def onpick3(event):
#     # on the pick event, find the orig line corresponding to the
#     # legend proxy line, and toggle the visibility
#     legline = event.artist
#     origline = lined[legline]
#     vis = not origline.get_visible()
#     origline.set_visible(vis)
#     # Change the alpha on the line in the legend so we can see what lines
#     # have been toggled
#     if vis:
#         legline.set_alpha(1.0)
#     else:
#         legline.set_alpha(0.2)
#     fig3.canvas.draw()
#
# def onpick4(event):
#     # on the pick event, find the orig line corresponding to the
#     # legend proxy line, and toggle the visibility
#     legline = event.artist
#     origline = lined[legline]
#     vis = not origline.get_visible()
#     origline.set_visible(vis)
#     # Change the alpha on the line in the legend so we can see what lines
#     # have been toggled
#     if vis:
#         legline.set_alpha(1.0)
#     else:
#         legline.set_alpha(0.2)
#     fig4.canvas.draw()
#
# def onpick5(event):
#     # on the pick event, find the orig line corresponding to the
#     # legend proxy line, and toggle the visibility
#     legline = event.artist
#     origline = lined[legline]
#     vis = not origline.get_visible()
#     origline.set_visible(vis)
#     # Change the alpha on the line in the legend so we can see what lines
#     # have been toggled
#     if vis:
#         legline.set_alpha(1.0)
#     else:
#         legline.set_alpha(0.2)
#     fig5.canvas.draw()
#
# def find_first_val(vals_list,max_val):
#     if (vals_list.__len__() > 0):
#         if (float(max(map(float,vals_list))) < max_val):
#             max_val = float(max(map(float,vals_list)))
#         return (i for i in range(0,vals_list.__len__()) if float(vals_list[i]) >= float(max_val)).next()
#     else:
#         return 0
#
#
#
#
#
# spacing=.5
# lined = dict()
#
#
# ###########
# ## CALC SOME MARKERS
# ###########
#
# #index_of_first_90_passrate = (i for i in range(0,a.__len__()) if a[i] > 21).next()
# #first_90_passrate= (([],[],[],[],[]),([],[],[],[],[]))
# #first_90_passrate_num_of_steps= (([],[],[],[],[]),([],[],[],[],[]))
#
# for lvl_i in range(0,5):
#     print "hello world"
#     #indx_of_dirty_lvl_90passrate = find_first_val(died_at_rel_value[1][lvl_i],0.9)
#     #val_of_dirty_lvl_90passrate = died_at_rel_value[1][lvl_i][indx_of_dirty_lvl_90passrate]
#     #steps_of_dirty_lvl_90passrate = cursteps_value[1][lvl_i][indx_of_dirty_lvl_90passrate]
#     #print "For level ",(lvl_i+1)," the agent reached ",val_of_dirty_lvl_90passrate,"% passrate with ",steps_of_dirty_lvl_90passrate," steps"
#
# #indx_of_dirty_1st_lvl_90passrate = find_first_val(died_at_rel_value[1][0],0.9)
# #steps_of_dirty_1st_lvl_90passrate = cursteps_value[1][0][indx_of_dirty_1st_lvl_90passrate]
# #print "steps=",steps_of_dirty_1st_lvl_90passrate," indx=",indx_of_dirty_1st_lvl_90passrate
#
#
#
# for lvl_i in range(0,5):
#     print "stam"
#     #first_90_passrate[0][lvl_i].append(find_first_val(died_at_rel_value[0][lvl_i],0.9)+1)
#     #first_90_passrate[1][lvl_i].append(find_first_val(died_at_rel_value[1][lvl_i],0.9)+1)
#
# print "searching steps"
# for lvl_i in range(0,5):
#     print "hi saar"
#     #first_90_passrate_num_of_steps[0][lvl_i].append(find_first_val(died_at_rel_value[0][lvl_i],0.9))
#     #first_90_passrate_num_of_steps[1][lvl_i].append(find_first_val(cursteps_value[1][lvl_i], float(steps_of_dirty_1st_lvl_90passrate))+1)
#
# #print "list is =",first_90_passrate_num_of_steps
#
# ## ####
# #PLOT MODEL
# ####
# (model_avg,model_var_high,model_var_low) = get_model(died_at_rel_value[0])
# fig16, ax16 = plt.subplots()
# ax_idx = ax16
# ax_idx.set_title('Model vs. results')
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), reward_value[0],'b',label=lognames[0])
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_avg,'r',label=lognames[0] + ' smooth')
# #ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_var_high,'g',label=lognames[0] + ' smooth')
# #ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_var_low,'g',label=lognames[0] + ' smooth')
#
#
# ## ####
# #PLOT MODEL from smooth
# ####
# (model_avg,model_var_high,model_var_low) = get_model(died_at_rel_value[0],True)
# fig26, ax26 = plt.subplots()
# ax_idx = ax26
# ax_idx.set_title('Model from smooth data vs. results')
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), reward_value[0],'b',label=lognames[0])
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_avg,'r',label=lognames[0] + ' smooth')
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_var_high,'g',label=lognames[0] + ' smooth')
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_var_low,'g',label=lognames[0] + ' smooth')
#
# ## ####
# #PLOT VARIANCE FROM MODEL from smooth
# ####
# model_avg = []
# model_var_high = []
# model_var_low = []
# var_g = []
# reward_value_sorted = []
# packed_v = []
# for lg in range(0,logs.__len__()):
#     model_avg.append(0)
#     model_var_high.append(9)
#     model_var_low.append(0)
#     var_g.append([])
#     packed_v.append([])
#     reward_value_sorted.append([])
#     (model_avg[lg],model_var_high[lg],model_var_low[lg]) = get_model(died_at_rel_value[lg],True)
#     for v in range(0,model_avg[lg].__len__()):
#         var_g[lg].append(model_var_high[lg][v]-model_avg[lg][v])
#         packed_v[lg].append((reward_value[lg][v],var_g[lg][v]))
#
#     packed_v[lg] = [x for (y,x) in sorted(packed_v[lg], key=lambda pair: pair[0])]
#     #var_g[lg] = [x for (y,x) in sorted(zip(reward_value[lg],var_g[lg]), key=lambda pair: pair[0])]
#     #var_g[lg] = packed_v[lg]
#     reward_value_sorted[lg] = reward_value[lg]
#
#
# fig26, ax26 = plt.subplots()
# ax_idx = ax26
# ax_idx.set_title('Variance vs. reward')
# #ax_idx.scatter(reward_value_sorted[0], var_g[0],color='g',label=lognames[0])
# ax_idx.scatter(reward_value_sorted[0], var_g[0],color='b',label=lognames[1])
# ax_idx.set_ylim([0,6000])
# ax_idx.set_xlim([0,14000])
#
# #ax_idx.scatter(reward_value_sorted[2], var_g[2],color='r',label=lognames[2] + ' smooth')
# #ax_idx.scatter(reward_value_sorted[3], var_g[3],color='y',label=lognames[3] + ' smooth')
#
#
# ## ####
# #PLOT MODEL2
# ####
# (model_avg,model_var_high,model_var_low) = get_model(died_at_rel_value[0])
# fig19, ax19 = plt.subplots()
# ax_idx = ax19
# ax_idx.set_title('Model vs. results')
# for ii in range(0,model_avg.__len__()):
#     model_avg[ii] = model_avg[147]
#     model_var_high[ii] = model_var_high[147]
#     model_var_low[ii] = model_var_low[147]
#
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), reward_value[0],'b',label=lognames[0])
# ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_avg,'r',label=lognames[0] + ' smooth')
# #ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_var_high,'g',label=lognames[0] + ' smooth')
# #ax_idx.plot(range(1,(reward_value[0].__len__()+1)), model_var_low,'g',label=lognames[0] + ' smooth')
#
#
#
# ###########
# ## PLOT PASS RATE
# ###########
#
# show_lvls_compare_graph(False,died_at_rel_value,'Pass percent at level ','Pass percent',1,True,True)
# show_lvls_compare_graph(True,died_at_rel_value,'Pass percent at level ','Pass percent',1)
#
# ###########
# ## PLOT PASS RATE WITH STEPS AS X AXIS
# ###########
#
# for lg in range(0,2):
#     for lvl_i in range(0,5):
#         prev_val = 0
#         for stp_idx in range(1,cursteps_value[lg][lvl_i].__len__()):
#             if float(cursteps_value[lg][lvl_i][stp_idx]) < float(cursteps_value[lg][lvl_i][stp_idx-1]):
#                 print "Found ! lg=",lg," ,, lvl_i=",lvl_i," ,, stp_idx=",stp_idx," ,, cursteps_value[lg][lvl_i][stp_idx]=",cursteps_value[lg][lvl_i][stp_idx],",, cursteps_value[lg][lvl_i][stp_idx-1]=",cursteps_value[lg][lvl_i][stp_idx-1]
#                 for rst_idx in range(0,(stp_idx+1)):
#                     cursteps_value[lg][lvl_i][rst_idx] = 0
#                     died_at_rel_value[lg][lvl_i][rst_idx] = 0
#                 for prnt_idx in range(stp_idx+1,(stp_idx+7)):
#                     print died_at_rel_value[lg][lvl_i][prnt_idx]," ,, ",cursteps_value[lg][lvl_i][prnt_idx]
#
#
#
# fig7, ax7 = plt.subplots(5,sharex=True)
# for lvl_i in range(0,5):
#   lvl_idx = lvl_i + 1
#   ax_idx = 4-lvl_i
#   #plt.subplot(6,1,6-lvl_idx) # 1 2 3 4 5
#   line1, = ax7[ax_idx].plot(cursteps_value[0][lvl_i], died_at_rel_value[0][lvl_i],'g',label=lognames[0])
#   line2, = ax7[ax_idx].plot(cursteps_value[1][lvl_i], died_at_rel_value[1][lvl_i],'b',label=lognames[1])
#   #ax[ax_idx].plot((first_90_passrate[0][lvl_i],first_90_passrate[0][lvl_i]), (0,1),'b')
#   #ax7[ax_idx].plot((first_90_passrate[1][lvl_i],first_90_passrate[1][lvl_i]), (0,1),'g')
#   #ax7[ax_idx].plot((first_90_passrate_num_of_steps[1][lvl_i],first_90_passrate_num_of_steps[1][lvl_i]), (0,1),'r')
#   ax7[ax_idx].set_ylim([0,1])
#   ax7[ax_idx].grid()
#   plt.xlabel('Steps')
#   ax7[ax_idx].set_ylabel('Steps')
#   ax7[ax_idx].set_ylabel('Pass precent')
#   ax7[ax_idx].set_title('Pass precent vs. steps at level %s'%(lvl_idx))
#   #ax.subplots_adjust(hspace=.5)
#   leg = ax7[ax_idx].legend(loc=1)
#   leg.get_frame().set_alpha(0.4)
#   #lines = [line1, line2]
#
#   #for legline, origline in zip(leg.get_lines(), lines):
#   #    legline.set_picker(5)  # 5 pts tolerance
#   #    lined[legline] = origline
#
#
# ###########
# ## PLOT PASS RATE WITH STEPS AS X AXIS - comparing shared network vs unshared
# ###########
# #
# # for lg in range(0,4):
# #     for lvl_i in range(0,5):
# #         prev_val = 0
# #         for stp_idx in range(1,cursteps_value[lg][lvl_i].__len__()):
# #             if float(cursteps_value[lg][lvl_i][stp_idx]) < float(cursteps_value[lg][lvl_i][stp_idx-1]):
# #                 print "Found ! lg=",lg," ,, lvl_i=",lvl_i," ,, stp_idx=",stp_idx," ,, cursteps_value[lg][lvl_i][stp_idx]=",cursteps_value[lg][lvl_i][stp_idx],",, cursteps_value[lg][lvl_i][stp_idx-1]=",cursteps_value[lg][lvl_i][stp_idx-1]
# #                 for rst_idx in range(0,(stp_idx+1)):
# #                     cursteps_value[lg][lvl_i][rst_idx] = 0
# #                     died_at_rel_value[lg][lvl_i][rst_idx] = 0
# #
# # fig17, ax17 = plt.subplots(5,sharex=True)
# # for lvl_i in range(0,5):
# #   lvl_idx = lvl_i + 1
# #   ax_idx = 4-lvl_i
# #   #plt.subplot(6,1,6-lvl_idx) # 1 2 3 4 5
# #   line1, = ax17[ax_idx].plot(cursteps_value[0][lvl_i], get_smooth_res(died_at_rel_value[0][lvl_i]),'g',label=lognames[0])
# #   line2, = ax17[ax_idx].plot(cursteps_value[1][lvl_i], get_smooth_res(died_at_rel_value[1][lvl_i]),'b',label=lognames[1])
# #   if logs.__len__() > 2:
# #     line3, = ax17[ax_idx].plot(new_stepsinlevel_value[2][lvl_i], get_smooth_res(died_at_rel_value[2][lvl_i]),'r',label=lognames[2])
# #   if logs.__len__() > 3:
# #     line4, = ax17[ax_idx].plot(new_stepsinlevel_value[3][lvl_i], get_smooth_res(died_at_rel_value[3][lvl_i]),'y',label=lognames[3])
# #   #ax[ax_idx].plot((first_90_passrate[0][lvl_i],first_90_passrate[0][lvl_i]), (0,1),'b')
# #   #ax7[ax_idx].plot((first_90_passrate[1][lvl_i],first_90_passrate[1][lvl_i]), (0,1),'g')
# #   #ax7[ax_idx].plot((first_90_passrate_num_of_steps[1][lvl_i],first_90_passrate_num_of_steps[1][lvl_i]), (0,1),'r')
# #   ax17[ax_idx].set_ylim([0,1])
# #   ax17[ax_idx].grid()
# #   plt.xlabel('Steps')
# #   ax17[ax_idx].set_ylabel('Steps')
# #   ax17[ax_idx].set_ylabel('Pass precent')
# #   ax17[ax_idx].set_title('Pass precent vs. steps at level %s'%(lvl_idx))
# #   #ax.subplots_adjust(hspace=.5)
# #   leg = ax17[ax_idx].legend(loc=1)
# #   leg.get_frame().set_alpha(0.4)
# #   #lines = [line1, line2]
#
#   #for legline, origline in zip(leg.get_lines(), lines):
#   #    legline.set_picker(5)  # 5 pts tolerance
#   #    lined[legline] = origline
#
# ###########
# ## PLOT NUM OF VISITS
# ###########
# show_lvls_compare_graph(False,visited_at_value,'Visited at level  ','Num of visits')
#
#
# ###########
# ## PLOT CURRENT EPSILON
# ###########
# show_lvls_compare_graph(False,curep_value,'Current epsilon of agent ','Epsilon value')
#
# ###########
# ## PLOT CURRENT STEPS
# ###########
# show_lvls_compare_graph(False,cursteps_value,'Steps of agent ','Steps')
#
#
# ###########
# ## PLOT CURRENT STEPS
# ###########
# show_lvls_compare_graph(False,new_stepsinlevel_value,'Steps of agent ','Steps')
#
# ###########
# ## PLOT TERMINALS STEPS
# ###########
# show_lvls_compare_graph(False,new_terminalsinlevel_value,'Num of terminals ','Terminals amount')
#
#
# ###########
# ## PLOT TERMINALS STEPS
# ###########
# show_lvls_compare_graph(False,terminal_visited_ratio,'Death ratio ','Ratio')
#
# ###########
# ## PLOT NEW CURRENT STEPS
# ###########
# show_lvls_compare_graph(False,died_at_total_value,'Died at total ','died at')
#
#
# ###########
# ## PLOT AVERAGE REWARD
# ###########
# #from scipy.interpolate import interp1d
# #f2 = np.polyfit(range(1,(reward_value[0].__len__()+1)), reward_value[0], 20)
# #print f2
# #xnew = np.linspace(1, (reward_value[0].__len__()), num=20, endpoint=True)
#
# fig2, ax2 = plt.subplots()
#
#
# #ax2.plot(xnew, f2,label=lognames[0])
# ax2.plot(range(1,(reward_value[0].__len__()+1)), reward_value[0],'g',label=lognames[0])
# ax2.plot(range(1,(reward_value[1].__len__()+1)), reward_value[1],'b',label=lognames[1])
# ax2.set_ylim([0,25000])
# ax2.grid()
# ax2.set_xlabel('Index of evaluation')
# ax2.set_ylabel('Average reward')
# ax2.set_title('Average reward')
# ax2.legend(loc=2)
#
# # smooth reward
# smooth_factor = 5
# reward_smooth = []
# reward_smooth.append([])
# reward_smooth.append([])
# reward_smooth.append([])
# reward_smooth.append([])
#
#
# for ind in range(0,logs.__len__()):
#     for i in range(0,smooth_factor-1):
#         reward_smooth[ind].append(float(reward_value[ind][i]))
#     for i in range(smooth_factor-1,reward_value[ind].__len__()-smooth_factor):
#         cur_sm = 0
#         for j in range(i-smooth_factor,i+smooth_factor+1):
#             cur_sm = cur_sm + int(reward_value[ind][j])
#         cur_sm = float(cur_sm) / float((2*smooth_factor)+1)
#         reward_smooth[ind].append(cur_sm)
#     for i in range(reward_value[ind].__len__()-smooth_factor,reward_value[ind].__len__()):
#         reward_smooth[ind].append(float(reward_value[ind][i]))
#
# fig15, ax15 = plt.subplots()
# ax_idx = ax15
# ax_idx.plot(range(1,(reward_smooth[0].__len__()+1)), get_smooth_res(reward_value[0]),'g',label=lognames[0] + ' smooth')
# ax_idx.plot(range(1,(reward_smooth[1].__len__()+1)), get_smooth_res(reward_value[1]),'b',label=lognames[1] + ' smooth')
# if logs.__len__() > 2:
#     ax_idx.plot(range(1,(reward_smooth[2].__len__()+1)), get_smooth_res(reward_value[2]),'r',label=lognames[2] + ' smooth')
# if logs.__len__() > 3:
#     ax_idx.plot(range(1,(reward_smooth[3].__len__()+1)), get_smooth_res(reward_value[3]),'y',label=lognames[3] + ' smooth')
#
# #ax_idx.plot(range(1,(reward_value[0].__len__()+1)), reward_value[0],label=lognames[0])
# #ax_idx.plot(range(1,(reward_value[1].__len__()+1)), reward_value[1],label=lognames[1])
# ax_idx.set_ylim([0,25000])
# ax_idx.grid()
# ax_idx.set_xlabel('Index of evaluation')
# ax_idx.set_ylabel('Average smooth reward')
# ax_idx.set_title('Average smooth reward')
# ax_idx.legend(loc=2)
#
#
#
#
#
# #ax_idx.plot(range(1,(reward_value[0].__len__()+1)), reward_value[0],label=lognames[0])
# #ax_idx.plot(range(1,(reward_value[1].__len__()+1)), reward_value[1],label=lognames[1])
# # ax_idx.set_ylim([0,25000])
# # ax_idx.grid()
# # ax_idx.set_xlabel('Index of evaluation')
# # ax_idx.set_ylabel('Average smooth reward with variance')
# # ax_idx.set_title('Average smooth reward with variance')
# # ax_idx.legend(loc=2)
#
# # fig14, ax14 = plt.subplots()
# # ax_idx = ax14
# # ax_idx.plot(range(1,(reward_value_calculated[0].__len__()+1)), reward_value_calculated[0],'g',label=lognames[0])
# # ax_idx.plot(range(1,(reward_value_calculated[1].__len__()+1)), reward_value_calculated[1],'b',label=lognames[1])
# # ax_idx.set_ylim([0,10000])
# # ax_idx.grid()
# # ax_idx.set_xlabel('Index of evaluation')
# # ax_idx.set_ylabel('Average reward without bonus')
# # ax_idx.set_title('Average reward without bonus')
# # ax_idx.legend(loc=2)
#
#
# plt.show()
