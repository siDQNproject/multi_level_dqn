--[[
Copyright (c) 2014 Google Inc.
See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')

cmd:option('-LVL1COLOR', 1, 'do color transformation')
cmd:option('-LVL2COLOR', 2, 'do color transformation')
cmd:option('-LVL3COLOR', 3, 'do color transformation')
cmd:option('-LVL4COLOR', 4, 'do color transformation')
cmd:option('-LVL5COLOR', 5, 'do color transformation')

cmd:option('-LVL1AGENT', 1, 'agent id of level')
cmd:option('-LVL2AGENT', 2, 'agent id of level')
cmd:option('-LVL3AGENT', 3, 'agent id of level')
cmd:option('-LVL4AGENT', 4, 'agent id of level')
cmd:option('-LVL5AGENT', 5, 'agent id of level')
cmd:option('-LVL6AGENT', 6, 'agent id of level')
cmd:option('-LVL7AGENT', 7, 'agent id of level')

cmd:option('-LVL1LEARN', 1, 'should we learn level')
cmd:option('-LVL2LEARN', 1, 'should we learn level')
cmd:option('-LVL3LEARN', 1, 'should we learn level')
cmd:option('-LVL4LEARN', 1, 'should we learn level')
cmd:option('-LVL5LEARN', 1, 'should we learn level')
cmd:option('-LVL6LEARN', 1, 'should we learn level')
cmd:option('-LVL7LEARN', 1, 'should we learn level')

cmd:option('-AGENT1NET', 'NO_NETWORK', 'network of agent')
cmd:option('-AGENT2NET', 'NO_NETWORK', 'network of agent')
cmd:option('-AGENT3NET', 'NO_NETWORK', 'network of agent')
cmd:option('-AGENT4NET', 'NO_NETWORK', 'network of agent')
cmd:option('-AGENT5NET', 'NO_NETWORK', 'network of agent')
cmd:option('-AGENT6NET', 'NO_NETWORK', 'network of agent')
cmd:option('-AGENT7NET', 'NO_NETWORK', 'network of agent')

cmd:option('-AGENT1EPSSTART', 1, 'epsilon start of agent')
cmd:option('-AGENT2EPSSTART', 1, 'epsilon start of agent')
cmd:option('-AGENT3EPSSTART', 1, 'epsilon start of agent')
cmd:option('-AGENT4EPSSTART', 1, 'epsilon start of agent')
cmd:option('-AGENT5EPSSTART', 1, 'epsilon start of agent')
cmd:option('-AGENT6EPSSTART', 1, 'epsilon start of agent')
cmd:option('-AGENT7EPSSTART', 1, 'epsilon start of agent')


cmd:option('-hide_score_pxls', 0, 'hide score pxls')
cmd:option('-num_of_agents_param', 1, 'number of agents')
cmd:option('-EVAL_ONLY_NUM', 0, 'number of evaluations')
cmd:option('-EVAL_EPS', 0.05, 'epsilon for evaluation')
cmd:option('-SI_SPLIT', 0, 'split agents or not')
cmd:option('-skillplay', 0, 'split agents or not')
cmd:option('-play_newgame', 0, 'play new game or not')
cmd:option('-debug_prints', 0, 'do debug prints or not')


cmd:text()

local game_env, game_actions, opt

local opt = cmd:parse(arg)

local opt_org = table.copy(opt)

function get_string_of_var(var_to_str)
      if var_to_str == nil then return "nil" end
      if type(var_to_str) == "boolean" then
        if var_to_str then
          return "true"
        else
          return "false"
        end
      end
      return var_to_str
end

-- ########################################################################################################################
-- ########################################################################################################################
-- ### CONSTS
-- ########################################################################################################################
-- ########################################################################################################################
local use_smart_eps_default_value          = false
local safe_eps_n_lowest_default_value      = 3
local smart_eps_choice_type_default_value  = "safeeps_nloest"
local clustering_eval_eps                  = 0.05
num_of_clusters                            = 5
use_networks_mode_default                  = 2

print("use_smart_eps_default_value is " .. get_string_of_var(use_smart_eps_default_value))
print("safe_eps_n_lowest_default_value is " .. safe_eps_n_lowest_default_value)
print("smart_eps_choice_type_default_value is " .. smart_eps_choice_type_default_value)
--smart_eps_choice_type can be:
-- "softmax_linear"
-- "softmax_pow7"
-- "softmax_pow3"
-- "softmax_exp1"
-- "softmax_normalized_exp"
-- "safeeps_nloest"
--    use_networks_mode = 0   -> use only clustering_net_current (which is freezed)
--    use_networks_mode = 1   -> use only clustering_net_future
--    use_networks_mode = 2   -> use network according to KNN

local stats_evl_smarteps_vals = {}
stats_evl_smarteps_vals[1] = {["use_smart_eps"]=use_smart_eps_default_value,["smart_eps_choice_type"]=smart_eps_choice_type_default_value,["safeeps_nloest"]=safe_eps_n_lowest_default_value,["epsilon_val"]=opt.EVAL_EPS,          ["eval_use_network_mode"]=2}
stats_evl_smarteps_vals[2] = {["use_smart_eps"]=false,                       ["smart_eps_choice_type"]="safeeps_nloest",                   ["safeeps_nloest"]=3,                             ["epsilon_val"]=0.0,                   ["eval_use_network_mode"]=2}
stats_evl_smarteps_vals[3] = {["use_smart_eps"]=false,                      ["smart_eps_choice_type"]="safeeps_nloest",                   ["safeeps_nloest"]=0,                              ["epsilon_val"]=0.0,                   ["eval_use_network_mode"]=1}
stats_evl_smarteps_vals[4] = {["use_smart_eps"]=false,                      ["smart_eps_choice_type"]="safeeps_nloest",                   ["safeeps_nloest"]=0,                              ["epsilon_val"]=clustering_eval_eps,   ["eval_use_network_mode"]=1}

cluster_on_eval = {}
cluster_on_eval[5] = true
cluster_on_eval[80] = true
cluster_on_eval[120] = true
cluster_on_eval[160] = true


local save_mid_nets = false
local save_mid_nets_steps = 10000000 -- Should by multiplication of save_freq

max_level_stats = 10
max_color_trans_level = 5
lvlbyscr_reward_history_size = 40
local lvlbyscr_pass_level_reward_threshold = 2500
use_networks_mode                          = use_networks_mode_default




-- Colors transform feature:


level_color = {}
level_color[1] = opt.LVL1COLOR
level_color[2] = opt.LVL2COLOR
level_color[3] = opt.LVL3COLOR
level_color[4] = opt.LVL4COLOR
level_color[5] = opt.LVL5COLOR

level_agent_id = {}
level_agent_id[1] = opt.LVL1AGENT
level_agent_id[2] = opt.LVL2AGENT
level_agent_id[3] = opt.LVL3AGENT
level_agent_id[4] = opt.LVL4AGENT
level_agent_id[5] = opt.LVL5AGENT
level_agent_id[6] = opt.LVL6AGENT
level_agent_id[7] = opt.LVL7AGENT

learn_level = {}
learn_level[1] = opt.LVL1LEARN
learn_level[2] = opt.LVL2LEARN
learn_level[3] = opt.LVL3LEARN
learn_level[4] = opt.LVL4LEARN
learn_level[5] = opt.LVL5LEARN
learn_level[6] = opt.LVL6LEARN
learn_level[7] = opt.LVL7LEARN


agent_net = {}
agent_net[1] = opt.AGENT1NET
agent_net[2] = opt.AGENT2NET
agent_net[3] = opt.AGENT3NET
agent_net[4] = opt.AGENT4NET
agent_net[5] = opt.AGENT5NET
agent_net[6] = opt.AGENT6NET
agent_net[7] = opt.AGENT7NET

agent_epsstart = {}
agent_epsstart[1] = opt.AGENT1EPSSTART
agent_epsstart[2] = opt.AGENT2EPSSTART
agent_epsstart[3] = opt.AGENT3EPSSTART
agent_epsstart[4] = opt.AGENT4EPSSTART
agent_epsstart[5] = opt.AGENT5EPSSTART
agent_epsstart[6] = opt.AGENT6EPSSTART
agent_epsstart[7] = opt.AGENT7EPSSTART

si_split_agent = opt.SI_SPLIT
si_split_agent = (si_split_agent > 0)
si_split_agent = true

skillplay_is_playing = opt.skillplay
skillplay_is_playing = (skillplay_is_playing > 0)

eval_only_num = opt.EVAL_ONLY_NUM
eval_only_mode = (eval_only_num > 0)

newgame_is_playing = opt.play_newgame
newgame_is_playing = (newgame_is_playing > 0)

do_debug_prints = opt.debug_prints
do_debug_prints = (do_debug_prints > 0)

if eval_only_mode then
        opt.steps = eval_only_num
end



--- General setup.
local num_of_agents
local opt_org = table.copy(opt)
num_of_agents = opt.num_of_agents_param
opt_of_agent = {}
for ag=1,num_of_agents do
        opt_of_agent[ag] = table.copy(opt_org)
end
agent = {}



-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

-- SI varsopt

local si_hide_score = opt.hide_score_pxls
local current_level = 1
local game_over_flag = false
local agents_use_smart_eps_string_vals = {}
local agents_safe_eps_nlowest_vals = {}
local agents_smart_eps_choice_type_vals = {}
local agents_use_smart_eps_saved_vals = {}
local agents_safe_eps_nlowest_saved_vals = {}
local agents_smart_eps_choice_type_saved_vals = {}
local stats_evl_visits_in_level = {}
local stats_evl_steps_in_level = {}
local stats_evl_gameovers_in_level = {}
local stats_evl_terminals_in_level = {}
local stats_evl_terminals_per_agent = {}
local stats_evl_steps_per_agent = {}
local stats_evl_total_lives_in_level_start = {}
local stats_evl_reward_in_level = {}
local stats_evl_max_games = 400
local stats_evl_current_game_idx = 1
local stats_evl_reward_per_game = {}
local stats_evl_num_of_terminals = {}
local stats_evl_level_per_terminal = {}
local stats_evl_avg_step_per_terminal = {}
local stats_evl_avg_step_per_agent = {}
local stats_evl_avg_reward_per_terminal = {}
local stats_evl_avg_reward_per_agent = {}
local stats_ttl_total_steps_in_level = {}
local stats_ttl_learning_steps_in_level = {}
local stats_ttl_choosed_skill = {}
local stats_ttl_choosed_skill_detailed = {}
local stats_ttl_choosed_skill_per_level = {}
local stats_evl_choosed_skill = {}
local stats_evl_choosed_skill_detailed = {}
local stats_evl_choosed_skill_per_level = {}
local stats_evl_clustered_as_per_level = {}
local stats_evl_clustered_as_per_agent = {}
local stats_evl_clustering_acc_per_agent = {}
local stats_agent_learning_steps_level = {}
local newgame_lives_lost_current_level = 0
local newgame_lives_lost_previous_levels = 0
local lvlbyscr_current_level = 1
local lvlbyscr_reward_history = {}
local lvlbyscr_reward_history_sum = 0
local lvlbyscr_reward_history_pointer = 1
local stepscounter_current_game_steps = 0
local terminalcount_current_lives = 4
local terminalcount_previous_lives = 4
local terminalcount_gameover_when_terminal = false
local terminalcount_steps_to_ignore_head_threshold = 75
local terminalcount_steps_to_ignore_head = 1
local terminalcount_per_level = {}
local terminalcount_total_step = 0
local learning_active_for_level = {}
local notlearning_epsilon_for_level = {}
local learning_active_for_agent = {}
local skillplay_max_original_action = 0
local splitting_max_learned_level = 1
local evaluating = false
local newgame_previous_level_terminal_count = 0
local episode_idx = 1
local episode_reward
local last_chosen_agent = 1

local eval_in_eval_idx = 1
local eval_total_eval_idx = 1
current_use_network_mode = 2


-- Reset variables


--stats_evl_level_per_terminal[y][x] is the number of times the yth terminal was in the xth level
-- For example:
--stats_evl_level_per_terminal[1][2] is the number of times the 1st terminal was in the 2nd level
stats_evl_level_per_terminal[1] = {}  -- 1st ternminal
stats_evl_level_per_terminal[2] = {}  -- 2nd terminal
stats_evl_level_per_terminal[3] = {}  -- 3rd terminal
stats_evl_level_per_terminal[4] = {}  -- 4th terminal


stats_evl_num_of_terminals[1]  = 0
stats_evl_num_of_terminals[2]  = 0
stats_evl_num_of_terminals[3]  = 0
stats_evl_num_of_terminals[4]  = 0

stats_evl_avg_step_per_terminal[1]  = 0
stats_evl_avg_step_per_terminal[2]  = 0
stats_evl_avg_step_per_terminal[3]  = 0
stats_evl_avg_step_per_terminal[4]  = 0

stats_evl_avg_reward_per_terminal[1] = 0
stats_evl_avg_reward_per_terminal[2] = 0
stats_evl_avg_reward_per_terminal[3] = 0
stats_evl_avg_reward_per_terminal[4] = 0


for rst_idx=1,max_level_stats do
        learning_active_for_level[rst_idx] = true
        notlearning_epsilon_for_level[rst_idx] = 0.05
        stats_evl_level_per_terminal[1][rst_idx] = 0
        stats_evl_level_per_terminal[2][rst_idx] = 0
        stats_evl_level_per_terminal[3][rst_idx] = 0
        stats_evl_level_per_terminal[4][rst_idx] = 0

        stats_evl_total_lives_in_level_start[rst_idx] = 0
        stats_ttl_total_steps_in_level[rst_idx] = 0
        terminalcount_per_level[rst_idx] = 0
end
for ag=1,num_of_agents do
        learning_active_for_agent[ag] = false
        stats_evl_avg_step_per_agent[ag] = 0
        stats_evl_avg_reward_per_agent[ag] = 0
        stats_agent_learning_steps_level[ag] = {}
        stats_ttl_choosed_skill_detailed[ag] = {}
        agents_use_smart_eps_string_vals[ag] = "false"
        agents_smart_eps_choice_type_vals[ag] = nil
        agents_safe_eps_nlowest_vals[ag] = 0
        stats_ttl_choosed_skill_per_level[ag] = {}
        stats_ttl_choosed_skill[ag] = 0
        stats_evl_choosed_skill_detailed[ag] = {}
        stats_evl_choosed_skill_per_level[ag] = {}
        stats_evl_choosed_skill[ag] = 0
        for rst_idx=1,max_level_stats do
                stats_agent_learning_steps_level[ag][rst_idx] = 0
                stats_ttl_choosed_skill_detailed[ag][rst_idx] = 0
                stats_ttl_choosed_skill_per_level[ag][rst_idx] = 0
                stats_evl_choosed_skill_detailed[ag][rst_idx] = 0
                stats_evl_choosed_skill_per_level[ag][rst_idx] = 0

        end
end
for gm=1,stats_evl_max_games do
        stats_evl_reward_per_game[gm] = 0
end


----------------------------------------------------
-- Some debug functions
----------------------------------------------------
lost_life_caller=""

----------------------------------------------------
-- Detect current level by score functions
----------------------------------------------------
function lvlbyscr_add_reward_to_history(reward_value)
        -- Update sum
        lvlbyscr_reward_history_sum = lvlbyscr_reward_history_sum + reward_value
        lvlbyscr_reward_history_sum = lvlbyscr_reward_history_sum - lvlbyscr_reward_history[lvlbyscr_reward_history_pointer]
        -- Add reward to history
        lvlbyscr_reward_history[lvlbyscr_reward_history_pointer] = reward_value
        -- Add to stats
        stats_evl_reward_in_level[current_level] = (stats_evl_reward_in_level[current_level] or 0) + reward_value

        -- Increase pointer
        lvlbyscr_reward_history_pointer = lvlbyscr_reward_history_pointer + 1
        -- Reset pointer if we passed the maximum array value
        if lvlbyscr_reward_history_pointer > lvlbyscr_reward_history_size then
                lvlbyscr_reward_history_pointer = 1
        end
        return true
end

function lvlbyscr_reset_reward_history()
        for reward_idx=1,lvlbyscr_reward_history_size do
                lvlbyscr_reward_history[reward_idx] = 0
        end
        lvlbyscr_reward_history_sum = 0
        return true
end
lvlbyscr_reset_reward_history()


function lvlbyscr_passed_level()
        if lvlbyscr_reward_history_sum > lvlbyscr_pass_level_reward_threshold then
                lvlbyscr_reset_reward_history()
                return true
        end
        return false
end


function lvlbyscr_get_level()
        -- Detect new level
        if lvlbyscr_passed_level() then
                lvlbyscr_current_level = lvlbyscr_current_level + 1
                -- Update visits stats
                stats_evl_visits_in_level[lvlbyscr_current_level] = (stats_evl_visits_in_level[lvlbyscr_current_level] or 0) + 1
                stats_evl_total_lives_in_level_start[lvlbyscr_current_level] = (stats_evl_total_lives_in_level_start[lvlbyscr_current_level] or 0) + terminalcount_current_lives
        end
        -- Update steps stats
        stats_evl_steps_in_level[lvlbyscr_current_level] = (stats_evl_steps_in_level[lvlbyscr_current_level] or 0) + 1
        stats_ttl_total_steps_in_level[lvlbyscr_current_level] = (stats_ttl_total_steps_in_level[lvlbyscr_current_level] or 0) + 1

        return lvlbyscr_current_level
end


----------------------------------------------------
-- Terminal counter functions
----------------------------------------------------

function terminalcount_sync()
        -- This function update current lives and previous lives
        terminalcount_previous_lives = terminalcount_current_lives
        terminalcount_current_lives = game_env._state.lives
        return 1 -- Return value has no meaning
end

function terminalcount_did_lost_life(term_var)
        -- This function return true if player lost life
        -- Should be called AFTER terminalcount_sync()
        if term_var or (terminalcount_current_lives < terminalcount_previous_lives) then
            -- Update stats (stats_evl_level_per_terminal)
            which_terminal = 1
            if (terminalcount_current_lives < terminalcount_previous_lives) then
                if (terminalcount_current_lives == 3) then
                    which_terminal = 1
                elseif (terminalcount_current_lives == 2) then
                    which_terminal = 2
                elseif (terminalcount_current_lives == 1) then
                    which_terminal = 3
                elseif (terminalcount_current_lives == 0) then
                    which_terminal = 4
                else
                  print("STRANGE BUG BUG terminalcount_current_lives=",terminalcount_current_lives," and terminalcount_previous_lives=",terminalcount_previous_lives)
                end
            elseif (terminalcount_current_lives == 1 and term_var) then
                which_terminal = 4
            else
                print("BUG BUG BUG terminalcount_current_lives=",terminalcount_current_lives," and terminalcount_previous_lives=",terminalcount_previous_lives)
            end
            stats_evl_level_per_terminal[which_terminal][current_level] = (stats_evl_level_per_terminal[which_terminal][current_level] or 0) + 1
            stats_evl_avg_step_per_terminal[which_terminal] = ((stats_evl_avg_step_per_terminal[which_terminal]*stats_evl_num_of_terminals[which_terminal]) + stepscounter_current_game_steps)/(stats_evl_num_of_terminals[which_terminal]+1)
            stats_evl_avg_reward_per_terminal[which_terminal] = ((stats_evl_avg_reward_per_terminal[which_terminal]*stats_evl_num_of_terminals[which_terminal]) + (episode_reward or 0))/(stats_evl_num_of_terminals[which_terminal]+1)
            stats_evl_num_of_terminals[which_terminal] = (stats_evl_num_of_terminals[which_terminal] or 0) + 1

            -- Return true since that the function purpose
            return true
        end
        return false
end

function terminalcount_did_gameover()
        -- This function return true if a new game started (also means the previous one ended with gameover)
        -- Should be called AFTER terminalcount_sync()
        if (terminalcount_current_lives == 4) and (terminalcount_previous_lives < 2) then
            return true
        end
        return false
end

function evlstats_lost_life()
        -- We call this function when we lost life and update stats
        stats_evl_terminals_in_level[current_level] = (stats_evl_terminals_in_level[current_level] or 0) + 1
        stats_evl_terminals_per_agent[last_chosen_agent] = stats_evl_terminals_per_agent[last_chosen_agent] + 1
        return 1 -- Return value has no meaning
end


----------------------------------------------------
-- Game over \ new level functions
----------------------------------------------------
function resets_new_level()
        lvlbyscr_reset_reward_history()
        return true -- Return value has no meaning
end

function resets_game_over()
        lvlbyscr_reset_reward_history()
        stats_evl_visits_in_level[1] = (stats_evl_visits_in_level[1] or 0) + 1
        stats_evl_total_lives_in_level_start[1] = (stats_evl_total_lives_in_level_start[1] or 0) + 4
        stats_evl_gameovers_in_level[current_level] = (stats_evl_gameovers_in_level[current_level] or 0) + 1
        lvlbyscr_current_level = 1
        stepscounter_current_game_steps = 0
        episode_idx = episode_idx + 1
        return true -- Return value has no meaning
end

function stats_resets()
        stats_evl_agents_switch = {}
        for ag=1,num_of_agents do
                stats_evl_terminals_per_agent[ag] = 0
                stats_evl_avg_step_per_terminal[ag] = 0
                stats_evl_steps_per_agent[ag] = 0
                stats_evl_clustering_acc_per_agent[ag] = 1
                stats_evl_choosed_skill_detailed[ag] = {}
                stats_evl_choosed_skill_per_level[ag] = {}
                stats_evl_choosed_skill[ag] = 0
                stats_evl_clustered_as_per_agent[ag] = {}
                for rst_idx=1,max_level_stats do
                        stats_evl_choosed_skill_detailed[ag][rst_idx] = 0
                        stats_evl_choosed_skill_per_level[ag][rst_idx] = 0
                        stats_evl_clustered_as_per_agent[ag][rst_idx] = 0
                end
                stats_evl_agents_switch[ag] = {}
                for ag_ag=1,num_of_agents do
                    stats_evl_agents_switch[ag][ag_ag] = 0
                end
        end
        stats_evl_agents_switch_total = 0
        stats_evl_avg_step_per_terminal[1] = 0
        stats_evl_avg_step_per_terminal[2] = 0
        stats_evl_avg_step_per_terminal[3] = 0
        stats_evl_avg_step_per_terminal[4] = 0

        stats_evl_avg_reward_per_terminal[1] = 0
        stats_evl_avg_reward_per_terminal[2] = 0
        stats_evl_avg_reward_per_terminal[3] = 0
        stats_evl_avg_reward_per_terminal[4] = 0

        stats_evl_num_of_terminals[1] = 0
        stats_evl_num_of_terminals[2] = 0
        stats_evl_num_of_terminals[3] = 0
        stats_evl_num_of_terminals[4] = 0


        for rst_idx=1,max_level_stats do
                stats_evl_clustered_as_per_level[rst_idx] = {}
                for clstr_idx=1,(num_of_clusters+1) do
                    stats_evl_clustered_as_per_level[rst_idx][clstr_idx] = 0
                end
                stats_evl_total_lives_in_level_start[rst_idx] = 0
                stats_evl_reward_in_level[rst_idx] = 0
                stats_evl_visits_in_level[rst_idx] = 0
                stats_evl_steps_in_level[rst_idx] = 0
                stats_evl_gameovers_in_level[rst_idx] = 0
                stats_evl_terminals_in_level[rst_idx] = 0
                stats_evl_level_per_terminal[1][rst_idx] = 0
                stats_evl_level_per_terminal[2][rst_idx] = 0
                stats_evl_level_per_terminal[3][rst_idx] = 0
                stats_evl_level_per_terminal[4][rst_idx] = 0



        end
        for gm=1,stats_evl_max_games do
                stats_evl_reward_per_game[gm] = 0
        end
        stats_evl_current_game_idx = 1
        return true -- Return value has no meaning
end
stats_resets()

----------------------------------------------------
-- Screen manipulation functions
----------------------------------------------------

function screen_manipulate_screen(screen,current_level)
        gamename = opt_org.env
        screen = screen_transform_level_colors(screen,current_level,gamename)
        if (si_hide_score == 1) then
	        screen = screen_hide_score_pixels(screen,gamename)
        end
	return screen
end


function screen_hide_score_pixels(screen,game)
        score_coord = {}
        num_of_score_coords = 0
        if game == "breakout" then
                score_coord[1] = {{1,(16)}} --,{76,82}}
                num_of_score_coords = 1
        end
        if game == "qbert" then
                score_coord[1] = {{20,(30)},{20,73}}
                score_coord[2] = {{30,(45)},{20,57}}
                num_of_score_coords = 2
        end

        for color_ind=1,3 do
                for coord=1,num_of_score_coords do
                        screen[1][color_ind][score_coord[coord]] = 0
                end
        end
        return screen
end

function get_rand_int(lower_bound,upper_bound)
	rand_int = math.floor(torch.uniform(lower_bound,(upper_bound+1)))
	while rand_int > upper_bound do
		rand_int = math.floor(torch.uniform(lower_bound,(upper_bound+1)))
	end
	return rand_int
end

function screen_transform_level_colors(screen,from_level,game)
        if game ~= "qbert" then
                return screen
        end
        local color_transform_max_level = 5
        from_level = from_level or 1
        local to_level = level_color[from_level] or from_level
        if newgame_is_playing then
                -- When playing newgame we always transform from level 1 colors
                from_level = 1
                to_level = current_levelcurrent_use_network_mode
                if (not evaluating) and (level_color[1] == 777) then
                        to_level = 777
                end
        end
        if (to_level == 777) then
                if not evaluating then
                        to_level = get_rand_int(1,color_transform_max_level)
                else
                        to_level = from_level
                end
        end
        if ((from_level <= color_transform_max_level) and (to_level <= color_transform_max_level) and (from_level ~= to_level)) then
                        -- level1: (side of brick   in color=1) = 0.32941177487373 == 84/255
                        -- level1: (side of brick   in color=2) = 0.54117649793625 == 138/255
                        -- level1: (side of brick   in color=3) = 0.82352948188782 == 210/255
                        -- level1: (untouched brick in color=1) = 0.32156863808632 == 82/255
                        -- level1: (untouched brick in color=2) = 0.49411767721176 == 126/255
                        -- level1: (untouched brick in color=3) = 0.17647059261799 == 45/255
                        -- level1: (touched brick in color=1)   = 0.82352948188782 == 210/255
                        -- level1: (touched brick in color=2)   = 0.64313727617264 == 164/255
                        -- level1: (touched brick in color=3)   = 0.29019609093666 == 74/255
                        level1_side_of_brick_colors   = {[1] = 0.32941177487373, [2] = 0.54117649793625, [3] = 0.82352948188782}
                        level1_untouched_brick_colors = {[1] = 0.32156863808632, [2] = 0.49411767721176, [3] = 0.17647059261799}
                        level1_touched_brick_colors   = {[1] = 0.82352948188782, [2] = 0.64313727617264, [3] = 0.29019609093666}

                        level2_side_of_brick_colors   = {[1] = 0.78431379795074, [2] = (72/255), [3] = (72/255)}
                        level2_untouched_brick_colors = {[1] = 0.82352948188782, [2] = 0.64313727617264, [3] = 0.29019609093666}
                        level2_touched_brick_colors   = {[1] = 0.32156863808632, [2] = 0.49411767721176, [3] = 0.17647059261799}


                        --level	3: (side of brick in color=1) = 0.66666668653488 == 170.00000506639/255
                        --level	3: (side of brick in color=2) = 0.66666668653488 == 170.00000506639/255
                        --level	3: (side of brick in color=3) = 0.66666668653488 == 170.00000506639/255
                        --level	3: (untouched brick in color=1)   = 0.75294125080109 == 192.00001895428/255
                        --level	3: (untouched brick in color=2)   = 0.75294125080109 == 192.00001895428/255
                        --level	3: (untouched brick in color=3)   = 0.75294125080109 == 192.00001895428/255
                        --level	3: (touched brick in color=1)    = 0.43529415130615 == 111.00000858307/255
                        --level	3: (touched brick in color=2)    = 0.43529415130615 == 111.00000858307/255
                        --level	3: (touched brick in color=3)    = 0.43529415130615 == 111.00000858307/255
                        level3_side_of_brick_colors   = {[1] = 0.66666668653488, [2] = 0.66666668653488, [3] = 0.66666668653488}
                        level3_untouched_brick_colors = {[1] = 0.75294125080109, [2] = 0.75294125080109, [3] = 0.75294125080109}
                        level3_touched_brick_colors   = {[1] = 0.43529415130615, [2] = 0.43529415130615, [3] = 0.43529415130615}

                        --level	4: (side of brick in color=1) = 0.66666668653488 == 170.00000506639/255
                        --level	4: (side of brick in color=2) = 0.66666668653488 == 170.00000506639/255
                        --level	4: (side of brick in color=3) = 0.66666668653488 == 170.00000506639/255
                        --level	4: (untouched brick in color=1)   = 0.52941179275513 ==
                        --level	4: (untouched brick in color=2)   = 0.71764707565308 ==
                        --level	4: (untouched brick in color=3)   = 0.32941177487373 ==
                        --level	4: (touched brick in color=1)    = 0.82352948188782 ==
                        --level	4: (touched brick in color=2)    = 0.64313727617264 ==
                        --level	4: (touched brick in color=3)    = 0.29019609093666 ==
                        level4_side_of_brick_colors   = {[1] = 0.66666668653488, [2] = 0.66666668653488, [3] = 0.66666668653488}
                        level4_untouched_brick_colors = {[1] = 0.52941179275513, [2] = 0.71764707565308, [3] = 0.32941177487373}
                        level4_touched_brick_colors   = {[1] = 0.82352948188782, [2] = 0.64313727617264, [3] = 0.29019609093666}


                        level5_side_of_brick_colors   = {[1] = 0.78431379795074, [2] = (72/255), [3] = (72/255)}
                        level5_untouched_brick_colors = {[1] = 0.32156863808632, [2] = 0.49411767721176, [3] = 0.17647059261799}
                        level5_touched05_brick_colors = {[1] = 0.82352948188782, [2] = 0.64313727617264, [3] = 0.29019609093666}
                        level5_touched_brick_colors   = {[1] = 0.72156864404678, [2] = 0.27450981736183, [3] = 0.63529413938522}
                        -- level5: Yellow -> Green -> Purple

                        -- Levels 1-4 doesn't have mid color (touched05), so we assign their touch05 value to be the touch value
                        levels_colors_side_of_brick   = {[1] = level1_side_of_brick_colors   , [2] = level2_side_of_brick_colors   , [3] = level3_side_of_brick_colors   , [4] = level4_side_of_brick_colors , [5] = level5_side_of_brick_colors}
                        levels_colors_untouched_brick = {[1] = level1_untouched_brick_colors , [2] = level2_untouched_brick_colors , [3] = level3_untouched_brick_colors , [4] = level4_untouched_brick_colors , [5] = level5_untouched_brick_colors}
                        levels_colors_touched05_brick = {[1] = level1_touched_brick_colors   , [2] = level2_touched_brick_colors   , [3] = level3_touched_brick_colors   , [4] = level4_touched_brick_colors , [5] = level5_touched05_brick_colors}
                        levels_colors_touched_brick   = {[1] = level1_touched_brick_colors   , [2] = level2_touched_brick_colors   , [3] = level3_touched_brick_colors   , [4] = level4_touched_brick_colors , [5] = level5_touched_brick_colors}


                        --Transform into temporary color to avoid collisions
                        -- Order is important ! (first transform touched and then touched05)
                        screen[1][1][screen[1][1]:eq(levels_colors_side_of_brick[from_level][1])]  = 0.111
                        screen[1][2][screen[1][2]:eq(levels_colors_side_of_brick[from_level][2])]  = 0.121
                        screen[1][3][screen[1][3]:eq(levels_colors_side_of_brick[from_level][3])]  = 0.131

                        screen[1][1][screen[1][1]:eq(levels_colors_untouched_brick[from_level][1])]  = 0.311
                        screen[1][2][screen[1][2]:eq(levels_colors_untouched_brick[from_level][2])]  = 0.321
                        screen[1][3][screen[1][3]:eq(levels_colors_untouched_brick[from_level][3])]  = 0.331

                        screen[1][1][screen[1][1]:eq(levels_colors_touched_brick[from_level][1])]  = 0.711
                        screen[1][2][screen[1][2]:eq(levels_colors_touched_brick[from_level][2])]  = 0.721
                        screen[1][3][screen[1][3]:eq(levels_colors_touched_brick[from_level][3])]  = 0.731

                        screen[1][1][screen[1][1]:eq(levels_colors_touched05_brick[from_level][1])]  = 0.811
                        screen[1][2][screen[1][2]:eq(levels_colors_touched05_brick[from_level][2])]  = 0.821
                        screen[1][3][screen[1][3]:eq(levels_colors_touched05_brick[from_level][3])]  = 0.831

                        --Transform temporary color to new color
                        screen[1][1][screen[1][1]:eq(0.111)]  =  levels_colors_side_of_brick[to_level][1]
                        screen[1][2][screen[1][2]:eq(0.121)]  =  levels_colors_side_of_brick[to_level][2]
                        screen[1][3][screen[1][3]:eq(0.131)]  =  levels_colors_side_of_brick[to_level][3]

                        screen[1][1][screen[1][1]:eq(0.311)]  =  levels_colors_untouched_brick[to_level][1]
                        screen[1][2][screen[1][2]:eq(0.321)]  =  levels_colors_untouched_brick[to_level][2]
                        screen[1][3][screen[1][3]:eq(0.331)]  =  levels_colors_untouched_brick[to_level][3]

                        screen[1][1][screen[1][1]:eq(0.711)]  =  levels_colors_touched_brick[to_level][1]
                        screen[1][2][screen[1][2]:eq(0.721)]  =  levels_colors_touched_brick[to_level][2]
                        screen[1][3][screen[1][3]:eq(0.731)]  =  levels_colors_touched_brick[to_level][3]

                        screen[1][1][screen[1][1]:eq(0.811)]  =  levels_colors_untouched_brick[to_level][1]
                        screen[1][2][screen[1][2]:eq(0.821)]  =  levels_colors_untouched_brick[to_level][2]
                        screen[1][3][screen[1][3]:eq(0.831)]  =  levels_colors_untouched_brick[to_level][3]
        end
        return screen
end

----------------------------------------------------
-- Agent's functions
----------------------------------------------------

function cluster_on_this_eval()
    return (current_eval_eps == clustering_eval_eps and current_use_network_mode == 1 and cluster_on_eval[eval_total_eval_idx])
end

function agent_set_smarteps(use_smart_eps_to_set,smart_eps_choice_type_to_set,safe_eps_nlowest_to_set,agent_id_to_set)
        -- If agent_id_to_set is not set, call the function for each agent
        if not agent_id_to_set then
          for ag=1,num_of_agents do
            agent_set_smarteps(use_smart_eps_to_set,smart_eps_choice_type_to_set,safe_eps_nlowest_to_set,ag)
          end
        end

        -- Set default values if necessary
        if use_smart_eps_to_set == nil then
          if agents_use_smart_eps_saved_vals[agent_id_to_set] ~= nil then
            -- If a saved value exists, use it
            use_smart_eps_to_set = agents_use_smart_eps_saved_vals[agent_id_to_set]
          else
            -- Use global default value
            use_smart_eps_to_set = use_smart_eps_default_value
          end
        end
        if safe_eps_nlowest_to_set == nil then
          if agents_safe_eps_nlowest_saved_vals[agent_id_to_set] ~= nil then
            -- If a saved value exists, use it
            safe_eps_nlowest_to_set = agents_safe_eps_nlowest_saved_vals[agent_id_to_set]
          else
            -- Use global default value
            safe_eps_nlowest_to_set = safe_eps_n_lowest_default_value
          end
        end
        if smart_eps_choice_type_to_set == nil then
          if agents_smart_eps_choice_type_saved_vals[agent_id_to_set] ~= nil then
            -- If a saved value exists, use it
            smart_eps_choice_type_to_set = agents_smart_eps_choice_type_saved_vals[agent_id_to_set]
          else
            -- Use global default value
            smart_eps_choice_type_to_set = smart_eps_choice_type_default_value
          end
        end

        -- Set the values
        if agent[agent_id_to_set] then
          agent[agent_id_to_set].use_smart_eps               = use_smart_eps_to_set
          agent[agent_id_to_set].safe_eps_n_lowest           = safe_eps_nlowest_to_set
          agent[agent_id_to_set].smart_eps_choice_type       = smart_eps_choice_type_to_set
          agents_use_smart_eps_string_vals[agent_id_to_set]  = get_string_of_var(use_smart_eps_to_set)
          agents_safe_eps_nlowest_vals[agent_id_to_set]      = safe_eps_nlowest_to_set
          agents_smart_eps_choice_type_vals[agent_id_to_set] = smart_eps_choice_type_to_set
        end
        return true -- Return value has no meaning
end

function agents_save_smarteps_vals()
    for ag=1,num_of_agents do
      if agent[ag] then
        agents_use_smart_eps_saved_vals[ag]         = agent[ag].use_smart_eps
        agents_safe_eps_nlowest_saved_vals[ag]      = agent[ag].safe_eps_n_lowest
        agents_smart_eps_choice_type_saved_vals[ag] = agent[ag].smart_eps_choice_type
      end
    end
    return true -- Return value has no meaning
end

function get_agent_id(level)
	level = level or 1
	agent_id_tmp = 1
	--- insert agent id code here:
	agent_id_tmp = level_agent_id[level] or 1
	--- end agent id code
	if (agent_id_tmp < 1 or agent_id_tmp > num_of_agents) then -- protection from illegal agent id
		agent_id_tmp = num_of_agents
	end
	if not agent[agent_id_tmp] then
		return 1
	end
	return agent_id_tmp
end


function agent_perceive(reward, screen, terminal, testing, testing_ep, agent_to_use)
    -- param1 = testing
    -- param2 = testing_ep
    -- Modes:
    --    use_networks_mode = 0   -> use only clustering_net_current (which is freezed)
    --    use_networks_mode = 1   -> use only clustering_net_future
    --    use_networks_mode = 2   -> use network according to KNN
    --nql:perceive(reward, rawstate, terminal, testing, testing_ep, current_level, episode_idx, get_act_and_q_from_saved_network)

    if use_networks_mode == 0 then
        selected_act = agent[1]:perceive(reward, screen, terminal, testing, testing_ep, current_level, episode_idx, true)
        stats_evl_steps_per_agent[1] = (stats_evl_steps_per_agent[1] or 0) + 1
    elseif use_networks_mode == 1 then
        selected_act = agent[1]:perceive(reward, screen, terminal, testing, testing_ep, current_level, episode_idx, false)
        stats_evl_steps_per_agent[1] = (stats_evl_steps_per_agent[1] or 0) + 1
    else
        -- Else: use_networks_mode = 2
        if not agent[1].saved_network then
            -- No saved network (aka "current_clustering_network")  ==> Just play with future_clustering_network
            selected_act = agent[1]:perceive(reward, screen, terminal, testing, testing_ep, current_level, episode_idx, false)
            stats_evl_steps_per_agent[1] = (stats_evl_steps_per_agent[1] or 0) + 1
        else
            -- Teach future clustering network:

            -- Forward in current clustering network:
            selected_act = agent[1]:perceive(reward, screen, terminal, true, 0, current_level, episode_idx, true)
            -- Select agent according to last activation (which is the current_clustering_network's activation)
            agent_to_use = 1 + agent[1]:get_cluster_of_last_state()
            if terminal and last_chosen_agent then
                agent_to_use = last_chosen_agent
            end
            if not agent[agent_to_use] then
                -- Create agent from main agent
                split_agent(1,agent_to_use)
            end
            if agent_to_use ~= last_chosen_agent then
                stats_evl_agents_switch[last_chosen_agent][agent_to_use] = stats_evl_agents_switch[last_chosen_agent][agent_to_use] + 1
                stats_evl_agents_switch_total = stats_evl_agents_switch_total + 1
            end
            last_chosen_agent = agent_to_use
            stats_evl_avg_step_per_agent[agent_to_use] = ((stats_evl_avg_step_per_agent[agent_to_use]*stats_evl_steps_per_agent[agent_to_use]) + stepscounter_current_game_steps)/(stats_evl_steps_per_agent[agent_to_use]+1)
            stats_evl_avg_reward_per_agent[agent_to_use] = ((stats_evl_avg_reward_per_agent[agent_to_use]*stats_evl_steps_per_agent[agent_to_use]) + (episode_reward or 0))/(stats_evl_steps_per_agent[agent_to_use]+1)
            stats_evl_steps_per_agent[agent_to_use] = (stats_evl_steps_per_agent[agent_to_use] or 0) + 1
            stats_evl_clustered_as_per_level[current_level][agent_to_use] = stats_evl_clustered_as_per_level[current_level][agent_to_use] + 1
            stats_evl_clustered_as_per_agent[agent_to_use][current_level] = stats_evl_clustered_as_per_agent[agent_to_use][current_level] + 1
            if learning_active_for_agent[agent_to_use] then
                selected_act = agent[agent_to_use]:perceive(reward, screen, terminal, testing, testing_ep, current_level, episode_idx, false)
            else
                selected_act = agent[agent_to_use]:perceive(reward, screen, terminal, true, 0.0, current_level, episode_idx, false)
            end
        end
    end
    return selected_act
end


function create_agent(new_agent_id)
        if (agent_net[new_agent_id] == 'NO_NETWORK') then
	        opt_of_agent[new_agent_id].network = ''
                print("Creating agent ",new_agent_id," from scratch")
        else
	        opt_of_agent[new_agent_id].network = agent_net[new_agent_id]
                print("Creating agent ",new_agent_id," from network ",agent_net[new_agent_id])
        end
        opt_of_agent[new_agent_id].agent_params = opt_of_agent[new_agent_id].agent_params .. ",ep=" .. agent_epsstart[new_agent_id]
        print("params of agent ",new_agent_id," are: ",opt_of_agent[new_agent_id].agent_params)
        learning_active_for_agent[new_agent_id] = true
        skillplay_num_of_skills = 0
        if skillplay_is_playing then
                skillplay_num_of_skills = new_agent_id - 1
        end
        game_env, game_actions, agent[new_agent_id], opt, skillplay_max_original_action = setup(opt_of_agent[new_agent_id],new_agent_id,skillplay_num_of_skills)
        agent_set_smarteps(nil,nil,nil,new_agent_id) -- Set safe eps values to default values
        if new_agent_id > 1 then
            agent[new_agent_id]:delete_activation_mem()
        end
        print(get_string_of_vals(game_actions,"game_actions"))
end

function agent_update_ep_end(ag_id,new_ep_end)
        if agent[ag_id] then
                agent[ag_id].ep_end = new_ep_end
        end
end

function split_agent(source_agent_id,dest_agent_id)
	source_agent_net_file = agent_net[source_agent_id]
	dest_agent_net_file   = agent_net[source_agent_id] .. "_to_" .. dest_agent_id .. ".t7"
	os.execute("cp " .. source_agent_net_file .. " " .. dest_agent_net_file)
	agent_net[dest_agent_id] = dest_agent_net_file
    print("Splitting from " .. source_agent_id .. " to " .. dest_agent_id)
        create_agent(dest_agent_id)
end

----------------------------------------------------
-- Other
----------------------------------------------------
function get_string_of_vals(arr,arr_name)
        out_string = arr_name .. ": "
        local k_tmp = 1
        for k in pairs(arr) do
                if type(arr[k]) == type(table) then
                        -- multidim arr
                        out_string = out_string .. get_string_of_vals(arr[k],arr_name .. "[" .. k_tmp .. "]")
                else
                        out_string = out_string .. " , " .. arr_name .. "[" .. k .. "]=" .. get_string_of_var(arr[k]) .. " , "
                end
                k_tmp = k_tmp + 1
        end
        return out_string
end


----------------------------------------------------
-- Get params func
----------------------------------------------------

function si_getparams(called_function,param1,param2)
        if (called_function == "game_env:getState") then
	        screen, reward, terminal = game_env:getState()
        elseif (called_function == "game_env:step" and param2 ~= nil) then
	        screen, reward, terminal = game_env:step(param1, param2)
        elseif (called_function == "game_env:step" and param2 == nil) then
	        screen, reward, terminal = game_env:step(param1)
        elseif (called_function == "game_env:nextRandomGame") then
                resets_new_level()
	        screen, reward, terminal = game_env:nextRandomGame()
        elseif (called_function == "game_env:newGame") then
                resets_game_over()
	        screen, reward, terminal = game_env:newGame()
        else
	        print("ERROR!")
        end
        stats_step_lost_life = false
        -- Fix bug of getting minus reward instead of terminal
        if (reward < 0) then
	        reward = 0
	        terminal = true
        end

        -- Update relevant counters
        terminalcount_total_step = terminalcount_total_step + 1
        stepscounter_current_game_steps = stepscounter_current_game_steps + 1
        lvlbyscr_add_reward_to_history(reward)
        terminalcount_sync()

        -- Detect terminal
        if terminalcount_did_lost_life(terminal) then
            evlstats_lost_life()
        end

        -- Get current level
        previous_level = current_level
        current_level = lvlbyscr_get_level()
        assert(current_level)

        -- Detect new level
        if (current_level ~= previous_level) and (current_level > 1) then
                resets_new_level()
                -- Detect agent switching
          	if (get_agent_id(current_level) ~= get_agent_id(previous_level)) then
                	--reward_countdown = 0 -- Currently not in use
	        end
        end

        -- Detect game over
        if terminalcount_did_gameover() then
            resets_game_over()
        end


        screen = screen_manipulate_screen(screen,current_level)
        return screen, reward, terminal, current_level
end

-- Create agents

for ag=1,num_of_agents do
        if ((not si_split_agent) or (ag == 1)) then
                create_agent(ag)
        end
end

local learn_start = agent[1].learn_start

local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes


local screen, reward, terminal, current_level = si_getparams("game_env:getState",nil,nil)
print("Iteration ..", step)
local win = nil
while step < opt.steps do
        step = step + 1
        if not eval_only_mode then
                -- Play&Learn!
                local action_index = agent_perceive(reward, screen, terminal, nil, nil) -- Note it's not agent:perceive but a customized functions

                -- game over? get next game!
                if not terminal then
                        screen, reward, terminal, current_level = si_getparams("game_env:step",game_actions[action_index],true)
                else
                        if opt.random_starts > 0 then
                                screen, reward, terminal, current_level = si_getparams("game_env:nextRandomGame",nil,nil)
                        else
                                screen, reward, terminal, current_level = si_getparams("game_env:newGame",nil,nil)
                        end
                end

                -- display screen
                win = image.display({image=screen, win=win})

                if step % opt.prog_freq == 0 then
                        --assert(step==agent.numSteps, 'trainer step: ' .. step ..
                        --        ' & agent.numSteps: ' .. agent.numSteps)
                        print("Steps: ", step)
                        agent[get_agent_id(current_level)]:report()
                        collectgarbage()
                end
        end

        if step%1000 == 0 then collectgarbage() end


        if (use_networks_mode == 2) and ((step % opt.eval_freq == 0 and step > learn_start) or (eval_only_mode)) then
          eval_in_eval_idx = 1
          -- Save smarteps vals so we can restore it after evaluation
          agents_save_smarteps_vals()
          for smarteps_idx in pairs(stats_evl_smarteps_vals) do
            last_chosen_agent = 1
            agent_set_smarteps(stats_evl_smarteps_vals[smarteps_idx]["use_smart_eps"],stats_evl_smarteps_vals[smarteps_idx]["smart_eps_choice_type"],stats_evl_smarteps_vals[smarteps_idx]["safeeps_nloest"])
            -- In the end of evaluation we set smarteps back to their saved values
            current_eval_eps = stats_evl_smarteps_vals[smarteps_idx]["epsilon_val"]
            current_use_network_mode = stats_evl_smarteps_vals[smarteps_idx]["eval_use_network_mode"]
            use_networks_mode = current_use_network_mode
            evaluating = true
            print("Evaluation start")
            stats_resets()

            screen, reward, terminal, current_level = si_getparams("game_env:newGame",nil,nil)

            total_reward = 0
            nrewards = 0
            nepisodes = 0
            episode_reward = 0
            current_eval_steps = opt.eval_steps
            if cluster_on_this_eval() then
                print("Clustering_start ....")
                agent[1]:start_saving_activations()
                current_eval_steps = agent[1].activation_mem_size
            end
            local eval_time = sys.clock()

            for estep=1,current_eval_steps do
                    local action_index = agent_perceive(reward, screen, terminal, true, current_eval_eps)

                    -- Play game in test mode (episodes don't end when losing a life)
                    screen, reward, terminal, current_level = si_getparams("game_env:step",game_actions[action_index],nil)

                    -- display screen
                    win = image.display({image=screen, win=win})

                    if estep%1000 == 0 then collectgarbage() end

                    -- record every reward
                    episode_reward = episode_reward + reward
                    if reward ~= 0 then
                            nrewards = nrewards + 1
                    end

                    if terminal then
                            stats_evl_reward_per_game[stats_evl_current_game_idx] = episode_reward
                            stats_evl_current_game_idx = stats_evl_current_game_idx + 1
                            total_reward = total_reward + episode_reward
                            episode_reward = 0
                            nepisodes = nepisodes + 1
                            screen, reward, terminal, current_level = si_getparams("game_env:nextRandomGame",nil,nil)
                    end
            end

            eval_time = sys.clock() - eval_time
            start_time = start_time + eval_time
            if agent[get_agent_id(current_level)] and agent[get_agent_id(current_level)].args and agent[get_agent_id(current_level)].args.term then
                    agent[get_agent_id(current_level)]:compute_validation_statistics()
            end

            local ind = #reward_history+1
            total_reward = total_reward/math.max(1, nepisodes)

            if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
                    for ag_id = 1,num_of_agents do
                            if agent[ag_id] then
                                    agent[ag_id].best_network = agent[ag_id].network:clone() -- SI might affect the learning rate
                            end
                    end
            end
            if agent[get_agent_id(current_level)].v_avg then    -- SI may need to add a loop
                    v_history[ind] = agent[get_agent_id(current_level)].v_avg
                    td_history[ind] = agent[get_agent_id(current_level)].tderr_avg
                    qmax_history[ind] = agent[get_agent_id(current_level)].q_max
            end
            print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

            reward_history[ind] = total_reward
            reward_counts[ind] = nrewards
            episode_counts[ind] = nepisodes

            time_history[ind+1] = sys.clock() - start_time

            local time_dif = time_history[ind+1] - time_history[ind]

            local training_rate = opt.actrep*opt.eval_freq/time_dif

            local cur_ep = {}
            local cur_steps = {}
            local cur_ag = {}

            for ag_id = 1,num_of_agents do
                if agent[ag_id] then
                    cur_ep[ag_id] = (agent[ag_id].ep_end +
                    math.max(0, (agent[ag_id].ep_start - agent[ag_id].ep_end) * (agent[ag_id].ep_endt -
                    math.max(0, agent[ag_id].numSteps - agent[ag_id].learn_start))/agent[ag_id].ep_endt))
                    cur_steps[ag_id] = agent[ag_id].numSteps
                    cur_ag[ag_id] = ag_id
                else
                    cur_ep[ag_id] = 0
                    cur_steps[ag_id] = 0
                    cur_ag[ag_id] = ag_id
                end
                agent_num_of_events = 0
                for lv_idx=1,max_level_stats do
                    agent_num_of_events = agent_num_of_events + stats_evl_clustered_as_per_agent[ag_id][lv_idx]
                end
                stats_evl_clustering_acc_per_agent[ag_id] = math.max(1,math.max(unpack(stats_evl_clustered_as_per_agent[ag_id])))/math.max(1,agent_num_of_events)
            end
            -- Modes:
            --    use_networks_mode = 0   -> use only clustering_net_current (which is freezed)
            --    use_networks_mode = 1   -> use only clustering_net_future
            --    use_networks_mode = 2   -> use network according to KNN

--            if current_eval_eps == clustering_eval_eps and current_use_network_mode == 1 and cluster_on_eval[eval_total_eval_idx] then
            if cluster_on_this_eval() then
                agent[1]:cluster_activations(num_of_clusters,eval_total_eval_idx)
                agent[1]:save_network()
                agent[1]:stop_saving_activations()
                print(" eval: " .. eval_total_eval_idx .. " Clustering on this eval!")
            end
            if cluster_on_eval[eval_total_eval_idx] then
                print(" eval: " .. eval_total_eval_idx .. " eval_only_mode = true")
                eval_only_mode = true
            else
                if eval_only_mode then
                    print(" eval: " .. eval_total_eval_idx .. " eval_only_mode = false")
                    eval_only_mode = false
                end
            end
            if (not cluster_on_this_eval()) and current_use_network_mode == 2 and current_eval_eps == 0.0 and agent[1].saved_network then
                for ag_id = 2,num_of_agents do
                    if stats_evl_steps_per_agent[ag_id] > 10000 and stats_evl_terminals_per_agent[ag_id] < 4 then
                        -- Set epsilon to 0 and turn off learning
                        if learning_active_for_agent[ag_id] == true then
                            learning_active_for_agent[ag_id] = false
                            print("[OFF] eval: " .. eval_total_eval_idx .. " Turning OFF exploration for agent=" .. ag_id .. " due to steps=" .. stats_evl_steps_per_agent[ag_id] .. " and terminals=" .. stats_evl_terminals_per_agent[ag_id])
            				os.execute("cp nothing.txt stopped" .. ag_id)
                        end
                    else
                        -- Set epsilon to 0.1 and turn on learning
                        if learning_active_for_agent[ag_id] == false then
                            learning_active_for_agent[ag_id] = true
                            print("[ON] eval: " .. eval_total_eval_idx .. " Turning ON exploration for agent=" .. ag_id .. " due to steps=" .. stats_evl_steps_per_agent[ag_id] .. " and terminals=" .. stats_evl_terminals_per_agent[ag_id])
                        end
                    end
                end
            end
            print(string.format(
                '\nSteps: %d (frames: %d), reward: %.2f, eval_total_eval_idx: %d (eval_in_eval_idx: %d), epsilon: %.2f, lr: %G, ' ..
                'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
                'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d , ' ..
                'cur_agent: %s , cur_ep: %s , cur_steps: %s , ' ..
                'current_eval_eps: %s , current_use_network_mode: %d , agents_use_smart_eps_string_vals: %s , agents_safe_eps_nlowest_vals: %s , agents_smart_eps_choice_type_vals: %s ,' ..
                'agents_use_smart_eps_saved_vals: %s , agents_safe_eps_nlowest_saved_vals: %s , agents_smart_eps_choice_type_saved_vals: %s , ' ..
                'stats_evl_visits_in_level: %s , stats_evl_steps_in_level: %s , stats_evl_gameovers_in_level: %s , stats_evl_terminals_in_level: %s , ' ..
                'stats_evl_level_per_terminal: %s , stats_evl_avg_step_per_terminal: %s, stats_evl_avg_reward_per_terminal: %s, stats_evl_num_of_terminals: %s, ' ..
                'stats_evl_reward_in_level: %s , stats_evl_total_lives_in_level_start: %s , stats_evl_reward_per_game: %s, ' ..
                'stats_evl_clustered_as_per_agent: %s ,  stats_evl_agents_switch: %s , stats_evl_agents_switch_total: %d' ..
                'stats_evl_clustered_as_per_level: %s , stats_evl_terminals_per_agent: %s , stats_evl_steps_per_agent: %s , stats_evl_clustering_acc_per_agent: %s , stats_evl_avg_step_per_agent: %s , stats_evl_avg_reward_per_agent: %s , ' ..
                'stats_ttl_total_steps_in_level: %s , stats_ttl_learning_steps_in_level: %s , stats_agent_learning_steps_level: %s , ',
                step, step*opt.actrep, total_reward, eval_total_eval_idx, eval_in_eval_idx, agent[get_agent_id(current_level)].ep, agent[get_agent_id(current_level)].lr, time_dif,
                training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
                nepisodes, nrewards,
                get_string_of_vals(cur_ag,"cur_ag"), get_string_of_vals(cur_ep,"cur_ep"), get_string_of_vals(cur_steps,"cur_steps"),
                current_eval_eps,current_use_network_mode,get_string_of_vals(agents_use_smart_eps_string_vals,"agents_use_smart_eps_string_vals"),get_string_of_vals(agents_safe_eps_nlowest_vals,"agents_safe_eps_nlowest_vals"),get_string_of_vals(agents_smart_eps_choice_type_vals,"agents_smart_eps_choice_type_vals"),
                get_string_of_vals(agents_use_smart_eps_saved_vals,"agents_use_smart_eps_saved_vals"),get_string_of_vals(agents_safe_eps_nlowest_saved_vals,"agents_safe_eps_nlowest_saved_vals"),get_string_of_vals(agents_smart_eps_choice_type_saved_vals,"agents_smart_eps_choice_type_saved_vals"),
                get_string_of_vals(stats_evl_visits_in_level,"stats_evl_visits_in_level"), get_string_of_vals(stats_evl_steps_in_level,"stats_evl_steps_in_level"), get_string_of_vals(stats_evl_gameovers_in_level,"stats_evl_gameovers_in_level"), get_string_of_vals(stats_evl_terminals_in_level,"stats_evl_terminals_in_level"),
                get_string_of_vals(stats_evl_level_per_terminal,"stats_evl_level_per_terminal"), get_string_of_vals(stats_evl_avg_step_per_terminal,"stats_evl_avg_step_per_terminal"), get_string_of_vals(stats_evl_avg_reward_per_terminal,"stats_evl_avg_reward_per_terminal"), get_string_of_vals(stats_evl_num_of_terminals,"stats_evl_num_of_terminals"),
                get_string_of_vals(stats_evl_reward_in_level,"stats_evl_reward_in_level"), get_string_of_vals(stats_evl_total_lives_in_level_start,"stats_evl_total_lives_in_level_start"), get_string_of_vals(stats_evl_reward_per_game,"stats_evl_reward_per_game"),
                get_string_of_vals(stats_evl_clustered_as_per_agent,"stats_evl_clustered_as_per_agent"), get_string_of_vals(stats_evl_agents_switch,"stats_evl_agents_switch"), stats_evl_agents_switch_total,
                get_string_of_vals(stats_evl_clustered_as_per_level,"stats_evl_clustered_as_per_level"), get_string_of_vals(stats_evl_terminals_per_agent,"stats_evl_terminals_per_agent"), get_string_of_vals(stats_evl_steps_per_agent,"stats_evl_steps_per_agent"), get_string_of_vals(stats_evl_clustering_acc_per_agent,"stats_evl_clustering_acc_per_agent"), get_string_of_vals(stats_evl_avg_step_per_agent,"stats_evl_avg_step_per_agent"), get_string_of_vals(stats_evl_avg_reward_per_agent,"stats_evl_avg_reward_per_agent"),
                get_string_of_vals(stats_ttl_total_steps_in_level,"stats_ttl_total_steps_in_level"), get_string_of_vals(stats_ttl_learning_steps_in_level,"stats_ttl_learning_steps_in_level"), get_string_of_vals(stats_agent_learning_steps_level,"stats_agent_learning_steps_level")
                    ))
            print("Evaluation end")
            current_use_network_mode = 2
            evaluating = false
            eval_in_eval_idx = eval_in_eval_idx + 1
            last_chosen_agent = 1
          end -- end of for smarteps_idx in pairs(stats_evl_smarteps_vals) do
          agent_set_smarteps(nil,nil,nil) -- Set safe eps back to saved values
          eval_total_eval_idx = eval_total_eval_idx + 1
          use_networks_mode = use_networks_mode_default
          use_networks_mode = 1 
        end -- end of if (step % opt.eval_freq == 0 and step > learn_start) or (eval_only_mode) then
        if (use_networks_mode == 1) and ((step % opt.eval_freq == 0 and step > learn_start) or (eval_only_mode)) then
            use_networks_mode = 2
        end
        if (step % opt.save_freq == 0 or step == opt.steps) and (not eval_only_mode) then
                for agent_id = 1,num_of_agents do
                        if agent[agent_id] then
                                local s, a, r, s2, term = agent[agent_id].valid_s, agent[agent_id].valid_a, agent[agent_id].valid_r,
                                    agent[agent_id].valid_s2, agent[agent_id].valid_term
                                agent[agent_id].valid_s, agent[agent_id].valid_a, agent[agent_id].valid_r, agent[agent_id].valid_s2,
                                    agent[agent_id].valid_term = nil, nil, nil, nil, nil, nil, nil
                                local w, dw, g, g2, delta, delta2, deltas, tmp = agent[agent_id].w, agent[agent_id].dw,
                                    agent[agent_id].g, agent[agent_id].g2, agent[agent_id].delta, agent[agent_id].delta2, agent[agent_id].deltas, agent[agent_id].tmp
                                agent[agent_id].w, agent[agent_id].dw, agent[agent_id].g, agent[agent_id].g2, agent[agent_id].delta, agent[agent_id].delta2,
                                    agent[agent_id].deltas, agent[agent_id].tmp = nil, nil, nil, nil, nil, nil, nil, nil

                                local filename = opt_of_agent[agent_id].name .. "_" .. agent_id
                                if (agent_net[agent_id] ~= 'NO_NETWORK') then
                                        filename = agent_net[agent_id]
                                end
                                if opt.save_versions > 0 then -- agent_net
                                        filename = filename .. "_" .. math.floor(step / opt.save_versions)
                                end
                                if not filename then
                                        filename = "net" .. agent_id
                                end
                                filename = filename
                                torch.save(filename .. ".t7", {agent = agent[agent_id],
                                                model = agent[agent_id].network,
                                                best_model = agent[agent_id].best_network,
                                                reward_history = reward_history,
                                                reward_counts = reward_counts,
                                                episode_counts = episode_counts,
                                                time_history = time_history,
                                                v_history = v_history,
                                                td_history = td_history,
                                                qmax_history = qmax_history,
                                                arguments=opt_of_agent[agent_id]})
                                if save_mid_nets and (step % save_mid_nets_steps == 0) then
                                  torch.save(filename .. "_step" .. step .. ".t7", {agent = agent[agent_id],
                                                  model = agent[agent_id].network,
                                                  best_model = agent[agent_id].best_network,
                                                  reward_history = reward_history,
                                                  reward_counts = reward_counts,
                                                  episode_counts = episode_counts,
                                                  time_history = time_history,
                                                  v_history = v_history,
                                                  td_history = td_history,
                                                  qmax_history = qmax_history,
                                                  arguments=opt_of_agent[agent_id]})
                                end
                                if opt.saveNetworkParams then
                                        local nets = {network=w:clone():float()}
                                        torch.save(filename..'.params.t7', nets, 'ascii')
                                end
                                agent[agent_id].valid_s, agent[agent_id].valid_a, agent[agent_id].valid_r, agent[agent_id].valid_s2,
                                    agent[agent_id].valid_term = s, a, r, s2, term
                                agent[agent_id].w, agent[agent_id].dw, agent[agent_id].g, agent[agent_id].g2, agent[agent_id].delta, agent[agent_id].delta2,
                                    agent[agent_id].deltas, agent[agent_id].tmp = w, dw, g, g2, delta, delta2, deltas, tmp
                                print('Saved:', filename .. '.t7')
                                io.flush()
                                collectgarbage()
                        end
                end
        end
        if stats_evl_visits_in_level[splitting_max_learned_level] > 0 and (agent_net[splitting_max_learned_level] ~= 'NO_NETWORK') and si_split_agent then
                --death_ratio_max_learned_level = stats_evl_terminals_in_level[splitting_max_learned_level]/stats_evl_visits_in_level[splitting_max_learned_level]
                --pass_percent_max_learned_level = stats_evl_visits_in_level[splitting_max_learned_level+1]/stats_evl_visits_in_level[splitting_max_learned_level]
                --if pass_percent_max_learned_level > 0.3 and death_ratio_max_learned_level < 2 then
                                --split_agent(splitting_max_learned_level,splitting_max_learned_level+1)
                                --print("Splitting from " .. splitting_max_learned_level)
                                --splitting_max_learned_level = splitting_max_learned_level + 1
                --end
        end


end
