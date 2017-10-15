--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require 'initenv'
end
require 'hdf5'
require 'si_clustering'

local nql = torch.class('dqn.NeuralQLearner')


function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start           = args.ep or 1
    self.ep                 = self.ep_start -- Exploration probability.
    self.ep_end             = args.ep_end or self.ep
    self.ep_endt            = args.ep_endt or 1000000
    self.safe_eps_n_lowest     = args.safe_eps_n_lowest or 3
    self.use_smart_eps         = args.use_smart_eps or false
    self.smart_eps_choice_type = args.smart_eps_choice_type or "safeeps_nloest"
    --self.smart_eps_choice_type can be:
    -- "softmax_linear"
    -- "softmax_pow7"
    -- "softmax_pow3"
    -- "softmax_exp1"
    -- "softmax_normalized_exp"
    -- "safeeps_nloest"

    ---- Activation memory
    self.save_activations               = false
    self.clustering_win_size            = 20
    self.knn_k                          = 3
    self:reset_activation_mem()
    self.saved_network = nil

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512

    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
    if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      print("Using cudnn")
    end
    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    if self.target_q then
        self.target_network = self.network:clone()
    end
    if pcall(require, 'cudnn') then
      require 'cudnn'
      cudnn.benchmark = true
      cudnn.convert(self.network, cudnn)
      print("Using cudnn")
    end

end

function nql:delete_activation_mem()
    self.activation_mem_size            = nil
    self.mem_full                       = nil
    self.activation_mem_activation_size = nil
    self.reward_mem                     = nil
    self.terminal_mem                   = nil
    self.level_mem                      = nil
    self.activation_clusters            = nil
    self.cluster_according_to_freezed   = nil
    self.activation_clusters_dict       = nil
    self.activation_l_mem               = nil
    self.last_activation                = nil
    self.last_activation_of_winsize     = nil
    self.last_activation_current_size   = nil
    self.last_activation_current_idx    = nil
    self.activation_mem_current_idx     = nil
    self.episode_idx_mem                = nil
    self.mu                             = nil
end

function nql:reset_activation_mem()
    self.activation_mem_size            = 120001 -- +1 because the reward of last record is invalid
    self.mem_full                       = false
    self.activation_mem_activation_size = 512
    self.reward_mem                     = torch.zeros(self.activation_mem_size)
    self.terminal_mem                   = torch.zeros(self.activation_mem_size)
    self.level_mem                      = torch.zeros(self.activation_mem_size)
    self.activation_clusters            = torch.zeros(self.activation_mem_size)
    self.cluster_according_to_freezed   = torch.zeros(self.activation_mem_size)
    self.mu                             = torch.zeros(5,self.activation_mem_activation_size)
    self.activation_clusters_dict       = {}
    self.activation_l_mem               = torch.FloatTensor(self.activation_mem_size,self.activation_mem_activation_size)
    self.last_activation                = nil
    self.last_activation_of_winsize     = torch.FloatTensor(self.clustering_win_size+1,self.activation_mem_activation_size)
    self.last_activation_current_idx    = 1
    self.last_activation_current_size   = 0
    self.activation_mem_current_idx     = 1
    self.episode_idx_mem                = torch.zeros(self.activation_mem_size)
end

function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end

function nql:save_network()
    self.saved_network = self.network:clone()
end

function nql:preprocess(rawstate)
    if self.preproc then
        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end

    return rawstate
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end

    -- Compute max_a Q(s_2, a).
    q2_max = target_q_net:forward(s2):float():max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    q2 = q2_max:clone():mul(self.discount):cmul(term)

    delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}

    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}

    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep, current_level, episode_idx, get_act_and_q_from_saved_network)
    -- Preprocess state (will be set to nil if terminal)
    assert(testing or not get_act_and_q_from_saved_network) -- Never teach saved network
    local state = self:preprocess(rawstate):float()
    local curState
    -- Add reward to previous step mem
    if self.save_activations and self.activation_mem_current_idx > 1 then
      self.reward_mem[(self.activation_mem_current_idx-1)]=reward
    end
    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    if not terminal and not self.save_activations then
        actionIndex = self:eGreedy(curState, testing_ep, get_act_and_q_from_saved_network)
    end
    if terminal and self.last_activation_of_winsize then
        self.last_activation_of_winsize     = torch.FloatTensor(self.clustering_win_size+1,self.activation_mem_activation_size)
        self.last_activation_current_idx    = 1
        self.last_activation_current_size   = 0
    end

    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch()
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end
    -- Store activation in memory
    if self.save_activations then
        local terminal_num_val
        if terminal then
          terminal_num_val = 1
        else
          terminal_num_val = 0
        end
        self.terminal_mem[self.activation_mem_current_idx]                   = terminal_num_val
        self.level_mem[self.activation_mem_current_idx]                      = current_level
        self.episode_idx_mem[self.activation_mem_current_idx]                = episode_idx
        self.activation_mem_current_idx = self.activation_mem_current_idx + 1
        if self.activation_mem_current_idx > self.activation_mem_size then
            self.mem_full = true
            self.activation_mem_current_idx = math.max(self.activation_mem_size,self.activation_mem_current_idx)
        end

    end
    if not testing then
        self.lastState = state:clone()
        self.lastAction = actionIndex
        self.lastTerminal = terminal
    end

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end

    if not terminal then
        return actionIndex
    else
        return 0
    end
end

function nql:get_safe_eps_act(state,q)
    local under_me = 0
    local randq
    while not (under_me >= self.safe_eps_n_lowest) do
        -- Get random action
        under_me = 0
        randq = torch.random(1, self.n_actions)
        -- Verify that the action is not among the self.safe_eps_n_lowest
        for a = 1, self.n_actions do
            if (q[a] <= q[randq]) and (a ~= randq) then -- Note that if q[a] == q[randq] we count a as smaller than randq
                under_me = under_me + 1
                if under_me >= self.safe_eps_n_lowest then break end
            end
        end
    end
    return randq
end

function nql:build_cdf(values)
  cdf = values:clone()
  -- Normalize
  cdf = cdf/(cdf:sum())
  update_idx = 1
  -- Summation
  while update_idx < cdf:size()[1] do
    cdf[update_idx+1] = cdf[update_idx+1] + cdf[update_idx]
    update_idx = update_idx + 1
  end
  return cdf
end

function nql:choose_from_cdf(cdf)
  selected = 1
  rand_res = torch.uniform()
  while selected < cdf:size()[1] do
    if rand_res < cdf[selected] then
      return selected
    end
    selected = selected + 1
  end
  return selected
end

function nql:softmax_linear(q_values)
  return self:choose_from_cdf(self:build_cdf(q_values))
end

function nql:softmax_pow(q_values,pow_val)
  return self:choose_from_cdf(self:build_cdf(torch.pow(q_values,pow_val)))
end

function nql:softmax_exp(q_values,tau)
  return self:choose_from_cdf(self:build_cdf(torch.exp(torch.div(q_values,tau))))
end


function nql:smart_random_act(state, q)
  choice_type = self.smart_eps_choice_type
  if choice_type == "softmax_linear" then
    return self:softmax_linear(q)
  elseif choice_type == "softmax_pow7" then
    return self:softmax_pow(q,7)
  elseif choice_type == "softmax_pow3" then
    return self:softmax_pow(q,3)
  elseif choice_type == "softmax_exp1" then
    return self:softmax_exp(q,1)
  elseif choice_type == "softmax_normalized_exp" then
    return self:softmax_exp(q,q:sum())
  elseif choice_type == "safeeps_nloest" then
    return self:get_safe_eps_act(state,q)
  end
  assert(false)
  return 1
end

function nql:save_data(eval_idx)
    --To save hdf5:
    --local myFile = hdf5.open('/path/to/write.h5', 'w')
    --myFile:write('/path/to/data', torch.rand(5, 5))
    --myFile:close()
    -- Source: https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md
    local myFile
    local myfile
      --local myFile

      myfile = hdf5.open('./data_h5/mu_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.mu)
      myfile:close()

      myfile = hdf5.open('./data_h5/activation_l_mem_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.activation_l_mem)
      myfile:close()

      myfile = hdf5.open('./data_h5/activation_clusters_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.activation_clusters)
      myfile:close()

      myfile = hdf5.open('./data_h5/prev_activation_l_mem_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.prev_activation_l_mem)
      myfile:close()

      myfile = hdf5.open('./data_h5/terminal_mem_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.terminal_mem)
      myfile:close()

      myfile = hdf5.open('./data_h5/level_mem_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.level_mem)
      myfile:close()

      myfile = hdf5.open('./data_h5/saved_network_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.saved_network)
      myfile:close()

      myfile = hdf5.open('./data_h5/activations_mean_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.activations_mean)
      myfile:close()

      myfile = hdf5.open('./data_h5/activations_std_eval' .. eval_idx .. '.h5', 'w')
      myfile:write('/path/to/data', self.activations_std)
      myfile:close()


      print("data saved")

end

function nql:start_saving_activations()
    self.save_activations         = true
    self.prev_activation_l_mem    = self.activation_l_mem:clone()
    self.prev_activation_clusters = self.activation_clusters:clone()
    self.prev_mu                  = self.mu:clone()
    self:reset_activation_mem()
end

function nql:stop_saving_activations()
    self.save_activations         = false
    self.prev_activation_l_mem    = 0
    self.prev_mu                  = 0
    self.prev_activation_clusters = 0
end

function nql:cluster_activations(num_of_clusters,eval_idx)
    print(os.date("Called cluster_activations at %H:%M"))
    -- If previous clustering exists, tag each activation according to previous clustering
    self.activations_mean, self.activations_std, self.activation_l_mem = normalize_batch(self.activation_l_mem)
    num_of_activations = (#self.activation_l_mem)[1]
    activations_tagging_according_prev = torch.zeros(num_of_activations,1)
    if self.saved_network then
        for activation_ind = 1, num_of_activations do
            activations_tagging_according_prev[activation_ind] = self.cluster_according_to_freezed[activation_ind] --KNN(self.prev_activation_l_mem, self.prev_activation_clusters, self.activation_l_mem[activation_ind], self.knn_k)
        end
    end
    for cl_ind = 0, num_of_clusters do
        print("activations_tagging_according_prev:eq(" .. cl_ind .. "):sum()=" .. (activations_tagging_according_prev:eq(cl_ind):sum()))
    end
    --print("activations_tagging_according_prev")
    --print(activations_tagging_according_prev)
    -- Cluster self.activation_l_mem
    print("Calling find_centers with num_of_clusters=" .. num_of_clusters .. " , self.clustering_win_size=" .. self.clustering_win_size)
    new_mu, clusters, labels = find_centers(self.activation_l_mem, num_of_clusters, self.clustering_win_size, self.terminal_mem)
    num_of_prev_clusters          = num_of_clusters -- torch.max(activations_tagging_according_prev)
    prev_to_new_tags_matrix       = torch.zeros(num_of_clusters,num_of_prev_clusters) -- prev_to_new_tags_matrix[new_cluster_indx][prev_cluster_indx] how many of the vectors in new_cluster_indx tagged as prev_cluster_indx in the previous clustering
    new_clusters_majority_tagging = torch.zeros(num_of_clusters,1)
    if self.saved_network then
        -- Tag new clusters according to majority of previous activations tags
        --    Return also the amount of activations from each cluster, and then if two clusters has the same tag, create a new agent (from the previous one) for the tag with less activations from that cluster
        for new_cluster_ind = 1, num_of_clusters do
            current_cluster_majority_tag = new_cluster_ind
            current_cluster_majority_amount = 0
            for prev_cluster_ind = 1, num_of_prev_clusters do
                prev_to_new_tags_matrix[new_cluster_ind][prev_cluster_ind] = torch.cmul(labels:eq(new_cluster_ind),activations_tagging_according_prev:eq(prev_cluster_ind)):sum()
                if prev_to_new_tags_matrix[new_cluster_ind][prev_cluster_ind] > current_cluster_majority_amount then
                    current_cluster_majority_amount = prev_to_new_tags_matrix[new_cluster_ind][prev_cluster_ind]
                    current_cluster_majority_tag    = prev_cluster_ind
                end
            end
            -- For each cluster, decide cluster's index according to majority of previous tagging
            new_clusters_majority_tagging[new_cluster_ind] = current_cluster_majority_tag
        end
        print("After prev tagging (1) , num_of_clusters=" .. num_of_clusters)
        print("prev_to_new_tags_matrix")
        print(prev_to_new_tags_matrix)
        print("new_clusters_majority_tagging")
        print(new_clusters_majority_tagging)
        -- In case two clusters got the same previous tagging, tag the one with less activations of that previous tagging with a new tag
        for prev_cluster_ind = 1, num_of_prev_clusters do
            if new_clusters_majority_tagging:eq(prev_cluster_ind):sum() > 1 then
                print("new_clusters_majority_tagging:eq(" .. prev_cluster_ind .. "):sum()=" .. new_clusters_majority_tagging:eq(prev_cluster_ind):sum())
                -- Find the one with most activations of that previous tagging with a new tag
                most_prev_activations = 0
                most_prev_activations_idx = 1
                for new_cluster_ind = 1, num_of_clusters do
                    if new_clusters_majority_tagging[new_cluster_ind][1] == prev_cluster_ind and prev_to_new_tags_matrix[new_cluster_ind][prev_cluster_ind] > most_prev_activations then
                        most_prev_activations = prev_to_new_tags_matrix[new_cluster_ind][prev_cluster_ind]
                        most_prev_activations_idx = new_cluster_ind
                    end
                end
                -- For all lowest, assign an unusable cluster
                for new_cluster_ind = 1, num_of_clusters do
                    print("new_cluster_ind=" .. new_cluster_ind .. " , new_clusters_majority_tagging[new_cluster_ind][1]=" .. new_clusters_majority_tagging[new_cluster_ind][1] .. " , prev_cluster_ind=" .. prev_cluster_ind .. " , most_prev_activations_idx=" .. most_prev_activations_idx)
                    if new_clusters_majority_tagging[new_cluster_ind][1] == prev_cluster_ind and new_cluster_ind ~= most_prev_activations_idx then
                        got_new_tag = false
                        for assign_prev_cluster_ind = 1, num_of_prev_clusters do
                            if not got_new_tag and new_clusters_majority_tagging:eq(assign_prev_cluster_ind):sum() == 0 then
                                new_clusters_majority_tagging[new_cluster_ind] = assign_prev_cluster_ind
                                print("new_clusters_majority_tagging[" .. new_cluster_ind .. "] = " .. assign_prev_cluster_ind)
                                got_new_tag = true
                            end
                        end
                        assert(got_new_tag)
                    end
                end

            end
        end
        print("After prev tagging (2)")
        print("prev_to_new_tags_matrix")
        print(prev_to_new_tags_matrix)
        print("new_clusters_majority_tagging")
        print(new_clusters_majority_tagging)
        -- Rename clusters tag to match previous ones
        for prev_cluster_ind = 1, num_of_prev_clusters do
            assert(not (new_clusters_majority_tagging:eq(prev_cluster_ind):sum() > 1))
        end
    else
        for new_cluster_ind = 1, num_of_clusters do
            new_clusters_majority_tagging[new_cluster_ind] = new_cluster_ind
        end
    end
    --print("labels:")
    --print(labels)
    --print("self.activation_clusters")
    --print(self.activation_clusters)
    -- Save tags (to be used in KNN)
    sorted_new_mu = new_mu:clone()
    for new_cluster_ind = 1, num_of_clusters do
        --print("cluster ind")
        --print(new_cluster_ind)
        --print("labels eq")
        --print(labels:eq(new_cluster_ind))
        --print("new_clusters_majority_tagging")
        --print(new_clusters_majority_tagging)
        --print("new_clusters_majority_tagging[new_cluster_ind]")
        --print(new_clusters_majority_tagging[new_cluster_ind])


        self.activation_clusters[labels:eq(new_cluster_ind)] = new_clusters_majority_tagging[new_cluster_ind][1]
        sorted_new_mu[new_clusters_majority_tagging[new_cluster_ind][1]] = new_mu[new_cluster_ind]:clone()
    end
    self.mu = sorted_new_mu:clone()
    -- Calc REAL correct tagging (according to the real level)
    num_of_levels_in_mem          = torch.max(self.level_mem)
    level_to_cluster_matrix       = torch.zeros(num_of_clusters,num_of_levels_in_mem)
    new_clusters_majority_level_tagging = torch.zeros(num_of_clusters,1)
    for new_cluster_ind = 1, num_of_clusters do
        current_cluster_majority_tag = 1
        current_cluster_majority_amount = 0
        for level_ind = 1, num_of_levels_in_mem do
            level_to_cluster_matrix[new_cluster_ind][level_ind] = torch.cmul(labels:eq(new_cluster_ind),self.level_mem:eq(level_ind)):sum()
            if level_to_cluster_matrix[new_cluster_ind][level_ind] > current_cluster_majority_amount then
                current_cluster_majority_amount = level_to_cluster_matrix[new_cluster_ind][level_ind]
                current_cluster_majority_tag    = level_ind
            end
        end
        -- For each cluster, decide cluster's index according to majority of previous tagging
        new_clusters_majority_level_tagging[new_cluster_ind] = current_cluster_majority_tag
    end
    self:save_data(eval_idx)
    print("level_to_cluster_matrix")
    print(level_to_cluster_matrix)
    print("new_clusters_majority_level_tagging")
    print(new_clusters_majority_level_tagging)
    print(os.date("Finsihed cluster_activations at %H:%M"))
end


function nql:get_cluster_of_last_state()
    -- Run KNN of self.last_activation on self.activation_l_mem ()
    -- Cluster should be from 1 to N
    --return KNN(self.activation_l_mem, self.activation_clusters, self.last_activation, self.knn_k)
--    return classify_single_vec(self.last_activation, self.mu)
    if self.last_activation_current_size < 1 then
        return 1
    end
    return classify_batch_vecs(self.last_activation_of_winsize[{{1,self.last_activation_current_size},{}}], self.mu)
end

function nql:get_cluster_by_freezed(state)
    -- this is not spatio temproal clustering
    if not self.saved_network then
        return 1
    end
    local net_to_use = self.saved_network
    local q = net_to_use:forward(state):float():squeeze()
    local nodes = net_to_use:findModules('nn.Linear')
    local activation = nodes[1].output:clone():float() -- Size: 1x512
    self.prev_activation_l_mem[self.activation_mem_current_idx] = activation:clone()
    return classify_single_vec(activation, self.mu)
end

function nql:eGreedy(state, testing_ep, get_act_and_q_from_saved_network)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local net_to_use = self.network
    if get_act_and_q_from_saved_network then
        net_to_use = self.saved_network
        if not self.saved_network then
            net_to_use = self.network
            print("ERROR!!!!!!! No saved network!")
        end
    end

    local q = net_to_use:forward(state):float():squeeze()

    local nodes = net_to_use:findModules('nn.Linear')
    local activation = nodes[1].output:clone():float() -- Size: 1x512
    if self.last_activation_of_winsize and self.saved_network then
        local q_freezed = self.saved_network:forward(state):float():squeeze()
        local nodes_freezed = self.saved_network:findModules('nn.Linear')
        local activation_freezed = nodes[1].output:clone():float() -- Size: 1x512

        self.last_activation_of_winsize[self.last_activation_current_idx] = activation_freezed:clone()
        if self.activations_mean then
            self.last_activation_of_winsize[self.last_activation_current_idx] = self.last_activation_of_winsize[self.last_activation_current_idx] - self.activations_mean
            self.last_activation_of_winsize[self.last_activation_current_idx] = torch.cdiv(self.last_activation_of_winsize[self.last_activation_current_idx],self.activations_std)
        end

        self.last_activation_current_size = self.last_activation_current_size + 1
        self.last_activation_current_idx = self.last_activation_current_idx + 1
        if self.last_activation_current_idx > (#self.last_activation_of_winsize)[1] then
            self.last_activation_current_idx = 1
        end
        if self.last_activation_current_size > (#self.last_activation_of_winsize)[1] then
            self.last_activation_current_size = (#self.last_activation_of_winsize)[1]
        end

--        self.last_activation_of_winsize     = torch.FloatTensor(self.clustering_win_size+1,self.activation_mem_activation_size)
  --      self.last_activation_current_idx    = 1
    --    self.last_activation_current_size   = 0

    end

    self.last_activation = activation:clone()
    if self.activations_mean then
        self.last_activation = self.last_activation - self.activations_mean
        self.last_activation = torch.cdiv(self.last_activation,self.activations_std)
    end

    -- Save activations
    if self.save_activations then
        self.activation_l_mem[self.activation_mem_current_idx] = activation:clone()
        -- Classify state with freezed network
        if self.saved_network then
            self.cluster_according_to_freezed[self.activation_mem_current_idx] = self.get_cluster_of_last_state()  -- WAS: self:get_cluster_by_freezed(state)
        end
    end
    -- Epsilon greedy
    if torch.uniform() < self.ep then
        if not self.use_smart_eps then
            return torch.random(1, self.n_actions)
        else
            return self:smart_random_act(state,q)
        end
    else
        return self:greedy(state,q)
    end
end


function nql:greedy(state,q)
    local maxq = q[1]
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        if q[a] > maxq then
            besta = { a }
            maxq = q[a]
        elseif q[a] == maxq then
            besta[#besta+1] = a
        end
    end
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    print(get_weight_norms(self.network))
    print(get_grad_norms(self.network))
end
