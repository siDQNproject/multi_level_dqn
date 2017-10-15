--[[
Copyright (c) 2017 SI.
See LICENSE file for full terms of limited license.
]]

require 'torch'
require 'math'

-- Source: http://lua-users.org/wiki/RandomSample
function permute(tab, n, count)
  n = n or #tab
  for i = 1, count or n do
    local j = torch.random(i, n)
    tab[i], tab[j] = tab[j], tab[i]
  end
  return tab
end



do
  local meta = {}
  function meta:__index(k) return k end
  function PositiveIntegers() return setmetatable({}, meta) end
end

function get_rand_indices(count, range)
  return {unpack(
               permute(PositiveIntegers(), range, count),
               1, count)
            }
end

-- spatio-temporal KMeans based on https://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/

function dist_between_two_vectors(vec1, vec2)
    return torch.sqrt(torch.sum(torch.pow(vec1-vec2,2)))
end

function compute_all_dists(X, mu, dists)
    num_of_samples_in_x = (#X)[1]
    K = (#mu)[1]
    for samp_ind = 1, num_of_samples_in_x do
        for mu_ind = 1, K do
            dists[samp_ind][mu_ind] = dist_between_two_vectors(X[samp_ind], mu[mu_ind]) -- torch.sqrt(torch.sum(torch.pow(X[samp_ind]-mu[mu_ind],2)))
        end
--        print(os.date("%H:%M:%S") .. "Computing dist of " .. samp_ind)
    end
    return true
end

function compute_dist(X, ind, dists, mukey, window_size, terminals)
    num_of_samples_in_x = (#X)[1]
    sum_of_dist = 0
    for neighbour_ind = ind, math.max(1,ind-window_size), -1 do
        --sum_of_dist = sum_of_dist + torch.sqrt(torch.sum(torch.pow(X[neighbour_ind]-mu,2)))
        if neighbour_ind ~= ind and terminals[neighbour_ind] == 1 then
            break
        end
        sum_of_dist = sum_of_dist + dists[neighbour_ind][mukey]
    end

    for neighbour_ind = ind+1, math.min(num_of_samples_in_x,ind+window_size) do
        --sum_of_dist = sum_of_dist + torch.sqrt(torch.sum(torch.pow(X[neighbour_ind]-mu,2)))
        if terminals[ind] == 1 or terminals[neighbour_ind] == 1 then
            break
        end
        sum_of_dist = sum_of_dist + dists[neighbour_ind][mukey]
    end
    return sum_of_dist
end

function classify_batch_vecs(vecs, mu)
    K = (#mu)[1]
    dim = (#mu)[2]
    num_of_vecs = (#vecs)[1]
    if num_of_vecs < 1 then
        return 1
    end
    bestmukey = 1
    best_size = math.huge
    for mukey = 1, K do
        current_dist = 0
        for vecind = 1, num_of_vecs do
            current_sample = torch.reshape(vecs[vecind],1,dim)
            current_dist = current_dist + dist_between_two_vectors(current_sample, mu[mukey])
        end
        if current_dist < best_size then
            bestmukey = mukey
            best_size = current_dist
        end
    end
    return bestmukey
end

function classify_single_vec(vec, mu)
    K = (#mu)[1]
    dim = (#mu)[2]
    bestmukey = 1
    best_size = math.huge
    current_sample = torch.reshape(vec,1,dim)
    for mukey = 1, K do
        current_dist = dist_between_two_vectors(current_sample, mu[mukey])
        if current_dist < best_size then
            bestmukey = mukey
            best_size = current_dist
        end
    end
    return bestmukey
end

function cluster_points(X, mu, window_size, terminals)
    num_of_samples_in_x = (#X)[1]
    dim = (#X)[2]
    K = (#mu)[1]
    clusters = {}
    labels   = torch.zeros(num_of_samples_in_x)
    dists    = torch.zeros(num_of_samples_in_x,K)
    compute_all_dists(X, mu, dists)

    start_time = os.clock()

    for ind = 1, num_of_samples_in_x do
        bestmukey = 1
        best_size = math.huge
        current_sample = torch.reshape(X[ind],1,dim)
        --print(os.date("%H:%M:%S") .. "Running sample " .. ind)
        if ind==10000 then
            print(string.format("elapsed time: %.2f\n", os.clock() - start_time))
        end
        for mukey = 1, K do
            current_dist = compute_dist(X, ind, dists, mukey, window_size, terminals)
            if current_dist < best_size then
                bestmukey = mukey
                best_size = current_dist
            end
        end
        labels[ind] = bestmukey
        if clusters[bestmukey] then
            clusters[bestmukey] = torch.cat(clusters[bestmukey]:clone(),current_sample:clone(),1)
        else
            clusters[bestmukey] = current_sample:clone()
        end
        --table.insert(clusters[bestmukey],current_sample:clone())
    end
    return clusters,labels
end

num_of_iter = 1
function has_converged(mu, oldmu)
    print(os.date("%H:%M:%S") .. " checking has converged")
    converged = true
    K = (#mu)[1]
    for mukey = 1, K do
        current_mu_exists_in_old = false
        for oldmukey = 1, K do
            if torch.all(mu[mukey]:eq(oldmu[oldmukey])) then
                current_mu_exists_in_old = true
            end
        end
        if not current_mu_exists_in_old then
            converged = false
            break
        end
    end
    print(os.date("%H:%M"))
    print("iter " .. tostring(num_of_iter))
    num_of_iter = num_of_iter + 1
    return converged or (num_of_iter > 1000)
    
end

function reevaluate_centers(mu, clusters)
    new_mu = mu:clone()
    for key, val in pairs(clusters) do
--        print(os.date("%H:%M:%S") .. "Running key " .. key)
        new_mu[key] = torch.mean(clusters[key],1)
    end
    return new_mu
end

function find_centers(X, K, window_size, terminals)
    num_of_samples_in_x = (#X)[1]
    rand_indices1 = get_rand_indices(K, num_of_samples_in_x)
    rand_indices2 = get_rand_indices(K, num_of_samples_in_x)
    oldmu = X[{{1,K}}]:clone()
    mu = X[{{1,K}}]:clone()
    for ii = 1, K do
        oldmu[ii] = X[rand_indices1[ii]]
        mu[ii] = X[rand_indices2[ii]]
    end
    while not has_converged(mu, oldmu) do
        oldmu = mu:clone()
        -- Assign all points in X to clusters
        clusters, labels = cluster_points(X, mu, window_size, terminals)
        -- Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    end
    return mu, clusters, labels
end

function normalize_batch(X)
    mean = torch.mean(X,1)
    std  = torch.std(X,1)
    --print("Normalizing mean=" .. tostring(mean) .. " , std=" .. tostring(std))
    X = (X-(mean:expandAs(X)))
    X = torch.cdiv(X,(std:expandAs(X)))
    return mean, std, X
end


------ KNN
function KNN(activation_l_mem, tags, vec_to_tag, K)
    tags_max_val = torch.max(tags)
    num_of_samples_in_act_mem = (#activation_l_mem)[1]
    dists_knn                 = torch.zeros(num_of_samples_in_act_mem,1)
    knns_dists                = torch.zeros(K,1)
    knns_tags                 = torch.zeros(K,1)
    for samp_ind = 1, K do
        knns_dists[samp_ind] = dist_between_two_vectors(activation_l_mem[samp_ind], vec_to_tag)
        knns_tags[samp_ind]  = tags[samp_ind]
    end
    knns_min_val, knns_min_ind = torch.min(knns_dists,1) -- NOTE: We just take the 1st lowest value, no special treatment in case we have multiple instances of the minimum value
    knns_min_val = knns_min_val[1][1]
    knns_min_ind = knns_min_ind[1][1]
    for samp_ind = (K+1), num_of_samples_in_act_mem do
        current_dist            = dist_between_two_vectors(activation_l_mem[samp_ind], vec_to_tag)
        if current_dist < knns_min_val then
            knns_dists[knns_min_ind] = current_dist
            knns_tags[knns_min_ind]  = tags[samp_ind]
            knns_min_val, knns_min_ind = torch.min(knns_dists,1) -- NOTE: We just take the 1st lowest value, no special treatment in case we have multiple instances of the minimum value
            knns_min_val = knns_min_val[1][1]
            knns_min_ind = knns_min_ind[1][1]
        end
    end
    -- Create tags histogram
    knns_tags_histogram              = torch.zeros(tags_max_val,1)
    for tag_val = 1, tags_max_val do
        knns_tags_histogram[tag_val] = knns_tags:eq(tag_val):sum()
    end
    -- Find max tags
    knns_max_tags     = {}
    knns_max_tags_len = 0
    knns_max_value    = 0

    for tag_val = 1, tags_max_val do
        if knns_tags_histogram[tag_val][1] == knns_max_value then
            knns_max_tags[knns_max_tags_len+1] = tag_val
            knns_max_tags_len = knns_max_tags_len + 1
        elseif knns_tags_histogram[tag_val][1] > knns_max_value then
            knns_max_tags     = {}
            knns_max_tags[1]  = tag_val
            knns_max_tags_len = 1
            knns_max_value    = knns_tags_histogram[tag_val][1]
        end
    end
    return knns_max_tags[torch.random(1,knns_max_tags_len)]
end
