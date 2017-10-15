require 'hdf5'
--require 'unsup'
--m = require 'manifold'
bh_tsne = require('tsne')

local cmd = torch.CmdLine()
cmd:text()
cmd:option('-N', '', 'number of states')
cmd:text()
cmd:option('-iter', '', 'number of iterations')
cmd:option('-convert', false, 'convert t7 to hdf5')
local args = cmd:parse(arg)
local N = tonumber(args.N)
local maxiter  = 1000 --tonumber(args.iter)

local pca_dims = 50

local convert_t7_to_hdf5 = false --cmd:parse(args.convert)
local myFile
local data
if convert_t7_to_hdf5 then
	myFile   = torch.DiskFile('./dqn_distill/hdrln_activations.t7', 'r')
	data     = myFile:readObject()--myFile:read('data'):all()
else
	myFile = hdf5.open('./activation_mem.h5', 'r')
	data = myFile:read('path/to/data'):all()
end
myFile:close()

print('Data loaded ' .. data:size(1) .. ' ' .. data:size(2))
data = data[{{1,120000}, {1,512}}]
print('Data loaded ' .. data:size(1) .. ' ' .. data:size(2))
--os.exit()

if convert_t7_to_hdf5 then
	data  = data:narrow(1, 1, N):double()
	local myFile2= hdf5.open('./dqn_distill/activations.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved activations')

	local myFile   = torch.DiskFile('./dqn_distill/hdrln_actions.t7', 'r') --hdf5.open('./dqn/tmp/global_activations.h5', 'r')
	local data     = myFile:readObject()--myFile:read('data'):all()
	myFile:close()
	data  = data:narrow(1, 1, N)
	local myFile2= hdf5.open('./dqn_distill/actions.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved actions')

	local myFile   = torch.DiskFile('./dqn_distill/hdrln_fullstates.t7', 'r') --hdf5.open('./dqn/tmp/global_activations.h5', 'r')
	local data     = myFile:readObject()--myFile:read('data'):all()
	myFile:close()
	data  = data:narrow(1, 1, N)
	local myFile2= hdf5.open('./dqn_distill/screens.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved screens')

	local myFile   = torch.DiskFile('./dqn_distill/hdrln_qvals.t7', 'r') --hdf5.open('./dqn/tmp/global_activations.h5', 'r')
	local data     = myFile:readObject()--myFile:read('data'):all()
	myFile:close()
	data  = data:narrow(1, 1, N)
	local myFile2= hdf5.open('./dqn_distill/qvals.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved qvals')

	local myFile   = torch.DiskFile('./dqn_distill/hdrln_rewards.t7', 'r') --hdf5.open('./dqn/tmp/global_activations.h5', 'r')
	local data     = myFile:readObject()--myFile:read('data'):all()
	myFile:close()
	data  = data:narrow(1, 1, N)
	local myFile2= hdf5.open('./dqn_distill/reward.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved reward')

	local myFile   = torch.DiskFile('./dqn_distill/hdrln_statespace.t7', 'r') --hdf5.open('./dqn/tmp/global_activations.h5', 'r')
	local data     = myFile:readObject()--myFile:read('data'):all()
	myFile:close()
	data  = data:narrow(1, 1, N)
	local myFile2= hdf5.open('./dqn_distill/states.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved states')

	local myFile   = torch.DiskFile('./dqn_distill/hdrln_terminal.t7', 'r') --hdf5.open('./dqn/tmp/global_activations.h5', 'r')
	local data     = myFile:readObject()--myFile:read('data'):all()
	myFile:close()
	data  = data:narrow(1, 1, N)
	local myFile2= hdf5.open('./dqn_distill/termination.h5', 'w')
	myFile2:write('data', data)
	myFile2:close()
	print('saved termination')
else
	data = data:double()
	local p = 512
	-- perform pca on the transpose
	local mean = torch.mean(data,1)
	local xm = data - torch.ger(torch.ones(data:size(1)),mean:squeeze())
	local c = torch.mm(xm:t(),xm)
	c:div(data:size(1)-1)
	local ce,cv = torch.symeig(c,'V')

	cv = cv:narrow(2,p-pca_dims+1,pca_dims)
	data = torch.mm(data, cv)

	--
	--opts = {ndims = 2, perplexity = 50,pca = 50, use_bh = true, theta = 0.5}
--	for perp = 750,850,100
--	for perp = 750
--	do
	perp = 750
		opts = {ndims = 2, perplexity = perp, use_bh = true, pca=pca_dims, theta = 0.5, n_iter=maxiter, max_iter = maxiter, method='barnes_hut'}
		print('run t-SNE')
		--mapped_activations = m.embedding.tsne(data:double(), opts)
		mapped_activations = bh_tsne(data:double(), opts)
		print('save t-SNE')
		local myFile2= hdf5.open('./dqn_distill/lowd_activations_'..perp..'.h5', 'w')
		myFile2:write('data', mapped_activations)
		myFile2:close()
		print('saved t-SNE')
--	end
end
