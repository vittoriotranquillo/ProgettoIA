require 'nn'
require 'image'
term = require 'term'
require 'optim'
colors = term.colors
local ParamBank = require 'ParamBank'
local label     = require 'overfeat_label'
torch.setdefaulttensortype('torch.FloatTensor')

local opt = lapp([[
Parameters
   --network        (default 'small')      network size ( small | big )
   --img            (default 'bee.jpg')    test image
   --backend        (default 'nn')         specify backend ( nn | cunn | cudnn )
   --inplace                               use inplace ReLU
   --spatial                               use spatial mode (detection/localization)
   --save           (default 'model.t7')   save model path
   --threads        (default 4)            nb of threads
]])
torch.setnumthreads(opt.threads)

if opt.backend == 'nn' or opt.backend == 'cunn' then
   require(opt.backend)
   SpatialConvolution = nn.SpatialConvolutionMM
   SpatialMaxPooling = nn.SpatialMaxPooling
   ReLU = nn.ReLU
   SpatialSoftMax = nn.SpatialSoftMax
else
   assert(false, 'Unknown backend type')
end

local net = nn.Sequential()
if opt.network == 'small' then
   --print('==> init a small overfeat network')
   net:add(SpatialConvolution(3, 96, 11, 11, 4, 4))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(96, 256, 5, 5, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(512, 1024, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(1024, 1024, 3, 3, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialMaxPooling(2, 2, 2, 2))
   net:add(SpatialConvolution(1024, 3072, 6, 6, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(3072, 4096, 1, 1, 1, 1))
   net:add(ReLU(opt.inplace))
   net:add(SpatialConvolution(4096, 1000, 1, 1, 1, 1))
   if not opt.spatial then net:add(nn.View(1000)) end
   net:add(SpatialSoftMax())

   --print('==> overwrite network parameters with pre-trained weigts')
   ParamBank:init("net_weight_0")
   ParamBank:read(        0, {96,3,11,11},    net:get(1).weight)
   ParamBank:read(    34848, {96},            net:get(1).bias)
   ParamBank:read(    34944, {256,96,5,5},    net:get(4).weight)
   ParamBank:read(   649344, {256},           net:get(4).bias)
   ParamBank:read(   649600, {512,256,3,3},   net:get(7).weight)
   ParamBank:read(  1829248, {512},           net:get(7).bias)
   ParamBank:read(  1829760, {1024,512,3,3},  net:get(9).weight)
   ParamBank:read(  6548352, {1024},          net:get(9).bias)
   ParamBank:read(  6549376, {1024,1024,3,3}, net:get(11).weight)
   ParamBank:read( 15986560, {1024},          net:get(11).bias)
   ParamBank:read( 15987584, {3072,1024,6,6}, net:get(14).weight)
   ParamBank:read(129233792, {3072},          net:get(14).bias)
   ParamBank:read(129236864, {4096,3072,1,1}, net:get(16).weight)
   ParamBank:read(141819776, {4096},          net:get(16).bias)
   ParamBank:read(141823872, {1000,4096,1,1}, net:get(18).weight)
   ParamBank:read(145919872, {1000},          net:get(18).bias)
else
   assert(false, 'Unknown network type')
end

ParamBank:close()

globalLabels = {}
function extractMyLabels(tipoPath)
	myPaths = {}
	tmpLabels = {}
	tmpLabelsDecimal = {}
	if(tipoPath == "test") then
		myPaths = testPaths
		tmpLabels = testLabels
		tmpLabelsDecimal = testLabelsDecimal
	elseif(tipoPath == "training") then
		myPaths = trainingPaths
		tmpLabels = trainingLabels
		tmpLabelsDecimal = trainingLabelsDecimal
	end

	
	for k, v in ipairs(myPaths) do
		indexes = {}
		local p = string.find(v, "/", 1)
		lastIndex = p

		while p do
			p = string.find(v, "/", p+1)
			table.insert(indexes, p)
			if p then
				lastIndex = p
			end
		end

		start = indexes[table.getn(indexes)-2]
		finish = indexes[table.getn(indexes)-1]
		currentLabel = string.sub(v, start+1, finish-1)
		table.insert(tmpLabels, currentLabel)
		table.insert(tmpLabelsDecimal, k)
		table.insert(globalLabels, currentLabel)
	end
end

function load(file, tipoDirectory)
	if(tipoDirectory == "test") then
		testPaths = {}
		testLabels = {}
		testLabelsDecimal = {}
		for line in io.lines(file) do 
			table.insert(testPaths, line)
		end
		extractMyLabels("test")
	elseif(tipoDirectory == "training") then
		trainingPaths = {}
		trainingLabels = {}
		trainingLabelsDecimal = {}
		for line in io.lines(file) do 
			table.insert(trainingPaths, line)
		end
		extractMyLabels("training")
	end
	return lines
end

local testLines = load("/home/giuseppe/Documenti/Università/Materie/IA/OverFeat/Path/testPaths (manuale).txt", "test")
local trainingLines = load("/home/giuseppe/Documenti/Università/Materie/IA/OverFeat/Path/trainingPaths (manuale).txt", "training")

function contains(tab, val)
	for index, value in ipairs(tab) do
		if value == val then return true end
	end
	return false
end

print("\nTabella indici/label:")
for k, v in ipairs(globalLabels) do
	print(colors.magenta(k) .. " " .. colors.cyan(v))
end

tabellaTensoriTest = {}
tabellaTensoriTraining = {}
tabellaTensori = {}
print("\nCarico le immagini in tabellaTensori... \n")

function loadTensors(myPaths, tabellaTensori)
	for k, v in ipairs(myPaths) do
		l = image.load(v, 3, 'float'):mul(255)
		table.insert(tabellaTensori, l)
	end
end

loadTensors(testPaths, tabellaTensoriTest)
loadTensors(trainingPaths, tabellaTensoriTraining)

function scaleTensors(tabellaTensori)
	for k, v in ipairs(tabellaTensori) do
		if not opt.spatial then
			local dim = (opt.network == 'small') and 231 or 221
			local img_scale = image.scale(tabellaTensori[k], '^' .. dim)
			local h = math.ceil((img_scale:size(2) - dim)/2)
			local w = math.ceil((img_scale:size(3) - dim)/2)
			tabellaTensori[k] = image.crop(img_scale, w, h, w + dim, h + dim):floor()
		end
	end
end

scaleTensors(tabellaTensoriTest)
scaleTensors(tabellaTensoriTraining)

net:remove(20)
net:remove(19)
net:remove(18)

tabellaFeatureTest = {}
tabellaFeatureTraining = {}

function initNewNetwork()
	net2 = nn.Sequential()
	net2:add(nn.Linear(4096, table.getn(trainingLabels))) -- Forse!
	net2:add(nn.LogSoftMax())
	w, dL_dw = net2:getParameters()
	crit = nn.ClassNLLCriterion()
	state = {learningRate = 0.01}
end

function feval()
	i = (i or 0) + 1
	if i > table.getn(tabellaFeatureTraining) then i = 1 end
	dL_dw:zero()
	local y = net2:forward(tabellaFeatureTraining[i])
	local L = crit:forward(y, trainingLabelsDecimal[i])
	local dL_dy = crit:backward(y, trainingLabelsDecimal[i])
	local dL_dx = net2:backward(tabellaFeatureTraining[i], dL_dy)
	return L, dL_dw
end

function executeSGD()
	n = table.getn(tabellaFeatureTraining)
	for e = 1, 100 do
		avg_L = 0
		for i = 1, n do
			_, L = optim.sgd(feval, w, state)
			avg_L = avg_L + L[1]
		end
		avg_L = avg_L/n -- Calcoliamo l'errore medio
	end
	print("\nErrore medio sul training set: " .. colors.cyan(avg_L))
end

timer = torch.Timer()
print("Classificazione OverFeat (su 17 layer):")
function classify(tabellaTensori, tabellaFeature)
	for k, v in ipairs(tabellaTensori, tabellaFeature) do
		tabellaTensori[k]:add(-118.380948):div(61.896913)
		table.insert(tabellaFeature, net:forward(tabellaTensori[k]):squeeze():clone())
		prob, idx = torch.max(tabellaFeature[k], 1)
		temp = label[idx:squeeze()]
		temp2 = prob:squeeze()
		print(colors.red(k) .. " " .. colors.green(temp) .. " " .. colors.yellow(temp2))
	end
end

classify(tabellaTensoriTest, tabellaFeatureTest)
classify(tabellaTensoriTraining, tabellaFeatureTraining)

initNewNetwork()
executeSGD()

output = {}
accuracy = 0

function accuracyTesting()
	print("\nClassificazione Softmax (su 2 layer):")
	for k, v in ipairs(tabellaFeatureTest) do
		table.insert(output, net2:forward(tabellaFeatureTest[k]):squeeze():clone())
		local _, predict = torch.max(output[k], 1)
		predict = predict[1]
		y_des = testLabelsDecimal[k]
		if predict == y_des then accuracy = accuracy + 1 end
		print(colors.green("Feature[" .. k .. "]: ") .. colors.magenta("label predetta " .. predict .. ", label corretta " .. y_des .. "."))
	end
end

accuracyTesting()
accuracy = accuracy/table.getn(tabellaFeatureTest)
print(colors.yellow("Accuratezza: ") .. colors.cyan(accuracy))

function round(num, n)
	local mult = 10^(n or 0)
	return math.floor(num*mult+0.5)/mult
end

print('\nTempo trascorso: ' .. round(timer:time().real, 2) .. ' secondi.\n')
