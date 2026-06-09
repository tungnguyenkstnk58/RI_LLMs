function trainning_multithead_templated_english()
% This function aims to trains matrices W_q, W_k, W_v used in large language models - attention mechanism
% The data set is synthesized from English words (subjects, objects, and verbs)
% ------------------------------------------------------------
%
% The script:
%   1. Generates a synthetic templated English corpus.
%   2. Builds a word-level vocabulary and next-token prediction dataset.
%   3. Trains a small multi-head attention model using Adam.
%   4. Evaluates training and validation accuracy/loss.
%   5. Saves the trained matrices and data for later metric evaluation.
%
% Required toolbox:
%   MATLAB Deep Learning Toolbox.

% Output file:
%   trained_multihead_english_attention.mat
%
% Saved variables:
%   model, cfg, history, dataTrain, dataVal, vocab, corpus
% ------------------------------------------------------------
clear; %#ok
clc; close all;
rng(7);

%% Configuration
% Store all model, training, and prompt-structure parameters in cfg.
% The context length L corresponds to the prompt:
%   "the [subject] [verb] on the [object] because it is"
% The subject, object, and query indices are used later when interpreting
% attention and designing object-only perturbation experiments
cfg.L = 9;                % context length
cfg.d = 32;               % token embedding dimension
cfg.dk = 16;              % per-head query/key dimension
cfg.dv = 16;              % per-head value dimension
cfg.mHeads = 2;           % number of heads
cfg.dComb = 16;           % combined representation dimension after Wo
cfg.numEpochs = 220;      % number of trainning epochs
cfg.batchSize = 32;       % the size of each batch
cfg.learnRate = 1e-3;     % the learning rate
cfg.weightDecay = 1e-4;

% Prompt structure: "the [subject] [verb] on the [object] because it is"
cfg.subjectIndex = 2;     % the position of the subject in the context
cfg.objectIndex = 6;      % the position of the object in the context
cfg.queryIndex = 9;       % the position of the query

%% Synthesized English corpus
% Define word pools used to generate the templated English corpus.
% Each generated sentence follows the form:
%   "the [subject] [verb] on the [object] because it is [adjective]"
% The adjective is tied to the object so that the next-token prediction
% task has a simple but meaningful structure
subjects = ["cat","dog","child","bird","cup","book","baby","car","fish","teacher", ...
            "student","family","rabbit","fox","artist","doctor","farmer","pilot","singer","writer", ...
            "nurse","driver","actor","player","mother","father","painter","chef","dancer","friend"];
verbs = ["sat","rested","stayed","lay"];
objects = ["mat","rug","chair","branch","shelf","table","crib","garage","pond","board", ...
            "library","garden","cushion","basket","sofa","bench"];
adjs = ["soft","warm","small","high","stable","heavy","safe","quiet","calm","clean", ...
        "silent","peaceful","cozy","round","comfortable","strong"];

corpus = generateTemplatedCorpus(subjects, verbs, objects, adjs); % use a local function to generate synthesized sentences
corpus = [corpus; "the cat sat on the mat because it is soft"];
corpus = unique(corpus, 'stable'); % remove repeated sentences
fprintf('Generated templated corpus with %d sentences.\n', numel(corpus));

%% Build vocabulary and dataset
% Build a vocabulary from the corpus and convert each sentence into
% integer token IDs. Then create a next-token prediction dataset:
%   context = previous L tokens,
%   target  = next token.
% The dataset is randomly shuffled and split into training and validation subsets.
[vocab, word2id] = buildVocabulary(corpus);
cfg.V = numel(vocab);
cfg.tau = 1 / sqrt(cfg.dk);

allData = makeWordLevelDataset(corpus, word2id, cfg.L); % synthesize dataset
N = numel(allData);
perm = randperm(N);
allData = allData(perm);
Ntrain = max(1, round(0.85 * N));
dataTrain = allData(1:Ntrain); % Randomly take 85% of data set as trainning data
dataVal = allData(Ntrain+1:end); % Randomly take 15% of data set as validation data

fprintf('Vocabulary size: %d\n', cfg.V);
fprintf('Training samples: %d, Validation samples: %d\n', numel(dataTrain), numel(dataVal));

%% Initialize learnables
% Initialize trainable matrices for embeddings, positional encodings,
% query/key/value projections, output projection, and final classifier.
% Also initialize Adam optimizer states:
%   trailingAvg   = first-moment estimates,
%   trailingAvgSq = second-moment estimates.
params = initLearnables(cfg); % initilize learning parameters
trailingAvg = initStateLike(params, 0);
trailingAvgSq = initStateLike(params, 0);
iteration = 0;

history.trainLoss = zeros(cfg.numEpochs,1);
history.trainAcc = zeros(cfg.numEpochs,1);
history.valLoss = zeros(cfg.numEpochs,1);
history.valAcc = zeros(cfg.numEpochs,1);

%% Training loop
% Train the model using mini-batch stochastic optimization.
% For each batch:
%   1. Convert samples into context and target arrays.
%   2. Compute loss and gradients using automatic differentiation.
%   3. Update all trainable parameters using Adam.
% After each epoch, evaluate the model on the validation set and store training/validation loss and accuracy in history.
numTrain = numel(dataTrain);
numItersPerEpoch = ceil(numTrain / cfg.batchSize);

for epoch = 1:cfg.numEpochs
    idx = randperm(numTrain);
    dataTrain = dataTrain(idx);

    epochLoss = 0;
    epochAcc = 0;
    seen = 0;

    for it = 1:numItersPerEpoch
        i1 = (it-1) * cfg.batchSize + 1;
        i2 = min(it * cfg.batchSize, numTrain);
        batch = dataTrain(i1:i2);

        [contexts, targets] = makeBatch(batch, cfg);
        iteration = iteration + 1;

        [loss, grads, batchAcc] = dlfeval(@modelGradients, params, contexts, targets, cfg);

        [params.E, trailingAvg.E, trailingAvgSq.E] = adamupdate(params.E, grads.E, ...
            trailingAvg.E, trailingAvgSq.E, iteration, cfg.learnRate);
        [params.P, trailingAvg.P, trailingAvgSq.P] = adamupdate(params.P, grads.P, ...
            trailingAvg.P, trailingAvgSq.P, iteration, cfg.learnRate);
        [params.Wq, trailingAvg.Wq, trailingAvgSq.Wq] = adamupdate(params.Wq, grads.Wq, ...
            trailingAvg.Wq, trailingAvgSq.Wq, iteration, cfg.learnRate);
        [params.Wk, trailingAvg.Wk, trailingAvgSq.Wk] = adamupdate(params.Wk, grads.Wk, ...
            trailingAvg.Wk, trailingAvgSq.Wk, iteration, cfg.learnRate);
        [params.Wv, trailingAvg.Wv, trailingAvgSq.Wv] = adamupdate(params.Wv, grads.Wv, ...
            trailingAvg.Wv, trailingAvgSq.Wv, iteration, cfg.learnRate);
        [params.Wo, trailingAvg.Wo, trailingAvgSq.Wo] = adamupdate(params.Wo, grads.Wo, ...
            trailingAvg.Wo, trailingAvgSq.Wo, iteration, cfg.learnRate);
        [params.U, trailingAvg.U, trailingAvgSq.U] = adamupdate(params.U, grads.U, ...
            trailingAvg.U, trailingAvgSq.U, iteration, cfg.learnRate);
        [params.b, trailingAvg.b, trailingAvgSq.b] = adamupdate(params.b, grads.b, ...
            trailingAvg.b, trailingAvgSq.b, iteration, cfg.learnRate);

        bsz = size(contexts,2);
        epochLoss = epochLoss + double(gather(extractdata(loss))) * bsz;
        epochAcc = epochAcc + batchAcc * bsz;
        seen = seen + bsz;
    end

    trainLoss = epochLoss / seen;
    trainAcc = epochAcc / seen;

    modelNum = stripLearnables(params);
    [valLoss, valAcc] = evaluateDataset(modelNum, dataVal, cfg);

    history.trainLoss(epoch) = trainLoss;
    history.trainAcc(epoch) = trainAcc;
    history.valLoss(epoch) = valLoss;
    history.valAcc(epoch) = valAcc;

    if mod(epoch,10)==0 || epoch==1 || epoch==cfg.numEpochs
        fprintf('Epoch %3d/%3d | TrainLoss %.4f | TrainAcc %.2f%% | ValLoss %.4f | ValAcc %.2f%%\n', ...
            epoch, cfg.numEpochs, trainLoss, 100*trainAcc, valLoss, 100*valAcc);
    end
end

%% Demo prompt
% Run one fixed example prompt through the trained model to sanity-check
% that the learned matrices produce a next-token prediction.
examplePrompt = ["the","cat","sat","on","the","mat","because","it","is"];
exampleIds = wordsToIds(examplePrompt, word2id);
out = forwardOneNumeric(stripLearnables(params), exampleIds, cfg);
[~, predId] = max(out.prob);
fprintf('\nExample prompt: %s\n', strjoin(examplePrompt, ' '));
fprintf('Predicted next word: %s\n', vocab(predId));

%% Save model
% Convert learnable dlarray parameters into ordinary double arrays and
% save them together with configuration, training history, datasets,
% vocabulary, and corpus. The saved file is used by the evaluation script.
model = stripLearnables(params);
save('trained_multihead_english_attention.mat', ...
    'model', 'cfg', 'history', 'dataTrain', 'dataVal', 'vocab', 'corpus');
fprintf('\nSaved learned matrices to trained_multihead_english_attention.mat\n');

%% Optional plots
figure('Name','Training diagnostics','Color','w');
subplot(1,2,1);
plot(history.trainLoss,'LineWidth',1.5); hold on;
plot(history.valLoss,'LineWidth',1.5);
grid on; xlabel('Epoch'); ylabel('Loss'); title('Training / Validation Loss');
legend('Train','Val','Location','best');
subplot(1,2,2);
plot(100*history.trainAcc,'LineWidth',1.5); hold on;
plot(100*history.valAcc,'LineWidth',1.5);
grid on; xlabel('Epoch'); ylabel('Accuracy (%)'); title('Training / Validation Accuracy');
legend('Train','Val','Location','best');

end

%% ========================= Local functions =========================
%
function corpus = generateTemplatedCorpus(subjects, verbs, objects, adjs)

% Generate the synthetic English corpus.
%
% Purpose:
%   Create templated sentences of the form
%       "the [subject] [verb] on the [object] because it is [adjective]".
%   The object and adjective lists are paired element-wise, while the verb
%   is selected cyclically to introduce simple variation.
%
% Inputs:
%   subjects : string array of possible subjects.
%   verbs    : string array of possible verbs.
%   objects  : string array of possible objects.
%   adjs     : string array of adjectives paired with objects.
%
% Output:
%   corpus   : string column vector containing all generated sentences.

numSubjects = numel(subjects);
numVerbs = numel(verbs);
numObjects = numel(objects);
assert(numObjects == numel(adjs), 'objects and adjs must have the same length');

sentences = strings(0,1);
for s = 1:numSubjects
    for o = 1:numObjects
        v = verbs(1 + mod(s + o - 2, numVerbs));
        sentence = "the " + subjects(s) + " " + v + " on the " + objects(o) + " because it is " + adjs(o);
        sentences(end+1,1) = sentence; %#ok
    end
end
corpus = sentences;
end
%


%
function [vocab, word2id] = buildVocabulary(corpus)

% Build a word vocabulary and word-to-index map.
%
% Purpose:
%   Extract all words from the corpus, remove duplicates while preserving
%   first-occurrence order, and assign each word a unique integer index.
%
% Input:
%   corpus  : string array containing all sentences.
%
% Outputs:
%   vocab   : string row vector of unique words.
%   word2id : containers.Map from word strings to integer token IDs.

allWords = strings(0,1);
for i = 1:numel(corpus)
    w = split(lower(strtrim(corpus(i))));
    allWords = [allWords; w(:)]; %#ok
end
vocab = unique(allWords, 'stable')';
word2id = containers.Map('KeyType','char','ValueType','int32');
for i = 1:numel(vocab)
    word2id(char(vocab(i))) = i;
end
end
%


function ids = wordsToIds(words, word2id)

% Convert words to their integer token IDs.
%
% Purpose:
%   Convert a list of words into the corresponding vocabulary indices.
%
% Inputs:
%   words   : string array or cell-like list of words.
%   word2id : containers.Map from words to token IDs.
%
% Output:
%   ids     : row vector of integer token IDs.

ids = zeros(1, numel(words));
for i = 1:numel(words)
    ids(i) = word2id(char(lower(string(words(i)))));
end
end


%
function data = makeWordLevelDataset(corpus, word2id, L)

% Create a word-level next-token prediction dataset.
%
% Purpose:
%   Convert each sentence into one or more samples for next-token
%   prediction. Each sample consists of:
%       context     = previous L token IDs,
%       target      = next token ID,
%       contextText = context as text,
%       targetText  = target word as text.
%
% Inputs:
%   corpus  : string array of sentences.
%   word2id : containers.Map from words to token IDs.
%   L       : context length.
%
% Output:
%   data    : struct array with fields:
%             context, target, contextText, targetText.

entries = struct('context', {}, 'target', {}, 'contextText', {}, 'targetText', {});
count = 0;
for s = 1:numel(corpus)
    words = split(lower(strtrim(corpus(s))))';
    if numel(words) < L+1
        continue;
    end
    ids = wordsToIds(words, word2id);
    for t = L+1:numel(ids)
        count = count + 1;
        entries(count).context = ids(t-L:t-1); %#ok
        entries(count).target = ids(t); %#ok
        entries(count).contextText = strjoin(words(t-L:t-1), ' '); %#ok<AGROW>
        entries(count).targetText = words(t); %#ok
    end
end
data = entries;
end
%


function [contexts, targets] = makeBatch(batch, cfg)

% Convert a struct mini-batch into numeric arrays.
%
% Purpose:
%   Collect context token IDs and target token IDs from a batch of samples
%   into matrix/vector form for training.
%
% Inputs:
%   batch : struct array of training samples.
%   cfg   : configuration struct containing cfg.L.
%
% Outputs:
%   contexts : L-by-B matrix of context token IDs, where B is batch size.
%   targets  : 1-by-B vector of target token IDs.

B = numel(batch);
contexts = zeros(cfg.L, B);
targets = zeros(1, B);
for i = 1:B
    contexts(:,i) = batch(i).context(:);
    targets(i) = batch(i).target;
end
end

function params = initLearnables(cfg)

% Initialize all trainable model parameters.
%
% Purpose:
%   Initialize embeddings, positional encodings, query/key/value matrices,
%   multi-head output projection, classifier weights, and bias.
%   Parameters are stored as dlarray objects so that MATLAB can compute
%   gradients using automatic differentiation.
%
% Input:
%   cfg    : configuration struct containing model dimensions.
%
% Output:
%   params : struct containing trainable dlarray parameters:
%            E, P, Wq, Wk, Wv, Wo, U, b.

scale = 0.10;
params.E = dlarray(scale * randn(cfg.d, cfg.V, 'single'));
params.P = dlarray(scale * randn(cfg.d, cfg.L, 'single'));
params.Wq = dlarray(scale * randn(cfg.dk, cfg.d, cfg.mHeads, 'single'));
params.Wk = dlarray(scale * randn(cfg.dk, cfg.d, cfg.mHeads, 'single'));
params.Wv = dlarray(scale * randn(cfg.dv, cfg.d, cfg.mHeads, 'single'));
params.Wo = dlarray(scale * randn(cfg.dComb, cfg.dv * cfg.mHeads, 'single'));
params.U = dlarray(scale * randn(cfg.dComb, cfg.V, 'single'));
params.b = dlarray(zeros(cfg.V, 1, 'single'));
end

function state = initStateLike(params, value)

% Initialize a parameter-shaped state structure.
%
% Purpose:
%   Create a struct with the same fields, sizes, and data types as params,
%   but filled with a constant value. This is mainly used to initialize
%   optimizer states such as Adam first- and second-moment estimates.
%
% Inputs:
%   params : struct of trainable parameters.
%   value  : scalar value used to fill each state array, typically 0.
%
% Output:
%   state  : struct with the same fields as params:
%            E, P, Wq, Wk, Wv, Wo, U, b.

state.E = value * zeros(size(params.E), 'like', extractdata(params.E));
state.P = value * zeros(size(params.P), 'like', extractdata(params.P));
state.Wq = value * zeros(size(params.Wq), 'like', extractdata(params.Wq));
state.Wk = value * zeros(size(params.Wk), 'like', extractdata(params.Wk));
state.Wv = value * zeros(size(params.Wv), 'like', extractdata(params.Wv));
state.Wo = value * zeros(size(params.Wo), 'like', extractdata(params.Wo));
state.U = value * zeros(size(params.U), 'like', extractdata(params.U));
state.b = value * zeros(size(params.b), 'like', extractdata(params.b));
end

function model = stripLearnables(params)

% Convert dlarray parameters to ordinary numeric arrays.
%
% Purpose:
%   Remove dlarray wrappers, gather data from the device if needed, and
%   convert all learnable parameters to double precision arrays. The output
%   model is used for numerical evaluation and saving to MAT files.
%
% Input:
%   params : struct of trainable dlarray parameters.
%
% Output:
%   model  : struct of ordinary double arrays with fields:
%            E, P, Wq, Wk, Wv, Wo, U, b.

model.E = double(gather(extractdata(params.E)));
model.P = double(gather(extractdata(params.P)));
model.Wq = double(gather(extractdata(params.Wq)));
model.Wk = double(gather(extractdata(params.Wk)));
model.Wv = double(gather(extractdata(params.Wv)));
model.Wo = double(gather(extractdata(params.Wo)));
model.U = double(gather(extractdata(params.U)));
model.b = double(gather(extractdata(params.b)));
end

function [loss, grads, acc] = modelGradients(params, contexts, targets, cfg)

% Compute loss, gradients, and batch accuracy.
%
% Purpose:
%   Perform a forward pass for each context in a mini-batch, compute the
%   cross-entropy next-token prediction loss with weight decay, and compute
%   gradients with respect to all learnable parameters using dlgradient.
%
% Inputs:
%   params   : struct of trainable dlarray parameters.
%   contexts : L-by-B matrix of context token IDs.
%   targets  : 1-by-B vector of target token IDs.
%   cfg      : configuration struct with model/training parameters.
%
% Outputs:
%   loss  : scalar dlarray containing batch loss.
%   grads : struct of gradients with same fields as params.
%   acc   : scalar batch accuracy in [0,1].

B = size(contexts, 2);
logits = dlarray(zeros(cfg.V, B, 'single'));
for b = 1:B
    contextIds = double(contexts(:,b))';
    out = forwardOneDL(params, contextIds, cfg);
    logits(:,b) = out.logits;
end

probs = softmaxColumns(logits);
loss = dlarray(single(0));
for b = 1:B
    loss = loss - log(probs(targets(b), b) + 1e-8);
end
loss = loss / B;

reg = sum(params.E.^2,'all') + sum(params.P.^2,'all') + ...
      sum(params.Wq.^2,'all') + sum(params.Wk.^2,'all') + ...
      sum(params.Wv.^2,'all') + sum(params.Wo.^2,'all') + ...
      sum(params.U.^2,'all') + sum(params.b.^2,'all');
loss = loss + cfg.weightDecay * reg;

[gE, gP, gWq, gWk, gWv, gWo, gU, gb] = dlgradient(loss, ...
    params.E, params.P, params.Wq, params.Wk, params.Wv, params.Wo, params.U, params.b);

grads.E = gE; grads.P = gP; grads.Wq = gWq; grads.Wk = gWk;
grads.Wv = gWv; grads.Wo = gWo; grads.U = gU; grads.b = gb;

P = gather(extractdata(probs));
preds = zeros(1,B);
for b = 1:B
    [~, preds(b)] = max(P(:,b));
end
acc = mean(preds == targets);
end

function out = forwardOneDL(params, contextIds, cfg)

% Differentiable forward pass for one context.
%
% Purpose:
%   Compute the model logits for a single context using dlarray parameters.
%   This function is used during training, so all operations remain
%   differentiable for automatic differentiation.
%
% Inputs:
%   params     : struct of trainable dlarray parameters.
%   contextIds : row vector of context token IDs of length cfg.L.
%   cfg        : configuration struct with model dimensions and tau.
%
% Output:
%   out        : struct containing:
%                logits = V-by-1 dlarray of next-token logits.

X = params.E(:, contextIds) + params.P;
Xhist = X(:,1:end-1);
xq = X(:,end);
headVecs = cell(1, cfg.mHeads);
for l = 1:cfg.mHeads
    q = params.Wq(:,:,l) * xq;
    K = params.Wk(:,:,l) * Xhist;
    V = params.Wv(:,:,l) * Xhist;
    scores = q' * K;
    alpha = softmaxRows(scores / cfg.tau);
    h_l = V * alpha';
    headVecs{l} = h_l;
end
hCat = cat(1, headVecs{:});
hCombined = params.Wo * hCat;
logits = params.U' * hCombined + params.b;
out.logits = logits;
end

function [loss, acc] = evaluateDataset(model, data, cfg)

% Evaluate numeric model on a dataset.
%
% Purpose:
%   Run the trained numeric model on each sample in a dataset and compute
%   average negative log-likelihood loss and prediction accuracy.
%
% Inputs:
%   model : struct of numeric model parameters.
%   data  : struct array of samples with fields context and target.
%   cfg   : configuration struct.
%
% Outputs:
%   loss  : average negative log-likelihood over the dataset.
%   acc   : average prediction accuracy over the dataset.

N = numel(data);
totalLoss = 0;
correct = 0;
for n = 1:N
    out = forwardOneNumeric(model, data(n).context, cfg);
    y = data(n).target;
    totalLoss = totalLoss - log(max(out.prob(y), 1e-12));
    correct = correct + double(out.pred == y);
end
loss = totalLoss / N;
acc = correct / N;
end

function out = forwardOneNumeric(model, contextIds, cfg)

% Numeric forward pass for one context.
%
% Purpose:
%   Compute the full forward pass using ordinary numeric arrays. In
%   addition to logits and prediction probabilities, this function also
%   stores per-head attention scores, attention weights, head vectors, and
%   the bounded-rationality metric value deltaV for each head.
%
% Inputs:
%   model      : struct of numeric model parameters.
%   contextIds : row vector of context token IDs of length cfg.L.
%   cfg        : configuration struct with model dimensions and tau.
%
% Output:
%   out        : struct containing:
%                X          = embedded context with positional encodings,
%                heads      = per-head scores, alpha, h, deltaV,
%                hCat       = concatenated head outputs,
%                hCombined  = combined representation,
%                logits     = next-token logits,
%                prob       = next-token probability vector,
%                pred       = predicted token ID.

X = double(model.E(:, contextIds) + model.P);
Xhist = X(:,1:end-1);
xq = X(:,end);
heads = struct('scores', cell(1,cfg.mHeads), 'alpha', cell(1,cfg.mHeads), ...
               'h', cell(1,cfg.mHeads), 'deltaV', cell(1,cfg.mHeads));
hCat = [];
for l = 1:cfg.mHeads
    q = model.Wq(:,:,l) * xq;
    K = model.Wk(:,:,l) * Xhist;
    V = model.Wv(:,:,l) * Xhist;
    scores = q' * K;
    alpha = softmaxRowNumeric(scores / cfg.tau);
    h_l = V * alpha';
    deltaV = cfg.tau * logsumexpNumeric(scores / cfg.tau) - max(scores);
    heads(l).scores = scores;
    heads(l).alpha = alpha;
    heads(l).h = h_l;
    heads(l).deltaV = deltaV;
    hCat = [hCat; h_l]; %#ok
end
hCombined = model.Wo * hCat;
logits = model.U' * hCombined + model.b;
prob = softmaxColNumeric(logits);
[~, pred] = max(prob);
out.X = X;
out.heads = heads;
out.hCat = hCat;
out.hCombined = hCombined;
out.logits = logits;
out.prob = prob;
out.pred = pred;
end

function y = softmaxRows(x)

% Compute row-wise softmax.

xmax = max(x, [], 2);
ex = exp(x - xmax);
y = ex ./ sum(ex, 2);
end

function y = softmaxColumns(x)

% Compute column-wise softmax.

xmax = max(x, [], 1);
ex = exp(x - xmax);
y = ex ./ sum(ex, 1);
end

function y = softmaxRowNumeric(x)

% Compute row-wise softmax for numeric arrays.

x = x - max(x, [], 2);
ex = exp(x);
y = ex ./ sum(ex, 2);
end

function y = softmaxColNumeric(x)

% Compute softmax for a numeric column vector.

x = x - max(x);
ex = exp(x);
y = ex ./ sum(ex);
end

function y = logsumexpNumeric(x)

% Numerically stable log-sum-exp.

m = max(x);
y = m + log(sum(exp(x - m)));
end
