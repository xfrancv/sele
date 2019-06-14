function [ w, stat ] = parbmrm( data, risk, lambda,  opts)
%PARBMRM Summary of this function goes here
%   Detailed explanation goes here

    if nargin<4, opts = struct(); end;
    if ~isfield(opts, 'tolRel'), opts.tolRel = 1e-3; end;
    if ~isfield(opts, 'tolAbs'), opts.tolAbs = 0; end;
    if ~isfield(opts, 'maxIter'), opts.maxIter = Inf; end;
    if ~isfield(opts, 'verb'), opts.verb = 0; end;
    if ~isfield(opts,'maxMemory'), opts.maxMemory = Inf; end
    if ~isfield(opts,'cleanBuffer'), opts.cleanBuffer = inf; end
    if ~isfield(opts,'useParfor'), opts.useParfor = 1; end
    if ~isfield(opts,'bufSize'), opts.bufSize = 1000; end
    if ~isfield(opts,'useCplex'), opts.useCplex = 0; end

    if ~iscell(data), data = {data}; end;
    nThreads = length(data);

    stime = cputime;
%     tic;
    clck0_ = clock;
    
    % constants
    exitflag = -1;
    
    % get space dimension
    [R, SG] = risk(data{1});
    
    wDim  = length(SG);
    w = zeros(wDim, 1, 'double');
    t = 0;
    Rt = zeros(1, nThreads);
    SGt = zeros(wDim, nThreads);
    
    % check if paralel toolbox is installed
    if(exist('parfor', 'builtin') & opts.useParfor)
        parfor p = 1:nThreads
            [Rt(p), SGt(:,p)] = risk(data{p}, w);
        end;
    else
        for p = 1:nThreads
            [Rt(p), SGt(:,p)] = risk(data{p}, w);
%            data{p} = ldata;
        end;
    end;
    hR = [];
    hSG = [];
    hw = [];
    I = [];
    f = [];
    alpha = [];
    nCP = 0;
    
    H = zeros( opts.bufSize*nThreads, opts.bufSize*nThreads );
    
    % alloc buffers for meassured statistics
%     hist_Fd = zeros(options.BufSize+1,1);
%     hist_Fp = zeros(options.BufSize+1,1);
%     hist_R = zeros(options.BufSize+1,1);
%     hist_runtime = zeros(options.BufSize+1,1);
    
    hist_Fd(1) = -inf;
    hist_Fp(1) = R+0.5*lambda*norm(w)^2;
    hist_R(1) = R;
    %hist_runtime(1) = toc;
    clck1_ = clock;
    hist_runtime(1) = etime(clck1_, clck0_);
    
    fprintf('%4d: Starting BRMR in %d threads\n', t, nThreads);
    
    while exitflag == -1;
        
        %tic;
        clck0 = clock;
        
        t = t + 1;
        nCP = nCP + nThreads;
        hR = [hR Rt'];
        for i = 1:nThreads
            hSG{end+1} = SGt(:,i);
        end;
%        hSG = [hSG SGt];
        hw = [hw w];
        f = [f (w'*SGt - Rt)];
        if t == 1
            %H = (SGt'*SGt)/lambda;            
            H(1:nThreads, 1:nThreads) = (SGt'*SGt)/lambda;
        else
            tmp = zeros(nThreads, nCP);
            for i = 1:nCP
                tmp(:,i) = SGt'*hSG{i}/lambda;
            end;
%             H = [H tmp(:,1:end-nThreads)'];
%             H = [H; tmp];
            
            H(1:nCP-nThreads,nCP-nThreads+1:nCP) = tmp(:,1:nCP-nThreads)';
            H(nCP-nThreads+1:nCP,1:nCP) = tmp;
 %           H(1:size(tmp,1),
        end;
        I = [I 1:nThreads];
        alpha = [alpha; zeros(nThreads, 1)];

        k = numel(f);
        switch opts.useCplex
            case 0
                [alpha, stat] = libqp_splx(H(1:k,1:k), f, ones(1, nThreads), I, ones(1, nThreads), alpha);
            case 1
              Aineq=zeros(nThreads,k);
              for i=1:nThreads
                  Aineq(i,find(i==I)) = 1; 
              end
              [alpha, stat.QP] = cplexqp(H(1:k,1:k),f,Aineq,ones(nThreads,1),[],[],zeros(k,1),ones(k,1));
        end
              
        w = zeros(size(SGt,1),1);
        for i = 1:nCP
            w = w - hSG{i}*alpha(i)/lambda;
        end;

        Rt = zeros(1, nThreads);
        SGt = zeros(wDim, nThreads);
        if(exist('parfor', 'builtin') & opts.useParfor)
            parfor p = 1:nThreads
                [Rt(p), SGt(:,p)] = risk(data{p}, w);
            end;
        else
            for p = 1:nThreads
                [Rt(p), SGt(:,p)] = risk(data{p}, w);
            end;
        end
        
%         R = max(reshape(w'*hSG-f, nThreads, t), [], 2);
        
        Fp = lambda/2*norm(w)^2+sum(Rt);
        Fd = -stat.QP;    

        if Fp-Fd<= opts.tolRel*abs(Fp)
            exitflag= 1;
        elseif Fp-Fd <= opts.tolAbs
            exitflag= 2;    
        elseif t >= opts.maxIter
            exitflag= 0;
        end         

        tmp = whos('H');
        mem = tmp.bytes;
        tmp = whos('hSG');
        mem = mem + tmp.bytes;
        tmp = whos('hR');
        mem = mem + tmp.bytes;
        
        %hist_runtime(t+1) = toc;
        clck1 = clock;
        hist_runtime(t+1) = etime(clck1, clck0);
        hist_Fp(t+1) = Fp;
        hist_Fd(t+1) = Fd;
        hist_R(t+1) = sum(Rt);
        
        nzA = length(find(alpha > 0));
        if mod(t,opts.verb) == 0 || exitflag ~= -1
%             fprintf('%4d: tim=%.3f, Fp=%f, Fd=%f, (Fp-Fd)=%f, (Fp-Fd)/Fp=%f, R=%f, nCP=%d, nzA=%d, mem=%dMB\n', ...
%                 t, cputime - stime, Fp, Fd, Fp-Fd,(Fp-Fd)/Fp, sum(Rt), nCP, nzA, ceil(mem/1024/1024));
            fprintf('%4d: tim=%.3f, Fp=%f, Fd=%f, (Fp-Fd)=%f, (Fp-Fd)/Fp=%f, R=%f, nCP=%d, nzA=%d, mem=%dMB\n', ...
                t, hist_runtime(t+1), Fp, Fd, Fp-Fd,(Fp-Fd)/Fp, sum(Rt), nCP, nzA, ceil(mem/1024/1024));
        end;

        if opts.maxMemory < (mem/1024/1024)
            exitflag = -2;
        end;
        
        if mod(t,opts.cleanBuffer) == 0 && nzA < t*nThreads
            old_nCP = nCP;
            idx = find(alpha > 0);
            nCP = length(idx);

            alpha = alpha(idx);
            H(1:nCP,1:nCP) = H(idx,idx);
            hSG(:,1:nCP) = hSG(:,idx);        
            hR(1:nCP) = hR(idx);        

            fprintf('done. (old_nCP=%d, new_nCP=%d) \n',old_nCP, nCP);
        end
        
        
    end;

    stat = [];
    stat.Fp = Fp;
    stat.Fd = Fd;
    stat.nIter = t;
    stat.hist.Fp = hist_Fp(1:t+1);
    stat.hist.Fd = hist_Fd(1:t+1);
    stat.hist.R = hist_R(1:t+1);
    stat.hist.runtime = hist_runtime(1:t+1);
    
end

