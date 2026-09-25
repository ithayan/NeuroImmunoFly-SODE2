% NeuroImmunoFly-SODE: MATLAB Simulation Engine
% Generates Connectome and Simulates Stochastic ODE

function simulate_neuroimmunology()
    fprintf('Starting MATLAB NeuroImmunoFly-SODE Simulation...\n');
    
    %% 1. Connectome Generation
    fprintf('Generating Connectome...\n');
    nodes = [50, 200, 20, 100]; % T1, T2, T3, T4
    total_nodes = sum(nodes);
    A = zeros(total_nodes, total_nodes);
    
    % Index ranges
    idx_T1 = 1:50;
    idx_T2 = 51:250;
    idx_T3 = 251:270;
    idx_T4 = 271:370;
    
    % Inter-tier probabilities
    p_T1_T2 = 0.08;
    p_T2_T3 = 0.12;
    p_T3_T4 = 0.15;
    p_feedback = 0.05;
    p_lateral = 0.03;
    
    % T1 -> T2
    A(idx_T1, idx_T2) = rand(length(idx_T1), length(idx_T2)) < p_T1_T2;
    % T2 -> T3
    A(idx_T2, idx_T3) = rand(length(idx_T2), length(idx_T3)) < p_T2_T3;
    % T3 -> T4
    A(idx_T3, idx_T4) = rand(length(idx_T3), length(idx_T4)) < p_T3_T4;
    % Feedback T4 -> T3
    A(idx_T4, idx_T3) = rand(length(idx_T4), length(idx_T3)) < p_feedback;
    
    % Intra-tier lateral
    for i = 1:4
        idx = sum(nodes(1:i-1))+1 : sum(nodes(1:i));
        lateral = rand(length(idx), length(idx)) < p_lateral;
        % Remove self-loops
        lateral = lateral - diag(diag(lateral));
        A(idx, idx) = A(idx, idx) | lateral;
    end
    
    % Export Adjacency Matrix
    % Use absolute path
    data_dir = 'c:\Users\91900\.gemini\antigravity-ide\scratch\NeuroImmunoFly-SODE\data';
    if ~exist(data_dir, 'dir')
        mkdir(data_dir);
    end
    csvwrite(fullfile(data_dir, 'connectome_adj.csv'), A);
    fprintf('Connectome saved to connectome_adj.csv\n');
    
    %% 2. SODE Simulation
    fprintf('Simulating SODE...\n');
    % Parameters
    alpha = 1.0;
    gamma = 0.3;
    k1 = 0.5;
    delta1 = 0.1;
    k2 = 0.4;
    delta2 = 0.15;
    mu = 0.05;
    sigma = 0.02;
    t_stress = 20;
    
    % Simulation settings
    T_end = 100;
    dt = 0.01;
    N_steps = round(T_end / dt);
    t = linspace(0, T_end, N_steps)';
    
    % Initialize state vectors
    N_t = zeros(N_steps, 1);
    H_t = zeros(N_steps, 1);
    I_t = zeros(N_steps, 1);
    
    % Initial conditions (steady state roughly)
    N_t(1) = 0;
    H_t(1) = 0;
    I_t(1) = 0;
    
    % Euler-Maruyama integration
    for i = 1:(N_steps-1)
        % Stress step function
        S = (t(i) >= t_stress);
        
        % Noise terms (Wiener process increments)
        dW_N = sqrt(dt) * randn();
        dW_H = sqrt(dt) * randn();
        dW_I = sqrt(dt) * randn();
        
        % SODE equations
        dN = (alpha - gamma * N_t(i) * S) * dt + sigma * dW_N;
        dH = (k1 * N_t(i) - delta1 * H_t(i)) * dt + sigma * dW_H;
        dI = (k2 * H_t(i) - delta2 * I_t(i) - mu * I_t(i)^2) * dt + sigma * dW_I;
        
        N_t(i+1) = N_t(i) + dN;
        H_t(i+1) = H_t(i) + dH;
        I_t(i+1) = I_t(i) + dI;
    end
    
    %% 3. Export Trajectories
    T = table(t, N_t, H_t, I_t, 'VariableNames', {'t', 'N', 'H', 'I'});
    writetable(T, fullfile(data_dir, 'matlab_sode_trajectories.csv'));
    fprintf('Trajectories saved to matlab_sode_trajectories.csv\n');
    fprintf('MATLAB Simulation Complete.\n');
end
