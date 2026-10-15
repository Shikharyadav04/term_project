%% ================================================================
%  AI-GUIDED MULTI-ORDER MOR DATASET GENERATOR
%
%  Corrected version
%
%  Purpose:
%    Generate ground-truth data for an AI-guided MOR framework that
%    can generalize across different FOM orders.
%
%  FOM orders:
%    20, 30, 40, 50, 60, 80, 100
%
%  Systems per order:
%    50
%
%  Candidate ROM fractions:
%    10%, 20%, ..., 90%
%
%  MOR methods:
%    1) Balanced Truncation - Truncate
%    2) Balanced Truncation - MatchDC
%
%  IMPORTANT:
%    MATLAB is the ground-truth authority. No failed reduction is
%    silently converted into a zero/valid result.
%
% ================================================================

%% ================================================================

clear;
clc;
close all;
rng(20261002);

fprintf('\n===============================================================\n');
fprintf(' CORRECTED AI-GUIDED MULTI-ORDER MOR DATASET GENERATION\n');
fprintf('===============================================================\n');

%% 1. CONFIGURATION
FOM_ORDERS = [20 30 40 50 60 80 100];
SYSTEMS_PER_ORDER = 50;
ORDER_FRACTIONS = [0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90];

NRMSE_THRESHOLD = 0.01;
DC_THRESHOLD    = 0.01;
HINF_THRESHOLD  = 0.01;

t = (0:0.01:5)';
u = ones(size(t));
w = logspace(-2,4,1000);

R_bounds = [0.1 5.0];
L_bounds = [0.05 1.0];
C_bounds = [0.001 0.05];

output_dir = 'C:\Users\shikh\OneDrive\Desktop\FINAL TERM PROJECT\PART1-GETTING THE BEST MODEL';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
output_file = fullfile(output_dir, 'ai_guided_multi_order_mor_dataset.csv');

methods = {'BT_Truncate'; 'BT_MatchDC'};

num_fom_orders = numel(FOM_ORDERS);
num_methods = numel(methods);
num_candidate_fractions = numel(ORDER_FRACTIONS);
estimated_rows = num_fom_orders * SYSTEMS_PER_ORDER * ...
    num_candidate_fractions * num_methods;

fprintf('\nFOM orders: ');
fprintf('%d ', FOM_ORDERS);
fprintf('\nSystems per FOM order: %d\n', SYSTEMS_PER_ORDER);
fprintf('Expected rows: %d\n', estimated_rows);
fprintf('Output file: %s\n', output_file);

%% 2. DATA STORAGE
dataset = struct([]);
row = 0;
global_system_id = 0;
total_start = tic;

%% 3. MAIN FOM-ORDER LOOP
for order_index = 1:num_fom_orders
    n = FOM_ORDERS(order_index);

    if mod(n,2) ~= 0
        error('FOM order must be even. Found N=%d.', n);
    end

    num_sections = n/2;
    fprintf('\n===============================================================\n');
    fprintf(' FOM ORDER = %d | RLC SECTIONS = %d\n', n, num_sections);
    fprintf('===============================================================\n');

    % Latin hypercube sampling for each order.
    lhs_samples = lhsdesign(SYSTEMS_PER_ORDER, 3*num_sections, ...
        'criterion','maximin', 'iterations',20);

    R_all = zeros(SYSTEMS_PER_ORDER,num_sections);
    L_all = zeros(SYSTEMS_PER_ORDER,num_sections);
    C_all = zeros(SYSTEMS_PER_ORDER,num_sections);

    for k = 1:num_sections
        R_all(:,k) = R_bounds(1) + lhs_samples(:,k) .* ...
            (R_bounds(2)-R_bounds(1));
        L_all(:,k) = L_bounds(1) + lhs_samples(:,num_sections+k) .* ...
            (L_bounds(2)-L_bounds(1));
        C_all(:,k) = C_bounds(1) + lhs_samples(:,2*num_sections+k) .* ...
            (C_bounds(2)-C_bounds(1));
    end

    %% 3A. SYSTEM LOOP
    for system_index = 1:SYSTEMS_PER_ORDER
        global_system_id = global_system_id + 1;

        R_vec = R_all(system_index,:).';
        L_vec = L_all(system_index,:).';
        C_vec = C_all(system_index,:).';

        %% Construct RLC ladder state-space model
        % States x = [i1 v1 i2 v2 ... iN vN]'
        A = zeros(n,n);
        B = zeros(n,1);
        Cmat = zeros(1,n);
        D = 0;

        for k = 1:num_sections
            i_state = 2*k - 1;
            v_state = 2*k;

            % Inductor equation
            A(i_state,i_state) = -R_vec(k)/L_vec(k);
            A(i_state,v_state) = -1/L_vec(k);

            if k == 1
                B(i_state) = 1/L_vec(k);
            else
                previous_voltage = 2*(k-1);
                A(i_state,previous_voltage) = 1/L_vec(k);
            end

            % Capacitor equation
            A(v_state,i_state) = 1/C_vec(k);
            if k < num_sections
                next_current = 2*k + 1;
                A(v_state,next_current) = -1/C_vec(k);
            end
        end

        % Output is the final capacitor voltage.
        Cmat(n) = 1;
        sys_fom = ss(A,B,Cmat,D);

        %% FOM sanity checks
        fom_poles = eig(sys_fom);
        if any(~isfinite(fom_poles))
            error('Non-finite FOM poles: System %d, N=%d.', global_system_id,n);
        end
        max_pole = max(real(fom_poles));
        if max_pole >= 1e-8
            error('Unstable FOM: System %d, N=%d, max pole=%.6e', ...
                global_system_id,n,max_pole);
        end

        fom_dc = dcgain(sys_fom);
        if ~isfinite(fom_dc)
            error('Non-finite FOM DC gain: System %d, N=%d.', global_system_id,n);
        end

        %% FOM step response
        try
            y_fom = real(lsim(sys_fom,u,t));
        catch ME
            error('FOM simulation failed for System %d, N=%d:\n%s', ...
                global_system_id,n,ME.message);
        end
        if any(~isfinite(y_fom))
            error('Non-finite FOM step response: System %d, N=%d.', global_system_id,n);
        end
        y_range = max(y_fom)-min(y_fom);
        y_scale = max(y_range,1e-12);

        %% FOM frequency response
        try
            [mag_fom,~] = bode(sys_fom,w);
            mag_fom = abs(squeeze(mag_fom));
            mag_fom = mag_fom(:);
        catch ME
            error('FOM frequency response failed for System %d:\n%s', ...
                global_system_id,ME.message);
        end
        if any(~isfinite(mag_fom))
            error('Non-finite FOM frequency response: System %d.',global_system_id);
        end
        freq_scale = max(max(mag_fom),1e-12);
        freq_denominator = max(mag_fom,1e-8*freq_scale);
        FOM_Hinf = max(mag_fom);

        %% Hankel singular values
        try
            hsv = hsvd(sys_fom);
            hsv = abs(squeeze(hsv));
            hsv = hsv(:);
        catch ME
            error('HSVD failed for System %d, N=%d:\n%s', ...
                global_system_id,n,ME.message);
        end
        if numel(hsv) ~= n
            error('Unexpected HSV count for System %d: expected %d, got %d.', ...
                global_system_id,n,numel(hsv));
        end
        if any(~isfinite(hsv))
            error('Non-finite HSV values for System %d.',global_system_id);
        end

        %% HSV features
        hsv_total = sum(hsv);
        hsv_max = max(hsv);
        hsv_min = min(hsv);
        hsv_mean = mean(hsv);
        hsv_std = std(hsv);

        if hsv_total > 0
            p = hsv/hsv_total;
            p_nonzero = p(p > 0);
            hsv_entropy = -sum(p_nonzero .* log(p_nonzero));
        else
            hsv_entropy = 0;
        end

        cumulative_hsv = cumsum(hsv);
        effective_order_90 = find(cumulative_hsv >= 0.90*hsv_total,1,'first');
        effective_order_95 = find(cumulative_hsv >= 0.95*hsv_total,1,'first');
        effective_order_99 = find(cumulative_hsv >= 0.99*hsv_total,1,'first');

        %% Physical features
        R_min = min(R_vec); R_max = max(R_vec); R_mean = mean(R_vec);
        R_std = std(R_vec); R_cv = R_std/max(R_mean,eps);
        L_min = min(L_vec); L_max = max(L_vec); L_mean = mean(L_vec);
        L_std = std(L_vec); L_cv = L_std/max(L_mean,eps);
        C_min = min(C_vec); C_max = max(C_vec); C_mean = mean(C_vec);
        C_std = std(C_vec); C_cv = C_std/max(C_mean,eps);

        %% Candidate ROM orders
        candidate_orders = zeros(1,numel(ORDER_FRACTIONS));
        for q = 1:numel(ORDER_FRACTIONS)
            r = round(ORDER_FRACTIONS(q)*n);
            r = max(r,2);
            r = min(r,n-2);
            if mod(r,2) ~= 0
                r = r-1;
            end
            candidate_orders(q) = r;
        end
        candidate_orders = unique(candidate_orders);

        %% Candidate-order loop
        for q = 1:numel(candidate_orders)
            candidate_order = candidate_orders(q);
            candidate_fraction = candidate_order/n;

            retained_hsv_sum = sum(hsv(1:candidate_order));
            discarded_hsv_sum = sum(hsv(candidate_order+1:end));
            retained_hsv_fraction = retained_hsv_sum/max(hsv_total,eps);
            discarded_hsv_fraction = discarded_hsv_sum/max(hsv_total,eps);
            bt_bound = 2*discarded_hsv_sum;

            %% MOR method loop
            for method_index = 1:num_methods
                method_name = methods{method_index};
                fprintf('System %3d | N=%3d | r=%3d | %-14s\n', ...
                    global_system_id,n,candidate_order,method_name);

                reduction_success = true;
                reduction_error_message = "";
                reduction_start = tic;
                sys_rom = [];

                try
                    switch method_name
                        case 'BT_Truncate'
                            opts = balredOptions('StateProjection','Truncate');
                            sys_rom = balred(sys_fom,candidate_order,opts);
                        case 'BT_MatchDC'
                            opts = balredOptions('StateProjection','MatchDC');
                            sys_rom = balred(sys_fom,candidate_order,opts);
                        otherwise
                            error('Unknown MOR method: %s',method_name);
                    end
                catch ME
                    reduction_success = false;
                    reduction_error_message = string(ME.message);
                end
                reduction_time = toc(reduction_start);

                %% Default metrics: NaN means metric could not be computed.
                actual_rom_order = NaN;
                stable_rom = false;
                max_rom_pole = NaN;
                RMSE = NaN;
                NRMSE = NaN;
                SS_Error = NaN;
                RelDC_Error = NaN;
                Hinf_Error = NaN;
                RelHinf_Error = NaN;

                %% ROM verification and metrics
                if reduction_success
                    try
                        [Arom,~,~,~] = ssdata(sys_rom);
                        actual_rom_order = size(Arom,1);

                        rom_poles = eig(Arom);
                        max_rom_pole = max(real(rom_poles));
                        stable_rom = all(real(rom_poles) < 1e-8);

                        y_rom = real(lsim(sys_rom,u,t));
                        if any(~isfinite(y_rom))
                            error('ROM step response contains NaN/Inf.');
                        end

                        error_signal = y_fom-y_rom;
                        RMSE = sqrt(mean(error_signal.^2));
                        NRMSE = RMSE/y_scale;
                        SS_Error = abs(y_fom(end)-y_rom(end));

                        rom_dc = dcgain(sys_rom);
                        if ~isfinite(rom_dc)
                            error('ROM DC gain is NaN/Inf.');
                        end
                        RelDC_Error = abs(fom_dc-rom_dc)/max(abs(fom_dc),1e-12);

                        [mag_rom,~] = bode(sys_rom,w);
                        mag_rom = abs(squeeze(mag_rom));
                        mag_rom = mag_rom(:);
                        if any(~isfinite(mag_rom))
                            error('ROM frequency response contains NaN/Inf.');
                        end

                        abs_frequency_error = abs(mag_fom-mag_rom);
                        Hinf_Error = max(abs_frequency_error);
                        relative_frequency_error = abs_frequency_error./freq_denominator;
                        RelHinf_Error = max(relative_frequency_error);

                    catch ME
                        reduction_success = false;
                        reduction_error_message = string(ME.message);
                    end
                end

                %% Pass conditions
                pass_error = reduction_success && ...
                    isfinite(NRMSE) && isfinite(RelDC_Error) && ...
                    isfinite(RelHinf_Error) && ...
                    NRMSE <= NRMSE_THRESHOLD && ...
                    RelDC_Error <= DC_THRESHOLD && ...
                    RelHinf_Error <= HINF_THRESHOLD;

                Pass_All = pass_error && stable_rom;

                %% Store row
                row = row+1;
                dataset(row).System_ID = global_system_id;
                dataset(row).FOM_Order = n;
                dataset(row).Num_Sections = num_sections;
                dataset(row).Candidate_Order = candidate_order;
                dataset(row).Order_Fraction = candidate_fraction;
                dataset(row).Candidate_Fraction = candidate_fraction;
                dataset(row).Method = string(method_name);

                % Physical features
                dataset(row).R_min = R_min;
                dataset(row).R_max = R_max;
                dataset(row).R_mean = R_mean;
                dataset(row).R_std = R_std;
                dataset(row).R_cv = R_cv;
                dataset(row).L_min = L_min;
                dataset(row).L_max = L_max;
                dataset(row).L_mean = L_mean;
                dataset(row).L_std = L_std;
                dataset(row).L_cv = L_cv;
                dataset(row).C_min = C_min;
                dataset(row).C_max = C_max;
                dataset(row).C_mean = C_mean;
                dataset(row).C_std = C_std;
                dataset(row).C_cv = C_cv;

                % FOM dynamic features
                dataset(row).Max_Pole = max_pole;
                dataset(row).FOM_DC = fom_dc;
                dataset(row).FOM_Hinf = FOM_Hinf;
                dataset(row).HSV_Total = hsv_total;
                dataset(row).HSV_Max = hsv_max;
                dataset(row).HSV_Min = hsv_min;
                dataset(row).HSV_Mean = hsv_mean;
                dataset(row).HSV_Std = hsv_std;
                dataset(row).HSV_Entropy = hsv_entropy;
                dataset(row).Effective_Order_90 = effective_order_90;
                dataset(row).Effective_Order_95 = effective_order_95;
                dataset(row).Effective_Order_99 = effective_order_99;

                % Candidate dynamic features
                dataset(row).Retained_HSV_Sum = retained_hsv_sum;
                dataset(row).Discarded_HSV_Sum = discarded_hsv_sum;
                dataset(row).Retained_HSV_Fraction = retained_hsv_fraction;
                dataset(row).Discarded_HSV_Fraction = discarded_hsv_fraction;
                dataset(row).BT_Bound = bt_bound;

                % Reduction information
                dataset(row).Reduction_Success = reduction_success;
                dataset(row).Actual_ROM_Order = actual_rom_order;
                dataset(row).Stable_ROM = stable_rom;
                dataset(row).Max_ROM_Pole = max_rom_pole;

                % Error metrics
                dataset(row).RMSE = RMSE;
                dataset(row).NRMSE = NRMSE;
                dataset(row).SS_Error = SS_Error;
                dataset(row).RelDC_Error = RelDC_Error;
                dataset(row).Hinf_Error = Hinf_Error;
                dataset(row).RelHinf_Error = RelHinf_Error;

                % Runtime and labels
                dataset(row).Reduction_Time = reduction_time;
                dataset(row).Pass_Error = pass_error;
                dataset(row).Pass_All = Pass_All;
                dataset(row).Error_Message = reduction_error_message;
            end
        end
    end
end

%% 4. CONVERT TO TABLE AND SORT
fprintf('\nConverting structure to table...\n');
T = struct2table(dataset);
T = sortrows(T,{'FOM_Order','System_ID','Candidate_Order','Method'});

%% 5. DATASET INTEGRITY CHECK
fprintf('\nRunning final MATLAB dataset checks...\n');
numeric_columns = varfun(@isnumeric,T,'OutputFormat','uniform');
numeric_data = T{:,numeric_columns};
num_nan = sum(isnan(numeric_data),'all');
num_inf = sum(isinf(numeric_data),'all');

fprintf('Rows      : %d\n',height(T));
fprintf('Columns   : %d\n',width(T));
fprintf('NaN cells : %d\n',num_nan);
fprintf('Inf cells : %d\n',num_inf);

if num_nan > 0 || num_inf > 0
    error(['Dataset contains NaN/Inf values. CSV was NOT saved. ' ...
        'Inspect failed reductions/metrics before training.']);
end

if height(T) ~= estimated_rows
    error('Unexpected row count. Expected %d, got %d.', ...
        estimated_rows,height(T));
end

%% 6. SAVE CSV
writetable(T,output_file);
fprintf('\nCSV saved successfully:\n%s\n',output_file);

%% 7. SUMMARY (uses findgroups/splitapply for compatibility)
fprintf('\n===============================================================\n');
fprintf(' DATASET GENERATION COMPLETE\n');
fprintf('===============================================================\n');
fprintf('Rows       : %d\n',height(T));
fprintf('Columns    : %d\n',width(T));
fprintf('Systems    : %d\n',numel(unique(T.System_ID)));
fprintf('FOM orders : ');
fprintf('%d ',unique(T.FOM_Order));
fprintf('\n');

fprintf('\nRows by FOM order:\n');
[unique_orders,~,group_idx] = unique(T.FOM_Order);
rows_by_order = accumarray(group_idx,1);
rows_summary = table(unique_orders,rows_by_order, ...
    'VariableNames',{'FOM_Order','Num_Rows'});
disp(rows_summary);

fprintf('\nPass rate by FOM order and method:\n');
[G, fom_groups, method_groups] = findgroups(T.FOM_Order,T.Method);
pass_rates = splitapply(@mean,double(T.Pass_All),G);
pass_summary = table(fom_groups,method_groups,pass_rates, ...
    'VariableNames',{'FOM_Order','Method','Mean_Pass_All'});
pass_summary.Pass_Rate_Percent = 100*pass_summary.Mean_Pass_All;
disp(pass_summary);

fprintf('\nMinimum passing candidate orders:\n');
pass_rows = T(T.Pass_All,:);
if isempty(pass_rows)
    fprintf('No candidate satisfied all thresholds.\n');
else
    [G,system_ids,fom_groups,method_groups] = findgroups( ...
        pass_rows.System_ID,pass_rows.FOM_Order,pass_rows.Method);
    min_orders = splitapply(@min,pass_rows.Candidate_Order,G);
    min_pass = table(system_ids,fom_groups,method_groups,min_orders, ...
        'VariableNames',{'System_ID','FOM_Order','Method', ...
        'Minimum_Passing_Order'});
    disp(min_pass(1:min(30,height(min_pass)),:));
end

fprintf('\nRequested vs actual ROM-order mismatches:\n');
mismatch = T(T.Reduction_Success & ...
    T.Candidate_Order ~= T.Actual_ROM_Order, ...
    {'System_ID','FOM_Order','Candidate_Order','Actual_ROM_Order','Method'});
if isempty(mismatch)
    fprintf('No requested/actual ROM-order mismatches found.\n');
else
    fprintf('Number of mismatch rows: %d\n',height(mismatch));
    disp(mismatch(1:min(30,height(mismatch)),:));
end

total_time = toc(total_start);
fprintf('\nGeneration time: %.2f seconds\n',total_time);
fprintf('\nSaved file:\n%s\n',output_file);
fprintf('\nNEXT STEP: run Python 01_validate.py\n');
fprintf('===============================================================\n');
