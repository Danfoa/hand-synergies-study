close all
clear all
load('KIN_MUS_UJI.mat')

data = EMG_KIN_v4;

% Create blank template structure
blank_struct = {};
fn = fieldnames(data(1));
for k=1:numel(fn)
    blank_struct.(fn{k}) = [];
end
blank_struct = rmfield(blank_struct, "Subject");

idxs = [data.Subject];
% Iterate through user data.
for sub=20:22
    % Find subject specific recording structure field ids    
    sub_idxs = find(idxs == sub);
    [~, num_records] = size(sub_idxs);
    
    user_data = blank_struct;
    % Iterate through each recording and generate structure 
    for i = sub_idxs
        record = data(i);
        if(isnan(record.time))
            continue;
        end 
        [n, ~] = size(record.time);
        user_data.Phase = [user_data.Phase; ones(n,1).*record.Phase];
        user_data.ADL = [user_data.ADL; ones(n,1).*record.ADL];
        user_data.time = [user_data.time; record.time];
        user_data.Kinematic_data = [user_data.Kinematic_data; record.Kinematic_data];
        user_data.EMG_data = [user_data.EMG_data; record.EMG_data];
    end 
    struct2csv(user_data, "CSV_Data/KIN_MUS_S"+sub+".csv");
    sub
end 