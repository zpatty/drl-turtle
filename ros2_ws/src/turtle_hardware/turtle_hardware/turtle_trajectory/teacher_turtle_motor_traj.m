clear
%% Straight
% Need to make sure that trajectory has the same starting and end point
% For any cyclical trajectory, the starting point is given 1 second to move
% from turtle's home position to that starting pointin the cycle
% the cycle trajectory (i.e 2nd and last element) must be the same
% up = 210;
% down = 150;

% you do it by shoulder so that you only copy the fins you actually want to
% copy

%% todo: need to fix the dq and ddqd generation 
q_data = load('TEST_TEACHER_6/qd.mat', 'qd');
q_data = q_data.qd;

shape = size(q_data);
traj_len = shape(2);
shoulder1 = q_data(1, :);
shoulder2 = q_data(2, :);
shoulder3 = q_data(3, :);
shoulder4 = q_data(4, :);
shoulder5 = q_data(5, :);
shoulder6 = q_data(6, :);
% up = 190;
% down = 110;
% straight = [230 150 230];
% % front fins
% % shoulder1 = [up down up];
% shoulder2 = [straight];
% shoulder3 = [240 100 240];
% shoulder4 = 180 + (180 - shoulder1);
% shoulder5 = 180 + (180 - shoulder2);
% shoulder6 = [120 260 120];

% % back fins 
% rear_yaw = 180;
% rear_pitch = 180;
% shoulder7 = [180 180 180];
% shoulder8 = [180 180 180];
shoulder7 = repelem(180, traj_len);
shoulder8 = repelem(180, traj_len);

shoulder9 = repelem(180, traj_len);
shoulder10 = repelem(180, traj_len);
% % shoulder7 = [180 170 120 180];
% % shoulder8 = [180 200 270 180];
% shoulder9 = 180 + (180 - shoulder8);
% shoulder10 = 180 + (180 - shoulder7);
% time points for a single cycle
% timePoints = [0 1 2];
dt = traj_len - 1;
timePoints = 0:2/dt:2;
% tshift = timePoints(2:end) + timePoints(2:end);
% timePoints_cycle = [timePoints, tshift(2:end)];

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6; shoulder7; shoulder8; shoulder9; shoulder10] * pi/180;
% wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]
tvec = 0:0.01:timePoints(end);

[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints,tvec);

tvec = 0:0.01:timePoints(end);


qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));   
ddqd = ddqd(:, 1:length(tvec));
save('teacher_traj/qd.mat','qd');
save('teacher_traj/dqd.mat','dqd');
save('teacher_traj/ddqd.mat','ddqd');
save('teacher_traj/tvec.mat','tvec');
