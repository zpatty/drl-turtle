clear
%% Cyclic Trajectory
% Need to make sure that trajectory has the same starting and end point
% For any cyclical trajectory, the starting point is given 1 second to move
% from turtle's home position to that starting pointin the cycle
% the cycle trajectory (i.e 2nd and last element) must be the same
shoulder1 = [180 240 110 240];
shoulder2 = [180 230 150 230];
shoulder3 = [180 240 140 240];
shoulder4 = 180 + (180 - shoulder1);
shoulder5 = 180 + (180 - shoulder2);
shoulder6 = [180 120 220 120];

% time points for a single cycle
timePoints = [0 2 4 6];
tshift = timePoints(2:end) + timePoints(2:end);
timePoints_cycle = [timePoints, tshift(2:end)]

wayPoints = [shoulder1; shoulder2; shoulder3; shoulder4; shoulder5; shoulder6] * pi/180;
wayPoints = [wayPoints(:,1), repmat(wayPoints(:,2:end-1),1,2), wayPoints(:,end)];
% wayPoints = [repmat(wayPoints(:, 1:end-1), 1,2),wayPoints(:, 1)]

tvec = 0:0.01:timePoints_cycle(end);
[qd,dqd,ddqd,ppc] = cubicpolytraj(wayPoints,timePoints_cycle,tvec);
% 
tvec = 0:0.01:timePoints(end);
qd = qd(:, 1:length(tvec));
dqd = dqd(:, 1:length(tvec));
ddqd = ddqd(:, 1:length(tvec));
save('qd.mat','qd');
save('dqd.mat','dqd');
save('ddqd.mat','ddqd');
save('tvec.mat','tvec');